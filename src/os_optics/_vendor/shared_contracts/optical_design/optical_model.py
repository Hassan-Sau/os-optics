"""Project-native optical models: optics a Project declares by their physics instead of by a catalog product.

A Project can carry a library of optical models (``ProjectDocumentV2.optical_models``) with the materials and coatings
they use (``model_materials``, ``model_coatings``). A component that is not a catalog product (``custom_part``) uses one
of them through ``optical_model_id`` and is then simulated, inspected and exchanged exactly like a catalog optic: the
local runtime compiles a declared model with the same code, into the same Engine surfaces, as a catalog product's model.

The model vocabulary is the public optical catalog's own (``SURFACE_SEQUENCE``, ``SURFACE_SET``,
``POLARIZATION_ELEMENT`` and ``NEUTRAL_DENSITY``, with its frames, design geometry and coatings -- including
angle-resolved tables and coating phases), plus seven model kinds a Project can declare directly:

* ``APERTURE_STOP``: an opaque plate with an opening (iris, pinhole, slit) and optionally a central obstruction. It is a
  geometric stop: rays outside the opening are blocked, diffraction at the opening is not modelled.
* ``IDEAL_THIN_LENS``: a paraxial ideal lens of given focal length at one plane (zero thickness, no aberration, no
  optical-path phase).
* ``COMPOUND``: one optical unit made of member models at fixed poses in the unit's own frame (an objective, a relay, a
  telescope). The unit is placed as a whole; every internal surface is simulated.
* ``TABULATED_THIN_LENS``: a thin focusing element whose response is a measured or externally computed table (focal
  length and delivered power per wavelength), with the domain and accuracy it was validated for and the evidence it
  rests on where the author states them (for example a metasurface lens characterized with an external solver); a
  table stated without evidence is an assumed response.
* ``FARADAY_ROTATOR``: a thin nonreciprocal polarization rotator whose response is a table (rotation angle and power
  transmittance per wavelength), with the operating conditions and evidence it rests on where the author states them.
  Its rotation sense is fixed to local +Z, not to the direction of travel: a round trip adds the rotation instead of
  cancelling it.
* ``DIFFRACTION_GRATING``: a thin periodic plane, reflecting or transmitting, whose order directions follow from its period,
  groove orientation and media. Its order powers come from a stated response table with its provenance; without one the
  orders are traced with a power that is not known.
* ``FACET_SOLID``: a solid of a declared material bounded by flat triangular facets (a prism, a corner cube, a CAD part
  exported as facets), with what the facets stand for (exactly the part, or a tessellation of curved surfaces within a
  stated chord deviation) and optional facet groups that are coated, mirrored or blackened. The run traces the facets.
* ``NONLINEAR_CRYSTAL``: a rectangular slab of a declared uniaxial crystal that generates the second harmonic of light
  entering it: its nonlinear coefficient (``d_eff`` or the ``d`` tensor), phase matching (birefringent type I / II, or a
  quasi-phase-matching period), temperature and thermo-optic coefficients. The run traces the slab like any crystal and
  generates the harmonic as a new spectral line of the source.

A declared model may state its power ratings (``power_ratings``, the catalog's ``PowerRatingV1``), which runs compare
with the local power and irradiance on the element.

Materials of declared models are self-contained: a medium id resolves only against ``model_materials`` (never against
the catalog), so the Project hash binds the complete physics of every declared optic.

A declared material may be a uniaxial crystal (``ProjectModelMaterialV1.birefringence``: ordinary and extraordinary index);
a ``SURFACE_SEQUENCE`` or ``SURFACE_SET`` model made of it states the crystal's optic axis in its own frame
(``ProjectOpticalModelV1.optic_axes``), so one crystal material can be cut differently in different elements (a beam
displacer, a waveplate, a polarizing prism).

Hash evolution (``project.py``): every field added here is optional with a ``None`` default, so a Project that uses none
of them keeps its exact bytes and hash.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, StringConstraints, model_validator

from ..base import OmitAbsentFields, StrictContractModel
from ..common.decimal_string import OpticalDecimal, read_decimal
from ..common.text import AuthorName, AuthorText
from ..common.geometry import PoseV1
from ..common.hashes import Sha256Digest
from ..common.identifiers import OpenSciId
from ..common.meshes import MAX_MESH_TRIANGLES, MAX_MESH_VERTICES, mesh_problem
from ..common.ordering import duplicate_keys
from ..design_catalog.enums import FactStatus
from ..design_catalog.material_physics import (
    AIR_MODEL_EDLEN_1953,
    AirModel,
    BulkLossV1,
    CatalogCopyV1,
    HerzbergerDispersionV1,
    IndexReferenceV1,
    PowerSeriesDispersionV1,
    TemperatureSellmeierDispersionV1,
    ThermoOpticV1,
)
from ..design_catalog.optical_catalog import (
    AIR,
    RESERVED_MEDIA,
    VACUUM,
    material_physics_problem,
    DECIMAL_CONTEXT,
    MAX_POWER_RATINGS,
    AngleRangeV1,
    ClearShapeV1,
    CoatingId,
    DesignGeometryV1,
    FramePortV1,
    MaterialId,
    NeutralDensityModelV1,
    OpticalFrameV1,
    PolarizationElementModelV1,
    PowerRatingV1,
    PublicCoatingV1,
    SellmeierDispersionV1,
    ModelFidelity,
    PrescriptionSurfaceV1,
    SimulationApproximation,
    SurfaceSequenceModelV1,
    SurfaceId,
    SurfaceSetModelV1,
    TabulatedDispersionV1,
    TracedBranches,
    WavelengthBandV1,
    check_power_ratings,
)
from .component import ModelId, require_unit_quaternion


MemberId = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Z][A-Z0-9_]{0,15}$"),
    Field(description="Member id within a COMPOUND unit: an uppercase letter, then uppercase letters, digits or '_' (e.g. L1, STOP)."),
]

#: The most members one COMPOUND unit holds.
MAX_COMPOUND_MEMBERS = 64
#: The most members a COMPOUND unit stands for once its member arrays are counted cell by cell.
MAX_COMPOUND_CELLS = 512
#: The most rows one TABULATED_THIN_LENS table holds, and the most validated metrics and tools it lists.
MAX_THIN_LENS_ROWS = 4096
MAX_VALIDATED_METRICS = 64
MAX_EVIDENCE_TOOLS = 16
#: A POINTS_ONLY table (or a one-row table) answers a wavelength within this relative distance of one of its rows.
THIN_LENS_POINT_TOLERANCE = Decimal("0.005")
#: The most rows one FARADAY_ROTATOR table holds.
MAX_FARADAY_ROWS = 4096
#: The angle of incidence a plane is no longer crossed at (deg).
GRAZING_AOI_DEG = Decimal(90)
#: The most orders one DIFFRACTION_GRATING declares or spans, and the most rows of its response tables.
MAX_GRATING_ORDERS = 65
MAX_GRATING_RESPONSE_ROWS = 8192


class DesignIntentV1(StrEnum):
    """What the Project is designed for.

    ``LAB_ASSEMBLY``: an optical layout to be built from individually mounted components (mounts, posts, a breadboard),
    whose mechanical realization OpenSci configures for a complete-system purchase. A Project that states no intent has
    this meaning.
    ``OPTICAL_SYSTEM``: an optical design to simulate, inspect and export without laboratory mechanics; it needs no
    mounts, posts, holders or breadboard. The physical validity of its optical bodies is still checked.
    """

    LAB_ASSEMBLY = "LAB_ASSEMBLY"
    OPTICAL_SYSTEM = "OPTICAL_SYSTEM"


def _decimal(value: str, label: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - OpticalDecimal already guarantees a decimal
        raise ValueError(f"{label} is not a decimal: {value!r}") from exc


def _zero(value: object) -> bool:
    """Whether a decimal as written (any spelling ``OpticalDecimal`` reads) is zero; anything else is not."""

    number = read_decimal(value)
    return number is not None and number.is_zero()


def _positive(value: str, label: str) -> Decimal:
    number = _decimal(value, label)
    if not number > 0:
        raise ValueError(f"{label} must be positive")
    return number


# ------------------------------------------------------------------------------------------------- materials


class ConstantIndexDispersionV1(StrictContractModel):
    """A refractive index that does not vary with wavelength inside the material's valid band."""

    kind: Literal["CONSTANT_INDEX"] = "CONSTANT_INDEX"
    refractive_index: OpticalDecimal

    @model_validator(mode="after")
    def _positive_index(self) -> ConstantIndexDispersionV1:
        _positive(self.refractive_index, "refractive_index")
        return self


ProjectDispersionV1 = Annotated[
    SellmeierDispersionV1 | TabulatedDispersionV1 | ConstantIndexDispersionV1 | PowerSeriesDispersionV1 | HerzbergerDispersionV1
    | TemperatureSellmeierDispersionV1,
    Field(discriminator="kind"),
]


class PropagationEnvironmentV1(StrictContractModel):
    """The medium a Project's optics operate in: air (``AIR``, everywhere outside the elements and in every ``AIR`` gap) at
    ``temperature_c`` and ``pressure_pa``, its index by ``air_model`` (``design_catalog.material_physics.air_index``); air at
    zero pressure is vacuum. ``temperature_c`` is also the temperature of the materials (a material with thermo-optic data
    is evaluated there; one without is evaluated as its source states it, at the source's temperature). A gap of a declared
    model may state ``VACUUM`` instead of ``AIR`` (an evacuated space whatever the environment).

    A Project that states no environment operates in air at 20 C and 101325 Pa (``DEFAULT_ENVIRONMENT``), the reference
    air of the glass catalogues. The environment is physics, not notation: wavelengths are vacuum wavelengths in every
    environment."""

    temperature_c: OpticalDecimal = "20"
    pressure_pa: OpticalDecimal = "101325"
    air_model: AirModel = AIR_MODEL_EDLEN_1953

    @model_validator(mode="after")
    def _physical(self) -> PropagationEnvironmentV1:
        if not Decimal(self.temperature_c) > Decimal("-273.15"):
            raise ValueError("temperature_c must be above absolute zero")
        if Decimal(self.pressure_pa) < 0:
            raise ValueError("pressure_pa cannot be negative")
        return self

    @property
    def is_vacuum(self) -> bool:
        return Decimal(self.pressure_pa) == 0


#: The environment of a Project that states none.
DEFAULT_ENVIRONMENT = PropagationEnvironmentV1()


#: Evenly spaced wavelengths (band ends included) at which a declared material's index is checked to be at least 1.
INDEX_CHECK_SAMPLES = 65


def _dispersion_problem(dispersion: object, low: Decimal, high: Decimal) -> str | None:
    """Why ``dispersion`` is not a physical refractive index over ``[low, high]`` nm, or None.

    The index must be real and at least 1 everywhere in the band: a Sellmeier formula has no pole inside the band (a term
    ``b l^2 / (l^2 - c)`` with ``b != 0`` and ``low^2 <= c <= high^2``, ``l`` in um) and ``n^2 >= 1`` at
    ``INDEX_CHECK_SAMPLES`` evenly spaced wavelengths; a constant index is at least 1; every tabulated index is at least 1
    (the monotone PCHIP interpolation stays between neighbouring table values)."""

    with localcontext(DECIMAL_CONTEXT):
        if isinstance(dispersion, ConstantIndexDispersionV1):
            index = Decimal(dispersion.refractive_index)
            return None if index >= 1 else f"refractive index {dispersion.refractive_index} is below 1"
        if isinstance(dispersion, TabulatedDispersionV1):
            below = next((n for n in dispersion.refractive_index if Decimal(n) < 1), None)
            return None if below is None else f"tabulated refractive index {below} is below 1"
        if isinstance(dispersion, (PowerSeriesDispersionV1, HerzbergerDispersionV1, TemperatureSellmeierDispersionV1)):
            for k in range(INDEX_CHECK_SAMPLES):
                lam = (low + (high - low) * k / (INDEX_CHECK_SAMPLES - 1)) / 1000
                if isinstance(dispersion, PowerSeriesDispersionV1):
                    n_sq = dispersion.n_squared_exact(lam)
                elif isinstance(dispersion, HerzbergerDispersionV1):
                    n = dispersion.index_exact(lam)
                    n_sq = n * n if n > 0 else Decimal(-1)
                else:
                    terms = dispersion.terms_at(Decimal(dispersion.reference_temperature_k))
                    lam_sq = lam * lam
                    if any(c != 0 and abs(lam_sq - c) < Decimal("1e-12") for _s, c in terms):
                        return f"its temperature Sellmeier law has a pole at {lam * 1000:.7g} nm"
                    n_sq = 1 + sum(s * lam_sq / (lam_sq - c) for s, c in terms)
                if n_sq < 1:
                    return (f"its {dispersion.kind} law gives n^2 = {n_sq:.6g} at {lam * 1000:.7g} nm: the refractive index must be real "
                            "and at least 1 throughout the valid band")
            return None
        if not isinstance(dispersion, SellmeierDispersionV1):  # pragma: no cover - the union is closed
            return None
        terms = [(Decimal(b), Decimal(c)) for b, c in zip(dispersion.b, dispersion.c_um2, strict=True)]
        low_sq, high_sq = (low / 1000) ** 2, (high / 1000) ** 2
        for b, c in terms:
            if b != 0 and low_sq <= c <= high_sq:
                return (f"its Sellmeier formula has a pole at {c.sqrt() * 1000:.6g} nm (c = {c.normalize():f} um^2), "
                        "inside the valid band")
        for k in range(INDEX_CHECK_SAMPLES):
            wavelength = low + (high - low) * k / (INDEX_CHECK_SAMPLES - 1)
            lam_sq = (wavelength / 1000) ** 2
            n_sq = 1 + sum(b * lam_sq / (lam_sq - c) for b, c in terms)
            if n_sq < 1:
                return (f"its Sellmeier formula gives n^2 = {n_sq:.6g} at {wavelength:.7g} nm: the refractive "
                        "index must be real and at least 1 throughout the valid band")
    return None


class UniaxialBirefringenceV1(StrictContractModel):
    """A uniaxial crystal (calcite, crystal quartz without its optical activity, YVO4, alpha-BBO, ...): the material's
    ``dispersion`` is its ordinary index ``n_o`` (light polarized perpendicular to the optic axis) and ``extraordinary`` its
    extraordinary index ``n_e`` (light polarized along the optic axis), in the same forms (Sellmeier, tabulated, constant)
    and over the same valid band. The optic axis is not a property of the material: every element made of the crystal
    states its direction (``ProjectOpticalModelV1.optic_axes``). Light entering the crystal splits into an ordinary and an
    extraordinary beam; the extraordinary ray walks off its wave normal (the Engine's uniaxial crystal model)."""

    kind: Literal["UNIAXIAL"] = "UNIAXIAL"
    extraordinary: ProjectDispersionV1


#: The most rows of a gradient constant tabulated over wavelength.
MAX_GRADIENT_ROWS = 64


class GradientRowV1(StrictContractModel):
    """A radial gradient constant ``g`` (1/mm) at one vacuum wavelength."""

    wavelength_nm: OpticalDecimal
    gradient_constant_per_mm: OpticalDecimal

    @model_validator(mode="after")
    def _positive_values(self) -> GradientRowV1:
        _positive(self.wavelength_nm, "wavelength_nm")
        _positive(self.gradient_constant_per_mm, "gradient_constant_per_mm")
        return self


class RadialGradientIndexV1(StrictContractModel):
    """A radial gradient-index medium (a GRIN rod lens): the index falls off from the axis as
    ``n(r, lambda)^2 = n0(lambda)^2 (1 - (g(lambda) r)^2)``, ``r`` the distance from the element's local Z axis, ``n0`` the
    material's ``dispersion`` (the index on the axis) and ``g`` the gradient constant: ``gradient_constant_per_mm`` at every
    wavelength, or ``gradient_rows`` (linear in wavelength between rows, covering the material's band). In this profile
    every ray, meridional or skew, follows ``x(z) = x0 cos(W z) + (x0' / W) sin(W z)`` (and the same for ``y``) with
    ``W = n0 g / beta`` and ``beta = n dz/ds`` the ray's invariant, so a paraxial ray's period along the axis is
    ``2 pi / g`` (the pitch ``P = g L / (2 pi)`` of a rod of length ``L``). The profile ``n(r) = n0 (1 - A r^2 / 2)`` with
    ``sqrt(A) = g``, in which rod lenses are often specified, agrees with it to order ``r^2`` (they differ by ``n0 (g r)^4 / 8``
    and higher terms)."""

    kind: Literal["RADIAL_QUADRATIC"] = "RADIAL_QUADRATIC"
    gradient_constant_per_mm: OpticalDecimal | None = None
    gradient_rows: tuple[GradientRowV1, ...] | None = Field(None, min_length=2, max_length=MAX_GRADIENT_ROWS)

    @model_validator(mode="after")
    def _one_law(self) -> RadialGradientIndexV1:
        if (self.gradient_constant_per_mm is None) == (self.gradient_rows is None):
            raise ValueError("state exactly one of gradient_constant_per_mm and gradient_rows")
        if self.gradient_constant_per_mm is not None:
            _positive(self.gradient_constant_per_mm, "gradient_constant_per_mm")
        rows = [Decimal(r.wavelength_nm) for r in self.gradient_rows or ()]
        if any(b <= a for a, b in zip(rows, rows[1:])):
            raise ValueError("gradient_rows must be in strictly increasing wavelength")
        return self

    def gradient_at(self, wavelength_nm: Decimal) -> Decimal:
        """``g`` (1/mm) at a vacuum wavelength inside the rows (linear between rows)."""

        if self.gradient_constant_per_mm is not None:
            return Decimal(self.gradient_constant_per_mm)
        rows = [(Decimal(r.wavelength_nm), Decimal(r.gradient_constant_per_mm)) for r in self.gradient_rows or ()]
        with localcontext(DECIMAL_CONTEXT):
            for (w0, g0), (w1, g1) in zip(rows, rows[1:]):
                if w0 <= wavelength_nm <= w1:
                    return g0 + (g1 - g0) * (wavelength_nm - w0) / (w1 - w0)
        raise ValueError(f"{wavelength_nm} nm is outside the gradient rows")

    def largest(self) -> Decimal:
        """The largest gradient constant it states (1/mm)."""

        if self.gradient_constant_per_mm is not None:
            return Decimal(self.gradient_constant_per_mm)
        return max(Decimal(r.gradient_constant_per_mm) for r in self.gradient_rows or ())


#: The most terms of an axial gradient (``a1 z``, ``a2 z^2``, ``a3 z^3``).
MAX_AXIAL_GRADIENT_TERMS = 3


class AxialGradientIndexV1(StrictContractModel):
    """An axial gradient-index medium: the index varies along the element's local Z axis and not across it,
    ``n(z, lambda) = n0(lambda) + a1 z + a2 z^2 + a3 z^3``, ``n0`` the material's ``dispersion`` (the index at ``z = 0``) and
    ``index_terms`` the coefficients ``a1`` (1/mm), ``a2`` (1/mm^2), ``a3`` (1/mm^3) in that order (fewer terms: the rest 0),
    the same at every wavelength. ``z`` (mm) is measured along local Z from the medium's entrance: the vertex of the surface
    in front of it (a ``SURFACE_SEQUENCE`` model), or the point where the body's face toward local -Z crosses local Z (a
    ``SURFACE_SET`` model with a round body). The index does not vary across the axis, so a ray keeps its transverse optical
    direction ``n t_perp`` and bends in the plane of the axis and its direction (``x(z) = x0 + k_x int dz / sqrt(n^2 - k^2)``)."""

    kind: Literal["AXIAL_POLYNOMIAL"] = "AXIAL_POLYNOMIAL"
    index_terms: tuple[OpticalDecimal, ...] = Field(min_length=1, max_length=MAX_AXIAL_GRADIENT_TERMS)

    @model_validator(mode="after")
    def _terms(self) -> AxialGradientIndexV1:
        terms = [_decimal(t, "index_terms") for t in self.index_terms]
        if all(t == 0 for t in terms):
            raise ValueError("index_terms are all 0: a medium without a gradient is homogeneous (omit axial_gradient)")
        return self

    def terms(self) -> tuple[Decimal, Decimal, Decimal]:
        """``(a1, a2, a3)``."""

        values = [Decimal(t) for t in self.index_terms]
        return tuple((values + [Decimal(0)] * MAX_AXIAL_GRADIENT_TERMS)[:MAX_AXIAL_GRADIENT_TERMS])  # type: ignore[return-value]

    def offset_at(self, z_mm: Decimal) -> Decimal:
        """``a1 z + a2 z^2 + a3 z^3`` (exact decimals)."""

        a1, a2, a3 = self.terms()
        with localcontext(DECIMAL_CONTEXT):
            return ((a3 * z_mm + a2) * z_mm + a1) * z_mm

    def least_offset(self, z0_mm: Decimal, z1_mm: Decimal) -> Decimal:
        """The least of ``a1 z + a2 z^2 + a3 z^3`` for ``z`` between ``z0_mm`` and ``z1_mm``: at an end or where its derivative
        ``a1 + 2 a2 z + 3 a3 z^2`` vanishes."""

        a1, a2, a3 = self.terms()
        lo, hi = min(z0_mm, z1_mm), max(z0_mm, z1_mm)
        candidates = [lo, hi]
        with localcontext(DECIMAL_CONTEXT):
            if a3 != 0:
                disc = 4 * a2 * a2 - 12 * a1 * a3
                if disc >= 0:
                    root = disc.sqrt()
                    candidates += [(-2 * a2 + root) / (6 * a3), (-2 * a2 - root) / (6 * a3)]
            elif a2 != 0:
                candidates.append(-a1 / (2 * a2))
            return min(self.offset_at(z) for z in candidates if lo <= z <= hi)


class ProjectModelMaterialV1(OmitAbsentFields, StrictContractModel):
    """A refractive material declared in the Project for its optical models.

    ``dispersion`` uses a catalog index law or a constant index. ``valid_band_nm`` gives the vacuum wavelength interval
    over which the refractive index must be real and at least 1. A Sellmeier formula must have no pole inside that band.
    ``absorption_coefficient_per_mm`` specifies bulk absorption ``alpha`` (power decays as ``exp(-alpha * path_mm)``).
    An explicit zero assumes lossless propagation with respect to absorption. If neither a scalar nor spectral bulk loss
    is given, the run uses zero absorption to calculate an upper bound on transmitted power.
    ``scattering_coefficient_per_mm``: bulk scattering ``mu_s`` (a traced beam's power decays as ``exp(-(alpha + mu_s) *
    path_mm)``; the scattered light is booked and not followed, so the planes its light reaches read a lower bound);
    ``anisotropy_g`` (``-1 < g < 1``, with ``mu_s`` only): the mean cosine of the scattering angle, reported.
    ``birefringence``: the material is a uniaxial crystal (``UniaxialBirefringenceV1``): ``dispersion`` is then its
    ordinary index and ``birefringence.extraordinary`` its extraordinary index, held to the same rules; absent, the material
    is isotropic. ``gradient``: the material is a radial gradient-index medium (``RadialGradientIndexV1``): ``dispersion`` is
    then its index on the axis; it is not a crystal as well. ``axial_gradient``: the material is an axial gradient-index
    medium (``AxialGradientIndexV1``): ``dispersion`` is then its index at the medium's entrance; it is neither a crystal nor
    a radial gradient-index medium as well.

    The material physics a catalog material carries (``design_catalog.material_physics``; all optional, copied verbatim from
    the catalog by ``catalog_copy``): ``index_reference`` -- the wavelength and index convention of the law (absent: vacuum
    wavelength and absolute index, the Project's own convention); ``bulk_loss`` -- measured spectral bulk loss (it replaces
    ``absorption_coefficient_per_mm`` on its wavelengths; the two are not stated together); ``thermo_optic`` -- the change
    of the index with temperature; ``copied_from`` -- the catalog release, the material id chosen in it and the edition of
    its source (the stable id names the maker's grade). The copy is complete: a
    Project reloads without the catalog.
    """

    # Serialized only when given: a document without it keeps its earlier payload and hash.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("gradient", "axial_gradient", "index_reference", "bulk_loss", "thermo_optic",
                                                  "copied_from")

    material_id: MaterialId
    display_name: AuthorName
    dispersion: ProjectDispersionV1
    valid_band_nm: WavelengthBandV1
    absorption_coefficient_per_mm: OpticalDecimal | None = None
    basis: AuthorText | None = None
    scattering_coefficient_per_mm: OpticalDecimal | None = None
    anisotropy_g: OpticalDecimal | None = None
    birefringence: UniaxialBirefringenceV1 | None = None
    gradient: RadialGradientIndexV1 | None = None
    axial_gradient: AxialGradientIndexV1 | None = None
    index_reference: IndexReferenceV1 | None = None
    bulk_loss: BulkLossV1 | None = None
    thermo_optic: ThermoOpticV1 | None = None
    copied_from: CatalogCopyV1 | None = None

    @model_validator(mode="before")
    @classmethod
    def _flat_axial_gradient_dropped(cls, data: Any) -> Any:
        """An ``axial_gradient`` whose ``index_terms`` are all zero is a homogeneous medium: dropped before validation."""

        if not isinstance(data, dict) or not isinstance(data.get("axial_gradient"), dict):
            return data
        terms = data["axial_gradient"].get("index_terms")
        if isinstance(terms, (list, tuple)) and terms and all(_zero(t) for t in terms):
            return {k: v for k, v in data.items() if k != "axial_gradient"}
        return data

    @property
    def is_crystal(self) -> bool:
        """Whether the material is a uniaxial crystal (``birefringence`` stated)."""

        return self.birefringence is not None

    @property
    def is_gradient(self) -> bool:
        """Whether the material is a gradient-index medium (``gradient`` or ``axial_gradient`` stated)."""

        return self.gradient is not None or self.axial_gradient is not None

    @property
    def is_axial_gradient(self) -> bool:
        """Whether the material is an axial gradient-index medium (``axial_gradient`` stated)."""

        return self.axial_gradient is not None

    @model_validator(mode="after")
    def _consistent(self) -> ProjectModelMaterialV1:
        if self.material_id in RESERVED_MEDIA:
            raise ValueError(f"{self.material_id} is a reserved medium, not a declared material")
        if not Decimal(self.valid_band_nm.min_nm) < Decimal(self.valid_band_nm.max_nm):
            raise ValueError(f"material {self.material_id}: the valid band must be an interval")
        laws = [("", self.dispersion)]
        if self.birefringence is not None:
            laws = [("ordinary index: ", self.dispersion), ("extraordinary index: ", self.birefringence.extraordinary)]
        low, high = Decimal(self.valid_band_nm.min_nm), Decimal(self.valid_band_nm.max_nm)
        problem = material_physics_problem(self)
        if problem is not None:
            raise ValueError(f"material {self.material_id}: {problem}")
        if self.bulk_loss is not None and self.absorption_coefficient_per_mm is not None:
            raise ValueError(f"material {self.material_id}: state bulk_loss (spectral) or absorption_coefficient_per_mm, not both")
        if self.is_gradient and (self.index_reference is not None or self.thermo_optic is not None):
            raise ValueError(f"material {self.material_id}: a gradient-index medium states its index in the Project's convention")
        if self.index_reference is not None:  # (the index checks below run on the law's own wavelength)
            with localcontext(DECIMAL_CONTEXT):
                low, high = self.index_reference.law_wavelength_exact(low), self.index_reference.law_wavelength_exact(high)
        if self.absorption_coefficient_per_mm is not None and _decimal(self.absorption_coefficient_per_mm, "absorption") < 0:
            raise ValueError(f"material {self.material_id}: absorption_coefficient_per_mm cannot be negative")
        if self.scattering_coefficient_per_mm is not None and _decimal(self.scattering_coefficient_per_mm, "scattering") < 0:
            raise ValueError(f"material {self.material_id}: scattering_coefficient_per_mm cannot be negative")
        if self.anisotropy_g is not None:
            if self.scattering_coefficient_per_mm is None:
                raise ValueError(f"material {self.material_id}: anisotropy_g describes bulk scattering: state scattering_coefficient_per_mm with it")
            if not -1 < _decimal(self.anisotropy_g, "anisotropy_g") < 1:
                raise ValueError(f"material {self.material_id}: anisotropy_g lies strictly between -1 and 1")
        for label, dispersion in laws:
            problem = _dispersion_problem(dispersion, low, high)
            if problem is not None:
                raise ValueError(f"material {self.material_id}: {label}{problem}")
        if self.axial_gradient is not None:
            if self.birefringence is not None or self.gradient is not None:
                raise ValueError(f"material {self.material_id}: state one of axial_gradient, gradient and birefringence (an axial "
                                 "gradient-index medium is isotropic and has no radial gradient)")
        if self.gradient is not None:
            if self.birefringence is not None:
                raise ValueError(f"material {self.material_id}: a gradient-index medium is isotropic (state gradient or birefringence)")
            rows = self.gradient.gradient_rows
            if rows is not None and (low < Decimal(rows[0].wavelength_nm) or high > Decimal(rows[-1].wavelength_nm)):
                raise ValueError(f"material {self.material_id}: the gradient rows must cover the valid band")
        return self


# ------------------------------------------------------------------------------------------ declared kinds


#: The most nodes of a sampled complex-transmission map (256 x 256), the most phase terms of a sample, the most Zernike terms of one.
MAX_SAMPLE_MAP_NODES = 65536
MAX_SAMPLE_PHASE_TERMS = 16
MAX_SAMPLE_ZERNIKE_TERMS = 231
#: The last FRINGE Zernike index.
SAMPLE_FRINGE_MAX_INDEX = 37


class SamplePhaseTermV1(StrictContractModel):
    """One analytic phase of a thin sample, in rad over its plane (``x``, ``y`` in mm in the element frame):

    * ``LINEAR_RAMP``: ``phi = g_x x + g_y y`` (``gradient_rad_per_mm``); it steers light like a prism (the local grating equation);
    * ``STEP``: ``phi = phase_rad`` where ``edge_normal . (x, y) >= edge_offset_mm``, 0 elsewhere (a step across a straight edge;
      ``edge_normal`` a unit direction);
    * ``ZERNIKE``: ``phi = sum value_rad Z_index`` over ``normalization_radius_mm`` (``NOLL`` with ``RMS_ORTHONORMAL``,
      ``FRINGE`` with ``PEAK_UNIT``), the opening inside the normalization radius."""

    kind: Literal["LINEAR_RAMP", "STEP", "ZERNIKE"]
    gradient_rad_per_mm: tuple[OpticalDecimal, OpticalDecimal] | None = None
    phase_rad: OpticalDecimal | None = None
    edge_normal: tuple[OpticalDecimal, OpticalDecimal] | None = None
    edge_offset_mm: OpticalDecimal | None = None
    indexing: Literal["NOLL", "FRINGE"] | None = None
    normalization: Literal["RMS_ORTHONORMAL", "PEAK_UNIT"] | None = None
    normalization_radius_mm: OpticalDecimal | None = None
    zernike_indices: tuple[int, ...] | None = Field(None, min_length=1, max_length=MAX_SAMPLE_ZERNIKE_TERMS)
    zernike_values_rad: tuple[OpticalDecimal, ...] | None = Field(None, min_length=1, max_length=MAX_SAMPLE_ZERNIKE_TERMS)

    @model_validator(mode="after")
    def _consistent(self) -> SamplePhaseTermV1:
        stated = {name for name in ("gradient_rad_per_mm", "phase_rad", "edge_normal", "edge_offset_mm", "indexing", "normalization",
                                    "normalization_radius_mm", "zernike_indices", "zernike_values_rad") if getattr(self, name) is not None}
        allowed = {"LINEAR_RAMP": ({"gradient_rad_per_mm"}, set()),
                   "STEP": ({"phase_rad", "edge_normal"}, {"edge_offset_mm"}),
                   "ZERNIKE": ({"indexing", "normalization", "normalization_radius_mm", "zernike_indices", "zernike_values_rad"}, set())}[self.kind]
        required, optional = allowed
        if not required <= stated or not stated <= required | optional:
            raise ValueError(f"a {self.kind} phase states {', '.join(sorted(required))}" + (f" (optionally {', '.join(sorted(optional))})" if optional else ""))
        if self.kind == "STEP":
            a, b = (_decimal(v, "edge_normal") for v in self.edge_normal)  # type: ignore[union-attr]
            if abs(a * a + b * b - 1) > Decimal("1e-6"):
                raise ValueError("a STEP's edge_normal is a unit direction")
        if self.kind == "ZERNIKE":
            _positive(self.normalization_radius_mm, "normalization_radius_mm")
            if (self.indexing == "NOLL") != (self.normalization == "RMS_ORTHONORMAL"):
                raise ValueError("NOLL indexing pairs with RMS_ORTHONORMAL, FRINGE with PEAK_UNIT")
            if len(self.zernike_indices) != len(self.zernike_values_rad):  # type: ignore[arg-type]
                raise ValueError("zernike_indices and zernike_values_rad have the same length")
            top = SAMPLE_FRINGE_MAX_INDEX if self.indexing == "FRINGE" else MAX_SAMPLE_ZERNIKE_TERMS
            if any(not 1 <= i <= top for i in self.zernike_indices) or len(set(self.zernike_indices)) != len(self.zernike_indices):  # type: ignore[union-attr]
                raise ValueError(f"zernike_indices are distinct indices from 1 to {top}")
        return self


class SampleMapV1(StrictContractModel):
    """A complex transmission sampled on a regular grid over a thin sample: ``nodes_x`` x ``nodes_y`` nodes, ``pitch_mm`` apart,
    centred on ``center_mm`` (node ``(i, j)`` at ``x = center_x + (i - (nodes_x - 1) / 2) pitch_x``, likewise y), listed row by
    row (``k = j nodes_x + i``). ``amplitude`` (0 .. 1; absent: 1) and ``phase_rad`` (the unwrapped phase; absent: 0) are
    interpolated bilinearly between the nodes (``interpolation``); structure finer than the pitch is not represented. The map
    covers the opening."""

    nodes_x: int = Field(ge=2, le=MAX_SAMPLE_MAP_NODES)
    nodes_y: int = Field(ge=2, le=MAX_SAMPLE_MAP_NODES)
    pitch_mm: tuple[OpticalDecimal, OpticalDecimal]
    center_mm: tuple[OpticalDecimal, OpticalDecimal] | None = None
    amplitude: tuple[OpticalDecimal, ...] | None = Field(None, min_length=4, max_length=MAX_SAMPLE_MAP_NODES)
    phase_rad: tuple[OpticalDecimal, ...] | None = Field(None, min_length=4, max_length=MAX_SAMPLE_MAP_NODES)
    interpolation: Literal["BILINEAR"] = "BILINEAR"

    @model_validator(mode="after")
    def _consistent(self) -> SampleMapV1:
        count = self.nodes_x * self.nodes_y
        if count > MAX_SAMPLE_MAP_NODES:
            raise ValueError(f"a sample map holds at most {MAX_SAMPLE_MAP_NODES} nodes")
        for value in self.pitch_mm:
            _positive(value, "pitch_mm")
        if self.amplitude is None and self.phase_rad is None:
            raise ValueError("a sample map states amplitude, phase_rad or both")
        for name in ("amplitude", "phase_rad"):
            values = getattr(self, name)
            if values is not None and len(values) != count:
                raise ValueError(f"{name} holds nodes_x x nodes_y = {count} values")
        if self.amplitude is not None and any(not 0 <= _decimal(v, "amplitude") <= 1 for v in self.amplitude):
            raise ValueError("amplitude values lie in [0, 1] (a passive sample)")
        return self


class ComplexTransmissionSampleV1(StrictContractModel):
    """A thin sample in the opening of an ``APERTURE_STOP``: its complex transmission ``t = A exp(i phi)`` over the opening, with
    ``A = amplitude x`` the map's amplitude and ``phi = phase_rad +`` the ``phase_terms`` ``+`` the map's phase (rad; ``x``, ``y``
    in the element frame). ``basis``: where the values come from (a measurement, a calculation).

    Model (thin, scalar, geometric): each ray crossing the opening gains the optical path ``phi / k0`` and the tangential
    wavevector ``grad phi`` and keeps its polarization; the beam's power is multiplied by the Gaussian-weighted mean of ``A^2`` over
    its footprint; a beam whose phase is not a plane across its footprint goes on as rays, its Gaussian quantities unavailable.
    Diffraction by the sample's structure, polarization-resolved maps and birefringence are not part of the model."""

    amplitude: OpticalDecimal | None = None
    phase_rad: OpticalDecimal | None = None
    phase_terms: tuple[SamplePhaseTermV1, ...] | None = Field(None, min_length=1, max_length=MAX_SAMPLE_PHASE_TERMS)
    sampled_map: SampleMapV1 | None = None
    basis: AuthorText | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ComplexTransmissionSampleV1:
        if self.amplitude is not None and not 0 <= _decimal(self.amplitude, "amplitude") <= 1:
            raise ValueError("a sample's amplitude lies in [0, 1] (a passive sample)")
        if self.phase_rad is not None:
            _decimal(self.phase_rad, "phase_rad")
        if self.amplitude is None and self.phase_rad is None and self.phase_terms is None and self.sampled_map is None:
            raise ValueError("a sample states its amplitude, phase, phase terms or map")
        return self


class ApertureStopModelV1(StrictContractModel):
    """An opaque plate in the plane ``z = 0`` of the element frame (normal local +Z) with one opening; the component's (or
    a compound member's) pose places that plane, and ``plane_z_mm`` is ``0``.

    The opening is exactly one of ``opening_radius_mm`` (a circle: iris, pinhole) and ``opening_half_widths_mm`` (a
    rectangle along local X / Y: a slit), centred on the axis or at ``opening_center_mm``; ``opening_shape``
    (``ClearShapeV1``) makes it the ellipse inscribed in that rectangle, that circle cut by a flat (a D shape) or a polygon
    (a triangle, a hexagon, any simple polygon) inside it. ``obstruction_radius_mm`` adds
    an opaque central disc (circular openings only; centred on the opening), leaving an annulus. The plate's outline is
    exactly one of ``edge_radius_mm`` and ``edge_half_widths_mm``; light beyond it passes by. A geometric stop: rays
    inside the opening pass unchanged, rays on the plate are blocked; diffraction is not modelled. ``sample``
    (``ComplexTransmissionSampleV1``): the opening holds a thin sample whose complex transmission the light crossing it takes.
    """

    model_kind: Literal["APERTURE_STOP"] = "APERTURE_STOP"
    model_version: Literal["1"] = "1"
    opening_radius_mm: OpticalDecimal | None = None
    opening_half_widths_mm: tuple[OpticalDecimal, OpticalDecimal] | None = None
    opening_center_mm: tuple[OpticalDecimal, OpticalDecimal] | None = None
    opening_shape: ClearShapeV1 | None = None
    obstruction_radius_mm: OpticalDecimal | None = None
    edge_radius_mm: OpticalDecimal | None = None
    edge_half_widths_mm: tuple[OpticalDecimal, OpticalDecimal] | None = None
    plane_z_mm: OpticalDecimal = "0"
    status: FactStatus
    sample: ComplexTransmissionSampleV1 | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ApertureStopModelV1:
        if Decimal(self.plane_z_mm) != 0:
            raise ValueError("an aperture stop lies in its frame's z = 0 plane: plane_z_mm is 0 (place the component at the plane)")
        if (self.opening_radius_mm is None) == (self.opening_half_widths_mm is None):
            raise ValueError("an aperture stop has exactly one of opening_radius_mm and opening_half_widths_mm")
        if (self.edge_radius_mm is None) == (self.edge_half_widths_mm is None):
            raise ValueError("an aperture stop has exactly one of edge_radius_mm and edge_half_widths_mm")
        cx, cy = (abs(Decimal(v)) for v in self.opening_center_mm) if self.opening_center_mm is not None else (Decimal(0), Decimal(0))
        if self.opening_shape is not None:
            if self.obstruction_radius_mm is not None:
                raise ValueError("a shaped opening has no central obstruction")
            radius = _positive(self.opening_radius_mm, "opening_radius_mm") if self.opening_radius_mm is not None else None
            half_widths = (None if self.opening_half_widths_mm is None
                           else tuple(_positive(v, "opening_half_widths_mm") for v in self.opening_half_widths_mm))
            self.opening_shape.check_within(radius, half_widths, "the opening")  # type: ignore[arg-type]
        if self.opening_radius_mm is not None:
            radius = _positive(self.opening_radius_mm, "opening_radius_mm")
            if self.obstruction_radius_mm is not None and not _positive(self.obstruction_radius_mm, "obstruction_radius_mm") < radius:
                raise ValueError("obstruction_radius_mm must be smaller than opening_radius_mm")
            # farthest extent of the opening along X / Y and from the axis
            extent_x, extent_y = cx + radius, cy + radius
            reach_sq = None
            reach_circle = (cx * cx + cy * cy, radius)
        else:
            if self.obstruction_radius_mm is not None:
                raise ValueError("a central obstruction needs a circular opening")
            hx, hy = (_positive(v, "opening_half_widths_mm") for v in self.opening_half_widths_mm)  # type: ignore[union-attr]
            extent_x, extent_y = cx + hx, cy + hy
            reach_sq = extent_x * extent_x + extent_y * extent_y
            reach_circle = None
        if self.edge_radius_mm is not None:
            edge = _positive(self.edge_radius_mm, "edge_radius_mm")
            if reach_circle is not None:
                centre_sq, radius = reach_circle
                inside = centre_sq <= (edge - radius) ** 2 and radius <= edge
            else:
                inside = reach_sq <= edge * edge  # type: ignore[operator]
            if not inside:
                raise ValueError("the opening must lie inside the plate's edge radius")
        else:
            ex, ey = (_positive(v, "edge_half_widths_mm") for v in self.edge_half_widths_mm)  # type: ignore[union-attr]
            if not (extent_x <= ex and extent_y <= ey):
                raise ValueError("the opening must lie inside the plate's edge half widths")
        return self


class IdealThinLensModelV1(StrictContractModel):
    """A paraxial ideal lens in the plane ``z = 0`` of the element frame (axis local +Z); the component's (or a compound
    member's) pose places that plane, and ``plane_z_mm`` is ``0``.

    Every ray is deflected as by a perfect lens of focal length ``focal_length_mm`` (negative: diverging) with no
    thickness, aberration or dispersion; ``transmittance`` is its power transmittance. The lens adds no optical-path
    phase, so phase and interference results through it are not defined. Light outside ``clear_radius_mm`` and inside
    ``edge_radius_mm`` is blocked; beyond the edge it passes by.
    """

    model_kind: Literal["IDEAL_THIN_LENS"] = "IDEAL_THIN_LENS"
    model_version: Literal["1"] = "1"
    focal_length_mm: OpticalDecimal
    clear_radius_mm: OpticalDecimal
    edge_radius_mm: OpticalDecimal
    transmittance: OpticalDecimal = "1"
    plane_z_mm: OpticalDecimal = "0"
    valid_band_nm: WavelengthBandV1
    status: FactStatus

    @model_validator(mode="after")
    def _consistent(self) -> IdealThinLensModelV1:
        if Decimal(self.plane_z_mm) != 0:
            raise ValueError("an ideal thin lens lies in its frame's z = 0 plane: plane_z_mm is 0 (place the component at the plane)")
        if _decimal(self.focal_length_mm, "focal_length_mm") == 0:
            raise ValueError("focal_length_mm cannot be zero")
        if not _positive(self.clear_radius_mm, "clear_radius_mm") <= _positive(self.edge_radius_mm, "edge_radius_mm"):
            raise ValueError("clear_radius_mm cannot exceed edge_radius_mm")
        if not 0 < _decimal(self.transmittance, "transmittance") <= 1:
            raise ValueError("transmittance must lie in (0, 1]")
        return self


class ThinLensResponseRowV1(StrictContractModel):
    """One row of a tabulated thin lens: at vacuum wavelength ``wavelength_nm``, focal length ``focal_length_mm``
    (negative: diverging) and ``transmittance``, the fraction of the incident power delivered into the modelled focused
    beam, in ``(0, 1]``."""

    wavelength_nm: OpticalDecimal
    focal_length_mm: OpticalDecimal
    transmittance: OpticalDecimal

    @model_validator(mode="after")
    def _consistent(self) -> ThinLensResponseRowV1:
        _positive(self.wavelength_nm, "wavelength_nm")
        if _decimal(self.focal_length_mm, "focal_length_mm") == 0:
            raise ValueError("focal_length_mm cannot be zero")
        if not 0 < _decimal(self.transmittance, "transmittance") <= 1:
            raise ValueError("transmittance must lie in (0, 1]")
        return self


class ValidatedMetricV1(StrictContractModel):
    """A result metric the tabulated response was validated for, with its error bound: ``abs_error`` (in the metric's
    unit) and / or ``rel_error`` (a fraction), and how the bound was established (``basis``)."""

    metric: OpenSciId
    abs_error: OpticalDecimal | None = None
    rel_error: OpticalDecimal | None = None
    basis: AuthorText

    @model_validator(mode="after")
    def _consistent(self) -> ValidatedMetricV1:
        if self.abs_error is None and self.rel_error is None:
            raise ValueError(f"metric {self.metric}: state abs_error, rel_error or both")
        for name in ("abs_error", "rel_error"):
            value = getattr(self, name)
            if value is not None and _decimal(value, name) < 0:
                raise ValueError(f"metric {self.metric}: {name} cannot be negative")
        return self


class EvidenceToolV1(StrictContractModel):
    """A tool used to produce a tabulated response: its ``name`` and exact ``version``."""

    name: AuthorName
    version: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[ -~]*[!-~][ -~]*$")]


EvidenceDate = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$"),
    Field(description="Calendar date 'YYYY-MM-DD'."),
]


class ResponseEvidenceV1(StrictContractModel):
    """Where a tabulated response comes from: the ``method`` (measurement or simulation), the ``tools`` and their
    versions, the SHA-256 of the source data (``source_data_sha256``) and of the tool configuration
    (``configuration_sha256``, when there is one), how the table was fitted from the source data (``fit_model``), how it
    was validated (``validation_summary``) and when (``date``)."""

    method: AuthorText
    tools: tuple[EvidenceToolV1, ...] = Field(min_length=1, max_length=MAX_EVIDENCE_TOOLS)
    source_data_sha256: Sha256Digest
    configuration_sha256: Sha256Digest | None = None
    fit_model: AuthorText
    validation_summary: AuthorText
    date: EvidenceDate


ThinLensPolarization = Literal["LINEAR_X", "LINEAR_Y", "ANY"]

#: What a tabulated response rests on: ``EVIDENCED`` (it states its ``evidence``) or ``ASSUMED`` (it states none).
ResponseDataKind = Literal["EVIDENCED", "ASSUMED"]


def response_data_kind(evidence: ResponseEvidenceV1 | None) -> ResponseDataKind:
    """The data kind of a tabulated response that states its evidence (``EVIDENCED``) or does not (``ASSUMED``)."""

    return "ASSUMED" if evidence is None else "EVIDENCED"


class TabulatedThinLensModelV1(OmitAbsentFields, StrictContractModel):
    """A thin focusing element in the plane ``z = 0`` of the element frame (axis local +Z) whose response is a table:
    the component's (or a compound member's) pose places that plane, and ``plane_z_mm`` is ``0``.

    ``rows`` (strictly increasing ``wavelength_nm``) give the focal length and the delivered power fraction per
    wavelength. Between rows the focal length follows ``focal_interpolation`` (``LINEAR_IN_OPTICAL_POWER``: the optical
    power ``1 / f`` is interpolated linearly in wavelength) and the transmittance ``transmittance_interpolation``
    (``LINEAR``); with ``POINTS_ONLY`` (and for a one-row table) a wavelength within ``THIN_LENS_POINT_TOLERANCE`` (0.5 %)
    of a row takes that row and any other wavelength is outside the model. The model's valid band is the rows' span.
    The power that is not delivered into the focused beam (``1 - transmittance``) is the element's unmodelled response.

    Validity domain: light inside ``clear_radius_mm`` (between it and ``edge_radius_mm`` light is blocked, beyond the
    edge it passes by), a chief ray at most ``max_field_angle_deg`` from the element axis at the element, a Gaussian
    beam radius at the element of at most ``max_beam_radius_mm`` and an input polarization among
    ``validated_polarizations`` (``LINEAR_X`` / ``LINEAR_Y`` along local X / Y, ``ANY`` every polarization). Each of
    the three limits is optional: a run outside a stated limit is outside the model's scope; a limit that is not stated
    is not checked.

    ``supported_metrics`` (optional) are the result metrics the response was validated for, each with its error bound; a
    target whose light passes the element and asks for another metric is outside the model's scope. Without the list no
    metric is singled out. ``evidence`` (optional) records how the table was produced and validated; a table without it
    is an assumed response (``data_kind``: ``ASSUMED``, else ``EVIDENCED``).
    """

    # Serialized only when given: a table stated without them has one payload, with or without exclude_none.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("max_field_angle_deg", "supported_metrics", "evidence")

    model_kind: Literal["TABULATED_THIN_LENS"] = "TABULATED_THIN_LENS"
    model_version: Literal["1"] = "1"
    rows: tuple[ThinLensResponseRowV1, ...] = Field(min_length=1, max_length=MAX_THIN_LENS_ROWS)
    focal_interpolation: Literal["LINEAR_IN_OPTICAL_POWER", "POINTS_ONLY"]
    transmittance_interpolation: Literal["LINEAR", "POINTS_ONLY"]
    clear_radius_mm: OpticalDecimal
    edge_radius_mm: OpticalDecimal
    max_field_angle_deg: OpticalDecimal | None = None
    max_beam_radius_mm: OpticalDecimal | None = None
    validated_polarizations: tuple[ThinLensPolarization, ...] | None = Field(None, min_length=1, max_length=3)
    supported_metrics: tuple[ValidatedMetricV1, ...] | None = Field(None, min_length=1, max_length=MAX_VALIDATED_METRICS)
    evidence: ResponseEvidenceV1 | None = None
    plane_z_mm: OpticalDecimal = "0"
    status: FactStatus

    @model_validator(mode="after")
    def _consistent(self) -> TabulatedThinLensModelV1:
        if Decimal(self.plane_z_mm) != 0:
            raise ValueError("a tabulated thin lens lies in its frame's z = 0 plane: plane_z_mm is 0 (place the component at the plane)")
        wavelengths = [Decimal(row.wavelength_nm) for row in self.rows]
        if any(b <= a for a, b in zip(wavelengths, wavelengths[1:])):
            raise ValueError("rows must be ordered by strictly increasing wavelength_nm")
        if self.focal_interpolation == "LINEAR_IN_OPTICAL_POWER":
            signs = {Decimal(row.focal_length_mm) > 0 for row in self.rows}
            if len(signs) > 1:
                raise ValueError("LINEAR_IN_OPTICAL_POWER interpolates between rows of one sign of focal length (a converging or a "
                                 "diverging lens); use POINTS_ONLY for a table that changes sign")
        if not _positive(self.clear_radius_mm, "clear_radius_mm") <= _positive(self.edge_radius_mm, "edge_radius_mm"):
            raise ValueError("clear_radius_mm cannot exceed edge_radius_mm")
        if self.max_field_angle_deg is not None and not 0 < _decimal(self.max_field_angle_deg, "max_field_angle_deg") < 90:
            raise ValueError("max_field_angle_deg must lie strictly between 0 and 90")
        if self.max_beam_radius_mm is not None:
            _positive(self.max_beam_radius_mm, "max_beam_radius_mm")
        polarizations = self.validated_polarizations or ()
        if len(set(polarizations)) != len(polarizations):
            raise ValueError("duplicate validated polarization")
        if "ANY" in polarizations and len(polarizations) > 1:
            raise ValueError("ANY covers every polarization: list it alone")
        duplicates = duplicate_keys(self.supported_metrics or (), "metric")
        if duplicates:
            raise ValueError(f"duplicate supported metric: {duplicates}")
        return self

    @property
    def valid_band_nm(self) -> WavelengthBandV1:
        """The rows' span (``status`` the model's)."""

        return WavelengthBandV1(min_nm=self.rows[0].wavelength_nm, max_nm=self.rows[-1].wavelength_nm, status=self.status)

    @property
    def data_kind(self) -> ResponseDataKind:
        """``EVIDENCED`` when the table states its ``evidence``, ``ASSUMED`` when it does not."""

        return response_data_kind(self.evidence)

    def supported_metric(self, metric: str) -> ValidatedMetricV1 | None:
        return next((entry for entry in self.supported_metrics or () if entry.metric == metric), None)


class FaradayResponseRowV1(StrictContractModel):
    """One row of a Faraday rotator's response: at vacuum wavelength ``wavelength_nm``, the polarization turns by
    ``rotation_deg`` about the element's local +Z (right-handed: from local +X toward local +Y; negative: the other way)
    and ``transmittance`` of the power passes, in ``(0, 1]``."""

    wavelength_nm: OpticalDecimal
    rotation_deg: OpticalDecimal
    transmittance: OpticalDecimal

    @model_validator(mode="after")
    def _consistent(self) -> FaradayResponseRowV1:
        _positive(self.wavelength_nm, "wavelength_nm")
        if not 0 < _decimal(self.transmittance, "transmittance") <= 1:
            raise ValueError("transmittance must lie in (0, 1]")
        return self


class FaradayRotatorModelV1(OmitAbsentFields, StrictContractModel):
    """A thin nonreciprocal polarization rotator in the plane ``z = 0`` of the element frame: the component's (or a compound
    member's) pose places that plane and the field axis local +Z, and ``plane_z_mm`` is ``0``.

    Rotation sense: the field vector turns by ``theta(lambda)`` right-handed about local +Z whichever way the light travels.
    In a ray's right-handed transverse basis ``(e1, e2)`` with ``e1 x e2 = d`` the Jones matrix is
    ``sqrt(T) [[cos b, -sin b], [sin b, cos b]]`` with ``b = sign(d . z) theta``. Going through and coming back after a
    normal-incidence mirror therefore turns the polarization by ``2 theta``; a reciprocal element cancels instead.

    ``rows`` (strictly increasing ``wavelength_nm``) state ``theta`` and the transmittance per wavelength. With
    ``interpolation`` ``LINEAR`` both are linear in wavelength between rows and the domain is the rows' span; with
    ``POINTS_ONLY`` (and for a one-row table) the domain is the rows' wavelengths exactly. Light of a wavelength outside the
    domain stops at the element: its response there is not stated, and what lies behind has no value for that light.

    ``valid_aoi_deg`` (optional): the chief-ray angles of incidence the response is stated for. ``theta`` applies unchanged
    at every angle inside it, and a use outside it is out of the model's scope; without it ``theta`` applies at every angle
    and no angle is checked. Inside ``clear_radius_mm`` light passes; between it and ``edge_radius_mm`` it is blocked; beyond
    the edge it passes by.

    ``operating_conditions`` (optional): the conditions the table holds for (temperature, magnet or field, as stated by its
    source). ``evidence`` (optional): how the table was obtained; a table without it is an assumed response (``data_kind``:
    ``ASSUMED``, else ``EVIDENCED``). ``design_geometry`` of the model gives the body drawn and checked for clearance;
    its length and refractive index do not act on the light: phase, optical path and group delay through the element are
    not stated, and they have no value behind it.
    """

    # Serialized only when given: a table stated without them has one payload, with or without exclude_none.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("valid_aoi_deg", "operating_conditions", "evidence")

    model_kind: Literal["FARADAY_ROTATOR"] = "FARADAY_ROTATOR"
    model_version: Literal["1"] = "1"
    rows: tuple[FaradayResponseRowV1, ...] = Field(min_length=1, max_length=MAX_FARADAY_ROWS)
    interpolation: Literal["LINEAR", "POINTS_ONLY"]
    clear_radius_mm: OpticalDecimal
    edge_radius_mm: OpticalDecimal
    valid_aoi_deg: AngleRangeV1 | None = None
    operating_conditions: AuthorText | None = None
    evidence: ResponseEvidenceV1 | None = None
    plane_z_mm: OpticalDecimal = "0"
    status: FactStatus

    @model_validator(mode="after")
    def _consistent(self) -> FaradayRotatorModelV1:
        if Decimal(self.plane_z_mm) != 0:
            raise ValueError("a Faraday rotator lies in its frame's z = 0 plane: plane_z_mm is 0 (place the component at the plane)")
        wavelengths = [Decimal(row.wavelength_nm) for row in self.rows]
        if any(b <= a for a, b in zip(wavelengths, wavelengths[1:])):
            raise ValueError("rows must be ordered by strictly increasing wavelength_nm")
        if not _positive(self.clear_radius_mm, "clear_radius_mm") <= _positive(self.edge_radius_mm, "edge_radius_mm"):
            raise ValueError("clear_radius_mm cannot exceed edge_radius_mm")
        if self.valid_aoi_deg is not None and not Decimal(self.valid_aoi_deg.max_deg) < GRAZING_AOI_DEG:
            raise ValueError("valid_aoi_deg must stay below 90 deg")
        return self

    @property
    def valid_band_nm(self) -> WavelengthBandV1:
        """The rows' span (``status`` the model's)."""

        return WavelengthBandV1(min_nm=self.rows[0].wavelength_nm, max_nm=self.rows[-1].wavelength_nm, status=self.status)

    @property
    def data_kind(self) -> ResponseDataKind:
        """``EVIDENCED`` when the table states its ``evidence``, ``ASSUMED`` when it does not."""

        return response_data_kind(self.evidence)


class GratingResponseRowV1(StrictContractModel):
    """One row of a grating's order response: the efficiency of diffraction order ``order`` at vacuum wavelength
    ``wavelength_nm`` and signed angle of incidence ``aoi_deg``.

    ``aoi_deg`` is measured in the incident medium, in the plane perpendicular to the grooves: positive when the light
    travels toward the grating vector's direction, so ``sin(aoi_deg)`` is the incident direction's component along it.
    ``efficiency_te`` / ``efficiency_tm`` belong to a ``TE_TM`` table (TE: the electric field along the grooves; TM: across
    them), ``efficiency`` to an ``UNPOLARIZED`` one.
    """

    wavelength_nm: OpticalDecimal
    aoi_deg: OpticalDecimal
    order: int
    efficiency_te: OpticalDecimal | None = None
    efficiency_tm: OpticalDecimal | None = None
    efficiency: OpticalDecimal | None = None

    @model_validator(mode="after")
    def _consistent(self) -> GratingResponseRowV1:
        _positive(self.wavelength_nm, "wavelength_nm")
        if not abs(_decimal(self.aoi_deg, "aoi_deg")) < GRAZING_AOI_DEG:
            raise ValueError("aoi_deg must lie between -90 and 90 deg")
        for name in ("efficiency_te", "efficiency_tm", "efficiency"):
            value = getattr(self, name)
            if value is not None and _decimal(value, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        return self


class GratingAbsorptionRowV1(StrictContractModel):
    """The fraction of the incident power a grating absorbs at ``wavelength_nm`` and ``aoi_deg``: ``absorption_te`` /
    ``absorption_tm`` in a ``TE_TM`` table, ``absorption`` in an ``UNPOLARIZED`` one."""

    wavelength_nm: OpticalDecimal
    aoi_deg: OpticalDecimal
    absorption_te: OpticalDecimal | None = None
    absorption_tm: OpticalDecimal | None = None
    absorption: OpticalDecimal | None = None

    @model_validator(mode="after")
    def _consistent(self) -> GratingAbsorptionRowV1:
        _positive(self.wavelength_nm, "wavelength_nm")
        if not abs(_decimal(self.aoi_deg, "aoi_deg")) < GRAZING_AOI_DEG:
            raise ValueError("aoi_deg must lie between -90 and 90 deg")
        for name in ("absorption_te", "absorption_tm", "absorption"):
            value = getattr(self, name)
            if value is not None and _decimal(value, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        return self


class GratingOrderResponseV1(StrictContractModel):
    """A grating's order response with its provenance.

    ``rows``: every order of the table at every wavelength and angle of the table (a full grid, one row per combination).
    ``normalization``: ``ABSOLUTE``, a fraction of the power incident on the grating, or ``RELATIVE``, relative to the
    reference ``basis`` names (a relative table does not give absolute order powers). ``polarization_basis``: ``TE_TM``,
    the efficiency for each eigen-polarization of the grooves, or ``UNPOLARIZED``, the mean of the two. A beam the Engine
    traces is fully polarized, so an ``UNPOLARIZED`` table bounds an order's power (at most twice the mean, at most the
    incident power) without giving it; state the same value for TE and TM when the response is known not to depend on the
    polarization. ``interpolation``: ``LINEAR``, bilinear in wavelength and angle over the grid's span, or ``POINTS_ONLY``,
    the grid's points exactly (a wavelength as stated, an angle within 1e-6 deg). Outside that domain the response is not
    available: nothing is extrapolated. ``max_conical_angle_deg``: how far from the plane perpendicular to the grooves
    the chief ray may arrive for the table to hold.

    In an ``ABSOLUTE`` table the orders and ``absorption`` of one wavelength, angle and polarization sum to at most 1; what
    they leave is light of orders the table does not represent, whose fate is not stated. ``data_kind``: ``MEASURED``,
    ``INDEPENDENTLY_COMPUTED`` (both with ``evidence``) or ``ASSUMED`` (a value chosen for the design, described in
    ``basis``). A table holds for light arriving from the side the model's ``mode`` names.
    """

    normalization: Literal["ABSOLUTE", "RELATIVE"]
    polarization_basis: Literal["TE_TM", "UNPOLARIZED"]
    interpolation: Literal["LINEAR", "POINTS_ONLY"]
    rows: tuple[GratingResponseRowV1, ...] = Field(min_length=1, max_length=MAX_GRATING_RESPONSE_ROWS)
    absorption: tuple[GratingAbsorptionRowV1, ...] = Field((), max_length=MAX_GRATING_RESPONSE_ROWS)
    max_conical_angle_deg: OpticalDecimal = "0"
    data_kind: Literal["MEASURED", "INDEPENDENTLY_COMPUTED", "ASSUMED"]
    basis: AuthorText
    evidence: ResponseEvidenceV1 | None = None

    @model_validator(mode="after")
    def _consistent(self) -> GratingOrderResponseV1:
        if (self.evidence is None) != (self.data_kind == "ASSUMED"):
            raise ValueError("measured or independently computed data states its evidence; assumed data states none and describes itself in basis")
        if not 0 <= _decimal(self.max_conical_angle_deg, "max_conical_angle_deg") < GRAZING_AOI_DEG:
            raise ValueError("max_conical_angle_deg must lie in [0, 90) deg")
        resolved = self.polarization_basis == "TE_TM"
        for row in self.rows:
            if resolved != (row.efficiency_te is not None and row.efficiency_tm is not None) or resolved == (row.efficiency is not None):
                raise ValueError("a TE_TM table states efficiency_te and efficiency_tm per row, an UNPOLARIZED table efficiency")
        for row in self.absorption:
            if resolved != (row.absorption_te is not None and row.absorption_tm is not None) or resolved == (row.absorption is not None):
                raise ValueError("a TE_TM table states absorption_te and absorption_tm per row, an UNPOLARIZED table absorption")
        if self.normalization == "RELATIVE" and self.absorption:
            raise ValueError("a RELATIVE table states no absorption: its rows are not fractions of the incident power")
        key = lambda row: (Decimal(row.wavelength_nm), Decimal(row.aoi_deg))  # noqa: E731
        grid = {key(row) for row in self.rows}
        orders = {row.order for row in self.rows}
        if len({(*key(row), row.order) for row in self.rows}) != len(self.rows):
            raise ValueError("response rows repeat an order at one wavelength and angle")
        wavelengths, angles = {w for w, _ in grid}, {a for _, a in grid}
        if len(self.rows) != len(wavelengths) * len(angles) * len(orders):
            raise ValueError("response rows must form a full grid: every order at every wavelength and every angle")
        losses = {key(row): row for row in self.absorption}
        if len(losses) != len(self.absorption):
            raise ValueError("absorption rows repeat a wavelength and angle")
        if self.absorption and set(losses) != grid:
            raise ValueError("absorption rows must cover the response rows' wavelengths and angles")
        if self.normalization == "ABSOLUTE":
            for name, loss_name in (("efficiency_te", "absorption_te"), ("efficiency_tm", "absorption_tm"), ("efficiency", "absorption")):
                totals: dict[tuple[Decimal, Decimal], Decimal] = {}
                with localcontext(DECIMAL_CONTEXT):
                    for row in self.rows:
                        value = getattr(row, name)
                        if value is not None:
                            totals[key(row)] = totals.get(key(row), Decimal(0)) + Decimal(value)
                    for point, total in totals.items():
                        loss = getattr(losses[point], loss_name) if point in losses else None
                        if total + Decimal(loss or 0) > 1:
                            raise ValueError(f"at {point[0]} nm and {point[1]} deg the {name} of the orders and the absorption sum to more than 1")
        return self


class DiffractionGratingModelV1(StrictContractModel):
    """A thin periodic plane in ``z = 0`` of the element frame; the component's (or a compound member's) pose places it, and
    ``plane_z_mm`` is ``0``. ``REFLECTION``: local +Z is the outward normal of the ruled face, light arrives from the +Z
    side and light from the -Z side is blocked. ``TRANSMISSION``: local +Z is the axis; light may cross either way and a
    response table holds for light travelling along +Z.

    Exactly one of ``period_nm`` and ``line_density_per_mm`` states the period ``L``. The grating vector ``K`` lies in the
    plane at ``grating_vector_angle_deg`` from local +X toward local +Y; the grooves run along ``local +Z x K`` (local +Y
    at the default angle).

    Sign convention (the general vector grating equation): an order ``m`` leaves with the tangential part
    ``n_exit t_exit = n_in t_in + m (lambda / L) K``, ``t`` being a direction's component in the plane and ``lambda`` the
    vacuum wavelength. Conical mounts follow from the same vector equation. Where ``|n_exit t_exit|`` is at least
    ``n_exit`` the order is nonpropagating: it is reported and not traced. At normal incidence an order ``+1`` turns toward
    ``+K``; with an incidence angle measured positive toward ``+K``, ``n_exit sin(exit) = n_in sin(incidence) + m lambda / L``.

    Exactly one of ``orders`` (the orders followed, strictly increasing) and ``all_propagating_orders`` (``[first, last]``:
    every propagating order of that range) names the orders followed; other propagating orders are reported and not
    followed. ``medium_positive_z`` / ``medium_negative_z``: the media on the two sides (``AIR`` or a declared material;
    a reflection grating has its medium on the +Z side). The clear aperture is one of ``clear_radius_mm`` and
    ``clear_half_widths_mm`` (a rectangle along local X / Y), inside the outline ``edge_radius_mm`` / ``edge_half_widths_mm``:
    between them light is blocked, beyond the edge it passes by.

    ``response``: the order powers and their provenance (``GratingOrderResponseV1``). Without one, or where it does not hold,
    each order's direction is known and its power is not: the incident power is booked once as the element's unmodelled
    response, and every order is followed with its power null and an upper bound. An order's polarization is known only
    when a ``TE_TM`` table holds and the light is purely TE or TM. Phase and optical path through the element are not
    stated and have no value behind it; resolving power, diffraction by the finite aperture and pulse dispersion are not
    modelled.
    """

    model_kind: Literal["DIFFRACTION_GRATING"] = "DIFFRACTION_GRATING"
    model_version: Literal["1"] = "1"
    mode: Literal["REFLECTION", "TRANSMISSION"]
    period_nm: OpticalDecimal | None = None
    line_density_per_mm: OpticalDecimal | None = None
    grating_vector_angle_deg: OpticalDecimal = "0"
    orders: tuple[int, ...] | None = Field(None, min_length=1, max_length=MAX_GRATING_ORDERS)
    all_propagating_orders: tuple[int, int] | None = None
    medium_positive_z: MaterialId = AIR
    medium_negative_z: MaterialId = AIR
    clear_radius_mm: OpticalDecimal | None = None
    clear_half_widths_mm: tuple[OpticalDecimal, OpticalDecimal] | None = None
    edge_radius_mm: OpticalDecimal | None = None
    edge_half_widths_mm: tuple[OpticalDecimal, OpticalDecimal] | None = None
    response: GratingOrderResponseV1 | None = None
    plane_z_mm: OpticalDecimal = "0"
    status: FactStatus

    @model_validator(mode="after")
    def _consistent(self) -> DiffractionGratingModelV1:
        if Decimal(self.plane_z_mm) != 0:
            raise ValueError("a diffraction grating lies in its frame's z = 0 plane: plane_z_mm is 0 (place the component at the plane)")
        if (self.period_nm is None) == (self.line_density_per_mm is None):
            raise ValueError("a diffraction grating has exactly one of period_nm and line_density_per_mm")
        _positive(self.period_nm if self.period_nm is not None else self.line_density_per_mm, "the period")  # type: ignore[arg-type]
        if (self.orders is None) == (self.all_propagating_orders is None):
            raise ValueError("a diffraction grating has exactly one of orders and all_propagating_orders")
        if self.orders is not None and any(b <= a for a, b in zip(self.orders, self.orders[1:])):
            raise ValueError("orders must be strictly increasing")
        if self.all_propagating_orders is not None:
            first, last = self.all_propagating_orders
            if not first <= last or last - first >= MAX_GRATING_ORDERS:
                raise ValueError(f"all_propagating_orders is [first, last] with first <= last, at most {MAX_GRATING_ORDERS} orders")
        if self.mode == "REFLECTION" and self.medium_negative_z != AIR:
            raise ValueError("a reflection grating has its medium on the +Z side: medium_negative_z stays AIR")
        if (self.clear_radius_mm is None) == (self.clear_half_widths_mm is None):
            raise ValueError("a diffraction grating has exactly one of clear_radius_mm and clear_half_widths_mm")
        if (self.edge_radius_mm is None) == (self.edge_half_widths_mm is None):
            raise ValueError("a diffraction grating has exactly one of edge_radius_mm and edge_half_widths_mm")
        if self.clear_radius_mm is not None:
            cx = cy = _positive(self.clear_radius_mm, "clear_radius_mm")
            reach_sq = cx * cx
        else:
            cx, cy = (_positive(v, "clear_half_widths_mm") for v in self.clear_half_widths_mm)  # type: ignore[union-attr]
            reach_sq = cx * cx + cy * cy
        if self.edge_radius_mm is not None:
            if reach_sq > _positive(self.edge_radius_mm, "edge_radius_mm") ** 2:
                raise ValueError("the clear aperture must lie inside the edge radius")
        else:
            ex, ey = (_positive(v, "edge_half_widths_mm") for v in self.edge_half_widths_mm)  # type: ignore[union-attr]
            if cx > ex or cy > ey:
                raise ValueError("the clear aperture must lie inside the edge half widths")
        return self

    @property
    def period(self) -> Decimal:
        """The period ``L`` (nm)."""

        with localcontext(DECIMAL_CONTEXT):
            return Decimal(self.period_nm) if self.period_nm is not None else Decimal(1000000) / Decimal(self.line_density_per_mm)  # type: ignore[arg-type]

    def material_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(m for m in (self.medium_positive_z, self.medium_negative_z) if m not in RESERVED_MEDIA))


#: The most facet groups of one FACET_SOLID.
MAX_FACET_GROUPS = 64

FacetIndex = Annotated[int, Field(strict=True, ge=0)]


class FacetGroupV1(StrictContractModel):
    """Facets of a ``FACET_SOLID`` with their own surface: ``triangles`` (indices into the model's ``triangles``) are
    ``TRANSMITTED`` (a refracting face, with ``coating_id`` its coating), ``REFLECTED`` (a mirrored face: light inside the
    solid reflects there, with ``coating_id`` its reflectance; no coating: an ideal reflector) or ``ABSORBED`` (a blackened
    face: light meeting it ends there)."""

    triangles: tuple[FacetIndex, ...] = Field(min_length=1, max_length=MAX_MESH_TRIANGLES)
    traced: Literal["TRANSMITTED", "REFLECTED", "ABSORBED"]
    coating_id: CoatingId | None = None

    @model_validator(mode="after")
    def _consistent(self) -> FacetGroupV1:
        if self.traced == "ABSORBED" and self.coating_id is not None:
            raise ValueError("an absorbing facet group states no coating")
        return self


class FacetSolidModelV1(StrictContractModel):
    """A solid of the material ``medium`` bounded by flat triangular facets (a prism, a corner cube, a polyhedral window,
    or a CAD part exported as facets), in the element frame: ``vertices_mm`` and ``triangles`` (vertex index triples,
    counter-clockwise seen from outside). The facets close around the solid: every edge belongs to exactly two facets that
    run it in opposite directions, no facet is degenerate and the enclosed volume is positive. Outside the solid is air.

    ``approximation``: ``FACETED_EXACT`` (the part is these facets) or ``TESSELLATED_FROM_CAD`` (the facets approximate
    curved surfaces within ``max_chord_deviation_mm``); ``source`` says where they come from. Every facet refracts with
    bare Fresnel power (or ``coating_id``'s), except the facets of a ``facet_groups`` entry, which take that group's surface.

    The element frame is the frame of the corners (the component's pose places it); a stated ``frame`` only names the
    element's ports (``reference_axis`` ``PORTS``: where the beam enters and leaves a part that folds it).

    The run traces the facets as they are: the edge-connected facets of one group that lie in one plane form one planar
    face. A tessellated surface's flat facets carry none of its curvature, so the Gaussian beam metrics behind them are out
    of model scope, and the part of a beam beyond a face's edge is not followed by that beam (its fate is stated as not
    known).
    """

    model_kind: Literal["FACET_SOLID"] = "FACET_SOLID"
    model_version: Literal["1"] = "1"
    vertices_mm: tuple[tuple[OpticalDecimal, OpticalDecimal, OpticalDecimal], ...] = Field(min_length=4, max_length=MAX_MESH_VERTICES)
    triangles: tuple[tuple[FacetIndex, FacetIndex, FacetIndex], ...] = Field(min_length=4, max_length=MAX_MESH_TRIANGLES)
    approximation: Literal["FACETED_EXACT", "TESSELLATED_FROM_CAD"]
    max_chord_deviation_mm: OpticalDecimal | None = None
    source: AuthorText | None = None
    medium: MaterialId
    coating_id: CoatingId | None = None
    facet_groups: tuple[FacetGroupV1, ...] | None = Field(None, min_length=1, max_length=MAX_FACET_GROUPS)
    valid_band_nm: WavelengthBandV1
    status: FactStatus

    @model_validator(mode="after")
    def _consistent(self) -> FacetSolidModelV1:
        if self.medium in RESERVED_MEDIA:
            raise ValueError(f"a facet solid is a solid of a declared material, not {self.medium}")
        if (self.approximation == "TESSELLATED_FROM_CAD") != (self.max_chord_deviation_mm is not None):
            raise ValueError("max_chord_deviation_mm is stated for TESSELLATED_FROM_CAD facets and only for them")
        if self.max_chord_deviation_mm is not None and _decimal(self.max_chord_deviation_mm, "max_chord_deviation_mm") < 0:
            raise ValueError("max_chord_deviation_mm cannot be negative")
        problem = mesh_problem(self.vertex_floats(), self.triangles, True)
        if problem is not None:
            raise ValueError(f"facets: {problem}")
        seen: dict[int, int] = {}
        for g, group in enumerate(self.facet_groups or ()):
            for k in group.triangles:
                if k >= len(self.triangles):
                    raise ValueError(f"facet group {g} names triangle {k}, outside 0 .. {len(self.triangles) - 1}")
                if k in seen:
                    raise ValueError(f"triangle {k} is in facet groups {seen[k]} and {g}: a facet has one surface")
                seen[k] = g
        return self

    def vertex_floats(self) -> tuple[tuple[float, float, float], ...]:
        """The corners as the nearest binary64 numbers (mm)."""

        return tuple((float(Decimal(x)), float(Decimal(y)), float(Decimal(z))) for x, y, z in self.vertices_mm)

    def material_ids(self) -> tuple[str, ...]:
        return (self.medium,)

    def coating_ids(self) -> tuple[str, ...]:
        ids = {self.coating_id} | {group.coating_id for group in self.facet_groups or ()}
        return tuple(sorted(i for i in ids if i is not None))


#: The most rows of a nonlinear crystal's thermo-optic table.
MAX_THERMO_OPTIC_ROWS = 64
#: The highest quasi-phase-matching order (odd orders only).
MAX_QPM_ORDER = 15
#: Engine surface ids of a nonlinear crystal's entrance (``z = 0``) and exit (``z = length``) faces.
NONLINEAR_FACE_IDS = ("S1", "S2")


class ThermoOpticRowV1(StrictContractModel):
    """The temperature coefficients of a crystal's ordinary and extraordinary indices at one vacuum wavelength (per kelvin)."""

    wavelength_nm: OpticalDecimal
    dno_dT_per_K: OpticalDecimal
    dne_dT_per_K: OpticalDecimal

    @model_validator(mode="after")
    def _consistent(self) -> ThermoOpticRowV1:
        _positive(self.wavelength_nm, "wavelength_nm")
        _decimal(self.dno_dT_per_K, "dno_dT_per_K")
        _decimal(self.dne_dT_per_K, "dne_dT_per_K")
        return self


class CrystalThermoOpticV1(StrictContractModel):
    """How a crystal's indices change with temperature: the material's index laws hold at ``reference_temperature_c``; at
    ``T`` each index is ``n(lambda, T_ref) + dn/dT(lambda) (T - T_ref)``, ``dn/dT`` linear in wavelength between the ``rows``
    (strictly increasing wavelengths; one row: the same at every wavelength). With two rows or more the rows span the model's
    ``valid_band_nm``, where the fundamental and its harmonic both lie. ``thermal_expansion_per_K``:
    the linear expansion along the crystal's length, which stretches a quasi-phase-matching period (absent: 0)."""

    reference_temperature_c: OpticalDecimal
    rows: tuple[ThermoOpticRowV1, ...] = Field(min_length=1, max_length=MAX_THERMO_OPTIC_ROWS)
    thermal_expansion_per_K: OpticalDecimal | None = None

    @model_validator(mode="after")
    def _consistent(self) -> CrystalThermoOpticV1:
        _decimal(self.reference_temperature_c, "reference_temperature_c")
        waves = [Decimal(row.wavelength_nm) for row in self.rows]
        if any(b <= a for a, b in zip(waves, waves[1:])):
            raise ValueError("thermo-optic rows must be ordered by strictly increasing wavelength_nm")
        if self.thermal_expansion_per_K is not None:
            _decimal(self.thermal_expansion_per_K, "thermal_expansion_per_K")
        return self


DTensorRow = tuple[OpticalDecimal, OpticalDecimal, OpticalDecimal, OpticalDecimal, OpticalDecimal, OpticalDecimal]


class NonlinearCrystalModelV1(StrictContractModel):
    """A rectangular slab of the uniaxial crystal ``medium`` (a material with ``birefringence``; its optic axis is stated in
    ``ProjectOpticalModelV1.optic_axes``) that generates the second harmonic of light entering it. Element frame: the
    entrance face is the plane ``z = 0``, the exit face ``z = length_mm``, both ``width_mm`` (local X) by ``height_mm``
    (local Y) centred on the Z axis, bare (Fresnel) or with ``coating_id``. The run traces the slab as a crystal (refraction,
    ordinary and extraordinary beams, walk-off).

    Nonlinearity: ``d_eff_pm_per_V`` states the effective coefficient of the interaction, or ``d_tensor_pm_per_V`` the
    contracted 3 x 6 tensor ``d_il`` (columns xx, yy, zz, yz, xz, xy) in the crystal frame whose Z is the optic axis and X
    ``crystal_x_axis`` (element frame, perpendicular to the optic axis): the run contracts it with the electric fields of the
    traced waves. ``nonlinear_basis`` says where the coefficients come from.

    Phase matching: ``fundamental_waves`` are the crystal waves of the two fundamental photons (``OO`` or ``EE``: type I, or
    with a quasi-phase-matching grating type 0; ``OE``: type II) and ``harmonic_wave`` the wave of the harmonic.
    ``qpm_period_um`` (order ``qpm_order``, odd, 50 % duty cycle): a periodically poled crystal whose grating vector lies along
    local Z; the stated coefficient is then the unpoled crystal's (``d_eff_pm_per_V``: for example ``d33`` of LiNbO3 for EE ->
    E, not the reduced ``2 d33 / pi``; the tensor as it is), and the run multiplies it by ``2 / (m pi)`` itself. The run computes
    the phase mismatch of the traced beam's waves (their directions to the optic axis, the indices at the temperature) itself.

    Temperature: ``temperature_c`` with ``thermo_optic`` (and only with it): the crystal's indices at that temperature;
    without them the material's index laws apply as stated and no temperature acceptance is computed.

    ``focusing``: ``NEAR_FIELD`` takes the beam radius as constant along the crystal (a thin crystal; a run checks it against
    the focused result), ``BOYD_KLEINMAN`` follows the Gaussian beam's focusing through it (both fundamental photons in one
    wave). The conversion is the undepleted-pump conversion of a TEM00 Gaussian beam of one continuous-wave frequency; where its
    conditions fail a run states the harmonic out of model scope. The crystal is uniaxial (biaxial crystals such as KTP or LBO
    and cubic crystals are not modelled); no crystal data are built in: the coefficients, the index laws and the thermo-optic
    data are stated with the Project."""

    model_kind: Literal["NONLINEAR_CRYSTAL"] = "NONLINEAR_CRYSTAL"
    model_version: Literal["1"] = "1"
    medium: MaterialId
    length_mm: OpticalDecimal
    width_mm: OpticalDecimal
    height_mm: OpticalDecimal
    coating_id: CoatingId | None = None
    fundamental_waves: Literal["OO", "EE", "OE"]
    harmonic_wave: Literal["O", "E"]
    d_eff_pm_per_V: OpticalDecimal | None = None
    d_tensor_pm_per_V: tuple[DTensorRow, DTensorRow, DTensorRow] | None = None
    crystal_x_axis: tuple[OpticalDecimal, OpticalDecimal, OpticalDecimal] | None = None
    qpm_period_um: OpticalDecimal | None = None
    qpm_order: int = Field(1, strict=True, ge=1, le=MAX_QPM_ORDER)
    temperature_c: OpticalDecimal | None = None
    thermo_optic: CrystalThermoOpticV1 | None = None
    focusing: Literal["NEAR_FIELD", "BOYD_KLEINMAN"] = "NEAR_FIELD"
    nonlinear_basis: AuthorText
    valid_band_nm: WavelengthBandV1
    status: FactStatus

    @model_validator(mode="after")
    def _consistent(self) -> NonlinearCrystalModelV1:
        if self.medium in RESERVED_MEDIA:
            raise ValueError(f"a nonlinear crystal is a slab of a declared crystal material, not {self.medium}")
        for label in ("length_mm", "width_mm", "height_mm"):
            _positive(getattr(self, label), label)
        if (self.d_eff_pm_per_V is None) == (self.d_tensor_pm_per_V is None):
            raise ValueError("state exactly one of d_eff_pm_per_V and d_tensor_pm_per_V")
        if self.d_eff_pm_per_V is not None:
            _decimal(self.d_eff_pm_per_V, "d_eff_pm_per_V")
        for row in self.d_tensor_pm_per_V or ():
            for value in row:
                _decimal(value, "d_tensor_pm_per_V")
        if (self.d_tensor_pm_per_V is None) != (self.crystal_x_axis is None):
            raise ValueError("crystal_x_axis is stated with d_tensor_pm_per_V and only with it")
        if self.crystal_x_axis is not None:
            with localcontext(DECIMAL_CONTEXT):
                if sum(_decimal(c, "crystal_x_axis") ** 2 for c in self.crystal_x_axis) == 0:
                    raise ValueError("crystal_x_axis must not be zero")
        if self.qpm_period_um is not None:
            _positive(self.qpm_period_um, "qpm_period_um")
        if self.qpm_order % 2 == 0:
            raise ValueError("qpm_order must be odd (a 50 % duty-cycle grating has no even orders)")
        if (self.temperature_c is None) != (self.thermo_optic is None):
            raise ValueError("temperature_c is stated with thermo_optic and only with it")
        if self.temperature_c is not None:
            _decimal(self.temperature_c, "temperature_c")
        if self.focusing == "BOYD_KLEINMAN" and self.fundamental_waves == "OE":
            raise ValueError("BOYD_KLEINMAN focusing takes both fundamental photons in one wave (OO or EE)")
        if self.thermo_optic is not None and len(self.thermo_optic.rows) > 1:
            low, high = Decimal(self.thermo_optic.rows[0].wavelength_nm), Decimal(self.thermo_optic.rows[-1].wavelength_nm)
            if not (low <= Decimal(self.valid_band_nm.min_nm) and Decimal(self.valid_band_nm.max_nm) <= high):
                raise ValueError("the thermo-optic rows must span valid_band_nm (the fundamental and harmonic wavelengths lie in it)")
        return self

    def as_surface_sequence(self) -> SurfaceSequenceModelV1:
        """The slab as the crystal surface sequence the run traces: ``S1`` at ``z = 0`` and ``S2`` at ``z = length_mm``,
        rectangular, the crystal between them."""

        with localcontext(DECIMAL_CONTEXT):
            half = (str(Decimal(self.width_mm) / 2), str(Decimal(self.height_mm) / 2))
        faces = tuple(PrescriptionSurfaceV1(surface_id=sid, vertex_z_mm=z, clear_half_widths_mm=half, coating_id=self.coating_id,
                                            traced=TracedBranches.TRANSMITTED, status=self.status)
                      for sid, z in zip(NONLINEAR_FACE_IDS, ("0", self.length_mm)))
        return SurfaceSequenceModelV1(fidelity=ModelFidelity.SURFACE_PRESCRIPTION, surfaces=faces, media=(self.medium,),
                                      edge_half_widths_mm=half, valid_band_nm=self.valid_band_nm,
                                      approximations=(SimulationApproximation.NOMINAL_PRESCRIPTION,
                                                      SimulationApproximation.IDEAL_SURFACE_FIGURE),
                                      status=self.status)

    def material_ids(self) -> tuple[str, ...]:
        return (self.medium,)

    def coating_ids(self) -> tuple[str, ...]:
        return () if self.coating_id is None else (self.coating_id,)


class MemberArrayV1(StrictContractModel):
    """Copies of a compound member on a lattice in the unit frame (a lens array, a pinhole array): ``counts`` = (columns
    along unit X, rows along unit Y), ``pitch_mm`` = (centre spacing along X within a row, row spacing along Y).

    ``RECTANGULAR``: cell ``(column i, row j)`` is moved by ``((i - (nx - 1) / 2) px, (j - (ny - 1) / 2) py)`` from the
    member's pose (the lattice is centred on it). ``HEXAGONAL``: the same, with neighbouring rows shifted against each
    other by half a pitch (even rows by ``-px / 4``, odd rows by ``+px / 4``; a single row is not shifted); a hexagonal
    close packing of round lenslets of diameter ``D`` has ``px = D`` and ``py = D sqrt(3) / 2``. Every cell keeps the
    member's orientation. Cell ``(i, j)`` is the member ``<member_id>_<j>_<i>``, its surfaces
    ``<member_id>_<j>_<i>-<surface_id>``."""

    layout: Literal["RECTANGULAR", "HEXAGONAL"]
    pitch_mm: tuple[OpticalDecimal, OpticalDecimal]
    counts: tuple[int, int]

    @model_validator(mode="after")
    def _consistent(self) -> MemberArrayV1:
        for value in self.pitch_mm:
            if not Decimal(value) > 0:
                raise ValueError(f"an array pitch is positive, got {value}")
        nx, ny = self.counts
        if nx < 1 or ny < 1:
            raise ValueError("an array has at least one column and one row")
        if nx * ny < 2:
            raise ValueError("an array has at least two cells (a single copy is the member itself)")
        if nx * ny > MAX_ARRAY_CELLS:
            raise ValueError(f"an array has at most {MAX_ARRAY_CELLS} cells, got {nx * ny}")
        return self

    def cells(self) -> tuple[tuple[int, int, Decimal, Decimal], ...]:
        """``(column, row, x offset, y offset)`` of every cell, row by row, in the unit frame (mm)."""

        with localcontext(DECIMAL_CONTEXT):
            px, py = Decimal(self.pitch_mm[0]), Decimal(self.pitch_mm[1])
            nx, ny = self.counts
            out = []
            for j in range(ny):
                shift = Decimal(0) if self.layout == "RECTANGULAR" or ny == 1 else (px / 4 if j % 2 else -px / 4)
                for i in range(nx):
                    out.append((i, j, (Decimal(i) - Decimal(nx - 1) / 2) * px + shift, (Decimal(j) - Decimal(ny - 1) / 2) * py))
            return tuple(out)


#: The most cells one member array holds.
MAX_ARRAY_CELLS = 256
#: A member id that is arrayed has at most this many characters (its cells' ids add ``_<row>_<column>``).
MAX_ARRAYED_MEMBER_ID = 8


class CompoundMemberV1(StrictContractModel):
    """One member of a COMPOUND unit: the model ``model_id`` placed at ``pose`` in the unit's frame; with ``array``, a
    lattice of copies of it (``MemberArrayV1``)."""

    member_id: MemberId
    model_id: ModelId
    pose: PoseV1 = Field(description="Pose of the member model's element frame in the unit frame.")
    array: MemberArrayV1 | None = None

    @model_validator(mode="after")
    def _unit_pose(self) -> CompoundMemberV1:
        require_unit_quaternion(self.pose, f"compound member {self.member_id!r} pose")
        if self.array is not None and len(self.member_id) > MAX_ARRAYED_MEMBER_ID:
            raise ValueError(f"an arrayed member's id has at most {MAX_ARRAYED_MEMBER_ID} characters (its cells add _<row>_<column>)")
        return self

    def cell_ids(self) -> tuple[str, ...]:
        """The member ids it stands for: itself, or one per array cell."""

        if self.array is None:
            return (self.member_id,)
        return tuple(f"{self.member_id}_{j}_{i}" for i, j, _x, _y in self.array.cells())


#: The most joints one COMPOUND unit states, and the widest cement layer of a joint (mm).
MAX_COMPOUND_JOINTS = 16
MAX_CEMENT_THICKNESS_MM = Decimal(1)


class MemberJointV1(StrictContractModel):
    """Two members of a COMPOUND unit joined face to face (a cemented doublet prism, a Wollaston or Rochon prism, a
    Glan-Thompson polarizer): surface ``surface_a`` of member ``member_a`` and surface ``surface_b`` of member ``member_b``.

    Both members are ``SURFACE_SET`` models with a body, both surfaces are uncoated plane faces of their bodies traced
    ``TRANSMITTED`` (air in front, the body's medium behind), and in the unit frame they face each other: their front normals
    are opposite. Without ``cement_material_id`` the two faces coincide (an optical contact): the unit has one interface
    there, member A's medium on one side and member B's on the other, two crystals cut differently included. With
    ``cement_material_id`` the faces are parallel and apart by the cement layer's thickness (more than 0, at most
    ``MAX_CEMENT_THICKNESS_MM``) and the layer between them is that declared isotropic, homogeneous material (no crystal, no
    gradient index): light crosses (or is totally reflected at) each face of the layer. The joint's interface is member A's
    surface (its clear aperture); the layer's thin-film interference is not modelled.
    """

    member_a: MemberId
    surface_a: SurfaceId
    member_b: MemberId
    surface_b: SurfaceId
    cement_material_id: MaterialId | None = None

    @model_validator(mode="after")
    def _distinct(self) -> MemberJointV1:
        if self.member_a == self.member_b:
            raise ValueError(f"a joint joins two different members, got {self.member_a} twice")
        return self


class CompoundModelV1(OmitAbsentFields, StrictContractModel):
    """One optical unit made of member models at fixed poses (an objective, a relay, a telescope).

    The unit is placed in the design as a whole (its component's ``nominal_pose`` places the unit frame); every surface of
    every member is simulated. ``envelope`` is the unit's outer boundary in its own frame (every member body lies
    inside it); ``ports`` optionally name the unit's external optical entry and exits. Members are not themselves
    COMPOUND models. ``joints`` join members face to face (``MemberJointV1``: an optical contact or a cement layer); a
    member with an ``array`` is not joined.
    """

    # Serialized only when given: a document without it keeps its earlier payload and hash.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("joints",)

    model_kind: Literal["COMPOUND"] = "COMPOUND"
    model_version: Literal["1"] = "1"
    members: tuple[CompoundMemberV1, ...] = Field(min_length=1, max_length=MAX_COMPOUND_MEMBERS)
    envelope: DesignGeometryV1
    ports: tuple[FramePortV1, ...] | None = Field(None, min_length=1)
    joints: tuple[MemberJointV1, ...] | None = Field(None, min_length=1, max_length=MAX_COMPOUND_JOINTS)

    @model_validator(mode="after")
    def _consistent(self) -> CompoundModelV1:
        duplicates = duplicate_keys(self.members, "member_id")
        if duplicates:
            raise ValueError(f"duplicate member_id in compound unit: {duplicates}")
        ids = [cell for member in self.members for cell in member.cell_ids()]
        clashes = sorted({cell for cell in ids if ids.count(cell) > 1})
        if clashes:
            raise ValueError(f"a member array's cell ids clash with other members: {clashes}")
        if len(ids) > MAX_COMPOUND_CELLS:
            raise ValueError(f"a compound unit holds at most {MAX_COMPOUND_CELLS} members and array cells, got {len(ids)}")
        port_duplicates = duplicate_keys(self.ports or (), "port_id")
        if port_duplicates:
            raise ValueError(f"duplicate port_id in compound unit: {port_duplicates}")
        by_member = {member.member_id: member for member in self.members}
        faces: list[tuple[str, str]] = []
        for joint in self.joints or ():
            for member_id, surface_id in ((joint.member_a, joint.surface_a), (joint.member_b, joint.surface_b)):
                member = by_member.get(member_id)
                if member is None:
                    raise ValueError(f"a joint names {member_id}, which is not a member of the unit")
                if member.array is not None:
                    raise ValueError(f"a joint names {member_id}, an arrayed member: array cells are not joined")
                faces.append((member_id, surface_id))
        repeated = sorted({f"{m}.{s}" for m, s in faces if faces.count((m, s)) > 1})
        if repeated:
            raise ValueError(f"a member surface is in more than one joint: {repeated}")
        return self

    def material_ids(self) -> tuple[str, ...]:
        """The cement materials of its joints (the members' own media belong to their models)."""

        return tuple(sorted({j.cement_material_id for j in self.joints or () if j.cement_material_id is not None}))


ProjectSimulationModelV1 = Annotated[
    SurfaceSequenceModelV1
    | SurfaceSetModelV1
    | PolarizationElementModelV1
    | NeutralDensityModelV1
    | ApertureStopModelV1
    | IdealThinLensModelV1
    | TabulatedThinLensModelV1
    | FaradayRotatorModelV1
    | DiffractionGratingModelV1
    | CompoundModelV1
    | FacetSolidModelV1
    | NonlinearCrystalModelV1,
    Field(discriminator="model_kind"),
]

#: Model kinds that carry their own fixed element frame (``ProjectOpticalModelV1.frame`` stays unset).
FRAME_FREE_KINDS = frozenset({"APERTURE_STOP", "IDEAL_THIN_LENS", "TABULATED_THIN_LENS", "FARADAY_ROTATOR", "COMPOUND"})


#: The most crystal materials one model orients.
MAX_OPTIC_AXES = 8
#: How far from 0 the cosine between a nonlinear crystal's X axis and its optic axis may be (two stated perpendicular axes).
CRYSTAL_AXES_PERPENDICULAR_TOL = Decimal("1e-9")
#: Model kinds whose media may be uniaxial crystals (their crystal faces are planes the light refracts at).
CRYSTAL_MODEL_KINDS = frozenset({"SURFACE_SEQUENCE", "SURFACE_SET", "NONLINEAR_CRYSTAL"})


class CrystalOpticAxisV1(StrictContractModel):
    """The optic axis of a uniaxial crystal in one element: ``material_id`` names a crystal material the model's media use
    (``ProjectModelMaterialV1.birefringence``) and ``direction`` the axis in the element frame (the model's own frame; a
    COMPOUND member's frame turns with the member's pose and the unit with the component's). Only the line matters: the
    direction is normalized and its sign does not change the physics. Every region of that material in the element has
    this axis (two parts of one crystal with different axes are two declared materials)."""

    material_id: MaterialId
    direction: tuple[OpticalDecimal, OpticalDecimal, OpticalDecimal]

    @model_validator(mode="after")
    def _direction(self) -> CrystalOpticAxisV1:
        with localcontext(DECIMAL_CONTEXT):
            if sum(Decimal(c) * Decimal(c) for c in self.direction) == 0:
                raise ValueError(f"optic axis of {self.material_id}: the direction must not be zero")
        return self

    def unit(self) -> tuple[Decimal, Decimal, Decimal]:
        """The direction scaled to length 1 (exact decimal arithmetic in ``DECIMAL_CONTEXT``)."""

        with localcontext(DECIMAL_CONTEXT):
            x, y, z = (Decimal(c) for c in self.direction)
            norm = (x * x + y * y + z * z).sqrt()
            return x / norm, y / norm, z / norm


class ProjectOpticalModelV1(StrictContractModel):
    """An optical model declared in the Project.

    ``model``: the physics (see the module docstring). ``frame``: the element frame, with the catalog's conventions
    (``OpticalFrameV1``); absent, it follows from the model (a transmissive element with its first vertex at the
    origin; a reflector placed by its coated face; a polarization or neutral-density element at its plane).
    APERTURE_STOP, IDEAL_THIN_LENS, TABULATED_THIN_LENS, FARADAY_ROTATOR and COMPOUND models have a fixed frame and no
    ``frame``; a DIFFRACTION_GRATING's follows from its mode, and it states one only to name its ports (the designed order's
    exit, an ``APERTURE_CENTER`` frame with ``reference_axis`` ``PORTS``).
    ``design_geometry``: the element's outer body for the 3D views and physical checks; absent, it is derived from the
    model's outline and surfaces. ``power_ratings``: the element's power-handling ratings as the Project's author states
    them (``PowerRatingV1``, normally with status ``DECLARED``), compared by a run with the local exposure of the
    element. A COMPOUND unit states the ratings on its member models.
    ``optic_axes``: the optic axis (``CrystalOpticAxisV1``) of every uniaxial crystal among the media of a
    ``SURFACE_SEQUENCE``, ``SURFACE_SET`` or ``NONLINEAR_CRYSTAL`` model, one per crystal material, in the element frame;
    absent when the model uses no crystal. Every surface bounding a crystal is a plane.
    """

    model_id: ModelId
    display_name: AuthorName | None = None
    model: ProjectSimulationModelV1
    frame: OpticalFrameV1 | None = None
    design_geometry: DesignGeometryV1 | None = None
    power_ratings: tuple[PowerRatingV1, ...] | None = Field(None, min_length=1, max_length=MAX_POWER_RATINGS)
    optic_axes: tuple[CrystalOpticAxisV1, ...] | None = Field(None, min_length=1, max_length=MAX_OPTIC_AXES)

    @model_validator(mode="after")
    def _consistent(self) -> ProjectOpticalModelV1:
        if self.model.model_kind in FRAME_FREE_KINDS and self.frame is not None:
            raise ValueError(f"model {self.model_id}: a {self.model.model_kind} model has a fixed frame (omit frame)")
        if self.optic_axes is not None:
            if self.model.model_kind not in CRYSTAL_MODEL_KINDS:
                raise ValueError(f"model {self.model_id}: a {self.model.model_kind} model has no crystal medium (omit optic_axes)")
            duplicates = duplicate_keys(self.optic_axes, "material_id")
            if duplicates:
                raise ValueError(f"model {self.model_id}: optic_axes states material {', '.join(d[0] for d in duplicates)} more than once")
            unused = sorted({a.material_id for a in self.optic_axes} - set(self.material_ids()))
            if unused:
                raise ValueError(f"model {self.model_id}: optic_axes names {unused}, not a medium of the model")
        if isinstance(self.model, CompoundModelV1) and self.design_geometry is not None:
            raise ValueError(f"model {self.model_id}: a COMPOUND unit's outer boundary is its envelope (omit design_geometry)")
        if isinstance(self.model, CompoundModelV1) and self.power_ratings is not None:
            raise ValueError(f"model {self.model_id}: a COMPOUND unit's ratings belong to its member models (omit power_ratings)")
        check_power_ratings(self.power_ratings)
        return self

    def material_ids(self) -> tuple[str, ...]:
        if isinstance(self.model, (SurfaceSequenceModelV1, SurfaceSetModelV1, DiffractionGratingModelV1, FacetSolidModelV1,
                                   NonlinearCrystalModelV1, CompoundModelV1)):
            return self.model.material_ids()
        return ()

    def coating_ids(self) -> tuple[str, ...]:
        if isinstance(self.model, (SurfaceSequenceModelV1, SurfaceSetModelV1, FacetSolidModelV1, NonlinearCrystalModelV1)):
            return self.model.coating_ids()
        return ()

    def optic_axis(self, material_id: str) -> CrystalOpticAxisV1 | None:
        """The optic axis this model states for the crystal ``material_id`` (``None``: not stated)."""

        return next((a for a in self.optic_axes or () if a.material_id == material_id), None)

    @property
    def valid_band_nm(self) -> WavelengthBandV1 | None:
        return getattr(self.model, "valid_band_nm", None)


def _crystal_sides(physics) -> list[tuple[str, bool, tuple[str, ...]]]:
    """``(surface id, is a plane, its non-air media)`` of every surface of a surface model."""

    if isinstance(physics, SurfaceSequenceModelV1):
        last = len(physics.surfaces) - 1
        return [(s.surface_id, s.is_plane, tuple(m for m in ((physics.media[i - 1] if i > 0 else AIR),
                                                             (physics.media[i] if i < last else AIR)) if m not in RESERVED_MEDIA))
                for i, s in enumerate(physics.surfaces)]
    if isinstance(physics, SurfaceSetModelV1):
        return [(s.surface_id, s.is_plane, tuple(m for m in (s.medium_front, s.medium_back) if m not in RESERVED_MEDIA)) for s in physics.surfaces]
    return []


def _check_crystals(model: ProjectOpticalModelV1, material_by_id: dict[str, ProjectModelMaterialV1]) -> None:
    """Crystal rules of one model: a crystal is a medium of a surface model only, every crystal medium has its optic axis
    (``optic_axes``) and only crystals have one, and every surface bounding a crystal is a plane."""

    crystals = {m for m in model.material_ids() if material_by_id[m].is_crystal}
    if isinstance(model.model, NonlinearCrystalModelV1) and model.model.medium not in crystals:
        raise ValueError(f"model {model.model_id}: the nonlinear crystal's medium {model.model.medium} is not a crystal (it states no "
                         "birefringence)")
    if crystals and model.model.model_kind not in CRYSTAL_MODEL_KINDS:
        raise ValueError(f"model {model.model_id}: a {model.model.model_kind} model cannot use the crystal {sorted(crystals)[0]} "
                         "(a crystal is a medium of a SURFACE_SEQUENCE or SURFACE_SET model)")
    for axis in model.optic_axes or ():
        if axis.material_id not in crystals:
            raise ValueError(f"model {model.model_id}: optic_axes names {axis.material_id}, which is not a crystal (it states no "
                             "birefringence)")
    missing = sorted(m for m in crystals if model.optic_axis(m) is None)
    if missing:
        raise ValueError(f"model {model.model_id}: the crystal {missing[0]} needs its optic axis in optic_axes")
    physics = model.model
    if isinstance(physics, NonlinearCrystalModelV1) and physics.crystal_x_axis is not None:
        axis = model.optic_axis(physics.medium)
        if axis is not None:
            with localcontext(DECIMAL_CONTEXT):
                x = [Decimal(c) for c in physics.crystal_x_axis]
                dot = sum(a * b for a, b in zip(x, axis.unit())) / sum(c * c for c in x).sqrt()
            if abs(dot) > CRYSTAL_AXES_PERPENDICULAR_TOL:
                raise ValueError(f"model {model.model_id}: crystal_x_axis is not perpendicular to the optic axis of {physics.medium}")
    for surface_id, plane, media in _crystal_sides(model.model):
        crystal = next((m for m in media if m in crystals), None)
        if crystal is not None and not plane:
            raise ValueError(f"model {model.model_id}: surface {surface_id} bounds the crystal {crystal} and is curved; every face "
                             "of a crystal is a plane")


def _joined_face(model_id: str, member: CompoundMemberV1, surface_id: str, target: ProjectOpticalModelV1) -> SurfaceSetModelV1:
    """The member's model, checked to be a ``SURFACE_SET`` with a body whose surface ``surface_id`` is an uncoated plane face of
    that body traced ``TRANSMITTED`` (air in front, the body's medium behind)."""

    physics = target.model
    where = f"model {model_id}: joint face {member.member_id}.{surface_id}"
    if not isinstance(physics, SurfaceSetModelV1) or physics.body is None:
        raise ValueError(f"{where}: a joined member is a SURFACE_SET model with a body (member model {target.model_id} is "
                         f"{physics.model_kind})")
    surface = next((s for s in physics.surfaces if s.surface_id == surface_id), None)
    if surface is None:
        raise ValueError(f"{where}: model {target.model_id} has no surface {surface_id}")
    if not surface.is_plane or surface.traced is not TracedBranches.TRANSMITTED or surface.coating_id is not None:
        raise ValueError(f"{where}: a joined face is an uncoated plane traced TRANSMITTED")
    if surface.medium_front != AIR or surface.medium_back != physics.body.medium:
        raise ValueError(f"{where}: a joined face is a face of its member's body (air in front, {physics.body.medium} behind)")
    return physics


def _check_joints(model: ProjectOpticalModelV1, physics: CompoundModelV1, by_id: dict[str, ProjectOpticalModelV1],
                  material_by_id: dict[str, ProjectModelMaterialV1]) -> None:
    """A unit's joints: both faces are joinable body faces (``_joined_face``) and a cement is a declared isotropic material
    of one index throughout (no gradient index: the layer is traced as homogeneous) whose band covers both members' bands. Where the faces lie (coincident, or parallel across the layer) is checked when
    the unit is compiled, in the unit frame."""

    members = {m.member_id: m for m in physics.members}
    for joint in physics.joints or ():
        sides = [_joined_face(model.model_id, members[member_id], surface_id, by_id[members[member_id].model_id])
                 for member_id, surface_id in ((joint.member_a, joint.surface_a), (joint.member_b, joint.surface_b))]
        if joint.cement_material_id is None:
            continue
        cement = material_by_id.get(joint.cement_material_id)
        if cement is None:
            raise ValueError(f"model {model.model_id}: cement {joint.cement_material_id} is not declared in model_materials")
        if cement.is_crystal:
            raise ValueError(f"model {model.model_id}: cement {joint.cement_material_id} is a crystal; a cement layer is isotropic")
        if cement.is_gradient:
            raise ValueError(f"model {model.model_id}: cement {joint.cement_material_id} has a gradient index; a cement layer is "
                             "homogeneous (one index throughout)")
        for side in sides:
            if not side.valid_band_nm.within(cement.valid_band_nm):
                raise ValueError(f"model {model.model_id}: a joined member's band lies outside cement {joint.cement_material_id}'s band")


#: Wavelengths (band ends included) at which a gradient-index medium's index at the element's rim is checked to be at least 1.
GRADIENT_CHECK_SAMPLES = 65


def _check_gradients(model: ProjectOpticalModelV1, material_by_id: dict[str, ProjectModelMaterialV1]) -> None:
    """Gradient-index rules of one model. A gradient-index material (radial or axial) is the medium of one rod whose axis is
    the model's local Z axis:

    * a round ``SURFACE_SEQUENCE`` model: it fills one gap, between two surfaces that are planes or spheres / conics (no
      aspheric, cylindrical, axicon or two-variable terms) whose profile is defined out to the element's edge radius;
    * a ``SURFACE_SET`` model whose body is a round plate of the material (``side_radius_mm``): two plane surfaces bound it,
      at any slant (an angled end face).

    Its index stays real and at least 1 throughout the rod over the model's band (a radial medium out to the rod's radius, an
    axial one over the rod's axial extent)."""

    gradients = {m for m in model.material_ids() if material_by_id[m].is_gradient}
    if not gradients:
        return
    physics = model.model
    gradient = sorted(gradients)[0]
    if isinstance(physics, SurfaceSetModelV1):
        extents = _rod_in_set(model, physics, gradient)
    elif isinstance(physics, SurfaceSequenceModelV1):
        if physics.edge_radius_mm is None:
            raise ValueError(f"model {model.model_id}: an element of the gradient-index medium {gradient} is round (edge_radius_mm)")
        extents = {material_id: _rod_in_sequence(model, physics, material_id) for material_id in sorted(gradients)}
    else:
        raise ValueError(f"model {model.model_id}: a {physics.model_kind} model cannot use the gradient-index medium {gradient} "
                         "(it is the medium of a round SURFACE_SEQUENCE model or of the round body of a SURFACE_SET model, whose "
                         "local Z axis is the profile's axis)")
    for material_id in sorted(gradients):
        material = material_by_id[material_id]
        radius, _z_ref, z_low, z_high = extents[material_id]
        with localcontext(DECIMAL_CONTEXT):
            low, high = Decimal(physics.valid_band_nm.min_nm), Decimal(physics.valid_band_nm.max_nm)
            least = None if material.axial_gradient is None else material.axial_gradient.least_offset(z_low, z_high)
            for k in range(GRADIENT_CHECK_SAMPLES):
                wavelength = low + (high - low) * k / (GRADIENT_CHECK_SAMPLES - 1)
                n0 = _index_at(material.dispersion, wavelength)
                if least is not None:
                    if n0 + least < 1:
                        raise ValueError(f"model {model.model_id}: at {wavelength:.7g} nm the axial gradient-index medium {material_id} "
                                         f"falls below index 1 within the rod (between z = {z_low:.6g} and {z_high:.6g} mm)")
                    continue
                assert material.gradient is not None
                reach = material.gradient.gradient_at(wavelength) * radius
                if reach >= 1 or n0 * n0 * (1 - reach * reach) < 1:
                    raise ValueError(f"model {model.model_id}: at {wavelength:.7g} nm the gradient-index medium {material_id} falls "
                                     f"below index 1 within the element's edge ({radius} mm from the axis)")


def gradient_rod(model: ProjectOpticalModelV1, material_id: str) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """The rod of the gradient-index medium ``material_id`` in ``model`` (``_check_gradients`` holds): its radius, the local
    ``z_ref`` of its entrance (where ``z = 0`` of an axial gradient lies on local Z) and its axial extent from there, ``z_low``
    to ``z_high`` (mm)."""

    physics = model.model
    if isinstance(physics, SurfaceSetModelV1):
        return _rod_in_set(model, physics, material_id)[material_id]
    assert isinstance(physics, SurfaceSequenceModelV1)
    return _rod_in_sequence(model, physics, material_id)


def _rod_in_sequence(model: ProjectOpticalModelV1, physics: SurfaceSequenceModelV1, material_id: str
                     ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """The rod of a gradient-index medium in a round ``SURFACE_SEQUENCE``: its radius (the edge radius), its entrance (the
    vertex z of its front surface) and its axial extent ``z`` from there (the faces' sags out to that radius included)."""

    gaps = [i for i, m in enumerate(physics.media) if m == material_id]
    if len(gaps) > 1:
        raise ValueError(f"model {model.model_id}: the gradient-index medium {material_id} fills more than one gap; one element "
                         "holds one rod of it")
    front, back = physics.surfaces[gaps[0]], physics.surfaces[gaps[0] + 1]
    radius = Decimal(physics.edge_radius_mm)  # type: ignore[arg-type]
    sags = []
    for face in (front, back):
        if face.is_plane:
            sags.append((Decimal(0), Decimal(0)))
            continue
        if (face.aspheric_coefficients is not None or face.is_cylindrical or face.is_axicon or face.sag is not None
                or face.radius_mm is None):
            raise ValueError(f"model {model.model_id}: surface {face.surface_id} bounds the gradient-index medium {material_id} and is "
                             "not a plane, sphere or conic: a curved end face of a gradient-index rod has no aspheric, cylindrical, "
                             "axicon or two-variable terms")
        with localcontext(DECIMAL_CONTEXT):
            c = 1 / Decimal(face.radius_mm)
            k = Decimal(face.conic) if face.conic is not None else Decimal(0)
            inside = 1 - (1 + k) * c * c * radius * radius
            if inside <= 0:
                raise ValueError(f"model {model.model_id}: surface {face.surface_id} bounds the gradient-index medium {material_id} and "
                                 f"its profile ends inside the rod's radius ({physics.edge_radius_mm} mm)")
            sag = c * radius * radius / (1 + inside.sqrt())
        sags.append((min(Decimal(0), sag), max(Decimal(0), sag)))
    with localcontext(DECIMAL_CONTEXT):
        length = Decimal(back.vertex_z_mm) - Decimal(front.vertex_z_mm)
        z_low, z_high = sags[0][0], length + sags[1][1]
        if sags[0][1] >= length + sags[1][0]:
            raise ValueError(f"model {model.model_id}: the end faces {front.surface_id} and {back.surface_id} of the gradient-index "
                             f"medium {material_id} reach past each other along the axis within the rod's radius")
    return radius, Decimal(front.vertex_z_mm), z_low, z_high


def _rod_in_set(model: ProjectOpticalModelV1, physics: SurfaceSetModelV1, material_id: str
                ) -> dict[str, tuple[Decimal, Decimal, Decimal, Decimal]]:
    """The rod of a gradient-index medium in a ``SURFACE_SET``: the model's round body made of it, bounded by two plane
    surfaces; its radius (the body's side radius) and its axial extent ``z`` from where the body's face toward -Z crosses
    local Z (the faces' slant out to that radius included)."""

    body = physics.body
    if body is None or body.medium != material_id or body.side_radius_mm is None:
        raise ValueError(f"model {model.model_id}: a SURFACE_SET model uses the gradient-index medium {material_id} as the medium of "
                         "its round body (side_radius_mm), whose local Z axis is the profile's axis")
    faces = [s for s in physics.surfaces if material_id in (s.medium_front, s.medium_back)]
    if len(faces) != 2 or any(s.medium_front == s.medium_back for s in faces):
        raise ValueError(f"model {model.model_id}: the round body of the gradient-index medium {material_id} is bounded by exactly two "
                         "surfaces with the medium on one side (its end faces)")
    curved = next((s for s in faces if not s.is_plane), None)
    if curved is not None:
        raise ValueError(f"model {model.model_id}: surface {curved.surface_id} bounds the gradient-index medium {material_id} and is "
                         "not a plane: the end faces of a rod in a SURFACE_SET model are planes (at any slant)")
    radius = Decimal(body.side_radius_mm)
    reach: list[tuple[Decimal, Decimal]] = []
    with localcontext(DECIMAL_CONTEXT):
        for face in faces:
            nx, ny, nz = (Decimal(c) for c in face.normal)
            if nz == 0:
                raise ValueError(f"model {model.model_id}: surface {face.surface_id} contains the rod's axis")
            ox, oy, oz = (Decimal(c) for c in face.origin_mm)
            z_axis = oz + (nx * ox + ny * oy) / nz
            spread = radius * (nx * nx + ny * ny).sqrt() / abs(nz)
            reach.append((z_axis - spread, z_axis + spread))
        reach.sort()
        if reach[0][1] >= reach[1][0]:
            raise ValueError(f"model {model.model_id}: the end faces of the gradient-index medium {material_id} reach past each other "
                             "along the axis within the rod's radius")
        start = (reach[0][0] + reach[0][1]) / 2
        return {material_id: (radius, start, reach[0][0] - start, reach[1][1] - start)}


def _index_at(dispersion: object, wavelength_nm: Decimal) -> Decimal:
    """A declared index law at a vacuum wavelength (exact decimals; a tabulated law by straight lines between its points,
    which bounds its monotone PCHIP interpolation's range)."""

    if isinstance(dispersion, ConstantIndexDispersionV1):
        return Decimal(dispersion.refractive_index)
    if isinstance(dispersion, SellmeierDispersionV1):
        lam_sq = (wavelength_nm / 1000) ** 2
        return (1 + sum(Decimal(b) * lam_sq / (lam_sq - Decimal(c)) for b, c in zip(dispersion.b, dispersion.c_um2, strict=True))).sqrt()
    assert isinstance(dispersion, TabulatedDispersionV1)
    rows = list(zip((Decimal(w) for w in dispersion.wavelength_nm), (Decimal(n) for n in dispersion.refractive_index)))
    for (w0, n0), (w1, n1) in zip(rows, rows[1:]):
        if w0 <= wavelength_nm <= w1:
            return min(n0, n1)
    return min(n for _w, n in rows)


def check_optical_models(
    models: tuple[ProjectOpticalModelV1, ...],
    materials: tuple[ProjectModelMaterialV1, ...],
    coatings: tuple[PublicCoatingV1, ...],
) -> None:
    """Cross-reference checks of a Project's model library (raise ``ValueError``): unique ids, every compound member and
    every medium / coating resolves, member models are not COMPOUND, and each model's band lies inside the bands of the
    materials and coatings it uses and inside a coating's phase table (the catalog's own rules, including
    ``TYPICAL_SPECTRAL_CURVE``). Every model kind is checked: the thin kinds (polarization, neutral-density, aperture
    stop, ideal and tabulated thin lens, Faraday rotator) use no material or coating, and a diffraction grating only its
    media."""

    for label, items, key in (("model_id", models, "model_id"), ("material_id", materials, "material_id"),
                              ("coating_id", coatings, "coating_id")):
        duplicates = duplicate_keys(items, key)
        if duplicates:
            raise ValueError(f"duplicate {label}: {duplicates}")
    by_id = {model.model_id: model for model in models}
    material_by_id = {material.material_id: material for material in materials}
    coating_by_id = {coating.coating_id: coating for coating in coatings}
    for model in models:
        physics = model.model
        if isinstance(physics, CompoundModelV1):
            for member in physics.members:
                target = by_id.get(member.model_id)
                if target is None:
                    raise ValueError(f"model {model.model_id}: member {member.member_id} uses unknown model {member.model_id!r}")
                if isinstance(target.model, CompoundModelV1):
                    raise ValueError(f"model {model.model_id}: member {member.member_id} is itself a COMPOUND unit")
            _check_joints(model, physics, by_id, material_by_id)
            continue
        band = model.valid_band_nm
        for material_id in model.material_ids():
            material = material_by_id.get(material_id)
            if material is None:
                raise ValueError(f"model {model.model_id}: material {material_id} is not declared in model_materials")
            if band is not None and not band.within(material.valid_band_nm):
                raise ValueError(f"model {model.model_id}: the model band lies outside material {material_id}'s band")
        typical = False
        for coating_id in model.coating_ids():
            coating = coating_by_id.get(coating_id)
            if coating is None:
                raise ValueError(f"model {model.model_id}: coating {coating_id} is not declared in model_coatings")
            low, high = coating.span_nm
            if band is not None and not (low <= Decimal(band.min_nm) and Decimal(band.max_nm) <= high):
                raise ValueError(f"model {model.model_id}: the model band lies outside coating {coating_id}'s bands")
            phase_span = coating.phase.span_nm if coating.phase is not None else None
            if band is not None and phase_span is not None and not (phase_span[0] <= Decimal(band.min_nm) and Decimal(band.max_nm) <= phase_span[1]):
                raise ValueError(f"model {model.model_id}: the model band lies outside coating {coating_id}'s phase table")
            typical = typical or coating.is_typical
        _check_crystals(model, material_by_id)
        _check_gradients(model, material_by_id)
        approximations = getattr(physics, "approximations", None)
        if approximations is not None and not typical and SimulationApproximation.TYPICAL_SPECTRAL_CURVE in approximations:
            raise ValueError(
                f"model {model.model_id}: TYPICAL_SPECTRAL_CURVE is listed only when a coating of the model is a typical curve"
            )


#: The approximation labels that follow from a declared model itself (``derived_approximations``): a Project model may
#: list them or leave them out; the published catalog lists each exactly when it holds.
DERIVED_APPROXIMATIONS = (SimulationApproximation.TYPICAL_SPECTRAL_CURVE, SimulationApproximation.UNTRACED_FACES_BLOCK)


def derived_approximations(model: ProjectOpticalModelV1, coatings: tuple[PublicCoatingV1, ...] = ()) -> frozenset[SimulationApproximation]:
    """The labels of ``DERIVED_APPROXIMATIONS`` that hold for ``model``: ``TYPICAL_SPECTRAL_CURVE`` when a coating it uses
    (of ``coatings``) is a typical curve, ``UNTRACED_FACES_BLOCK`` when it is a ``SURFACE_SET`` whose body has a face that
    carries no traced surface."""

    by_id = {coating.coating_id: coating for coating in coatings}
    labels: set[SimulationApproximation] = set()
    if any(by_id[c].is_typical for c in model.coating_ids() if c in by_id):
        labels.add(SimulationApproximation.TYPICAL_SPECTRAL_CURVE)
    if isinstance(model.model, SurfaceSetModelV1) and model.model.has_untraced_faces:
        labels.add(SimulationApproximation.UNTRACED_FACES_BLOCK)
    return frozenset(labels)


__all__ = [
    "MAX_ARRAY_CELLS",
    "MAX_ARRAYED_MEMBER_ID",
    "MAX_COMPOUND_CELLS",
    "MAX_CEMENT_THICKNESS_MM",
    "MAX_COMPOUND_JOINTS",
    "MAX_COMPOUND_MEMBERS",
    "MAX_FACET_GROUPS",
    "MAX_FARADAY_ROWS",
    "MAX_GRADIENT_ROWS",
    "MAX_GRATING_ORDERS",
    "MAX_GRATING_RESPONSE_ROWS",
    "MAX_OPTIC_AXES",
    "MAX_QPM_ORDER",
    "MAX_THERMO_OPTIC_ROWS",
    "NONLINEAR_FACE_IDS",
    "MAX_THIN_LENS_ROWS",
    "THIN_LENS_POINT_TOLERANCE",
    "ApertureStopModelV1",
    "CRYSTAL_MODEL_KINDS",
    "CompoundMemberV1",
    "CompoundModelV1",
    "CrystalThermoOpticV1",
    "ConstantIndexDispersionV1",
    "CrystalOpticAxisV1",
    "DesignIntentV1",
    "DiffractionGratingModelV1",
    "FacetGroupV1",
    "FacetSolidModelV1",
    "EvidenceToolV1",
    "FRAME_FREE_KINDS",
    "GRADIENT_CHECK_SAMPLES",
    "GradientRowV1",
    "FaradayResponseRowV1",
    "FaradayRotatorModelV1",
    "GratingAbsorptionRowV1",
    "GratingOrderResponseV1",
    "GratingResponseRowV1",
    "IdealThinLensModelV1",
    "ComplexTransmissionSampleV1",
    "SampleMapV1",
    "SamplePhaseTermV1",
    "MemberArrayV1",
    "MemberId",
    "MemberJointV1",
    "ModelId",
    "NonlinearCrystalModelV1",
    "ProjectDispersionV1",
    "ProjectModelMaterialV1",
    "PropagationEnvironmentV1",
    "DEFAULT_ENVIRONMENT",
    "ProjectOpticalModelV1",
    "ProjectSimulationModelV1",
    "RadialGradientIndexV1",
    "ResponseEvidenceV1",
    "TabulatedThinLensModelV1",
    "ThermoOpticRowV1",
    "ThinLensResponseRowV1",
    "UniaxialBirefringenceV1",
    "ValidatedMetricV1",
    "DERIVED_APPROXIMATIONS",
    "check_optical_models",
    "derived_approximations",
]
