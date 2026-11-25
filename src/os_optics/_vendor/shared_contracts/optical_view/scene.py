"""`OpticalSimulationSceneV1`: the renderer-independent scene of the local Optical Simulation View.

The view answers "what is happening optically?". It holds the optical content of the design:

* optical elements, placed at their Project ``nominal_pose`` and drawn with their public design geometry;
* sources;
* observation / target planes, with the Engine's beam footprint on them;
* the Engine's traced beam paths: chief ray plus pupil sample rays, as global polylines;
* optical-axis markers, scene bounds and a camera hint.

Element geometry comes from the public optical catalog, from an optical model the Project declares
(``ProjectDocumentV2.optical_models``), or is a symbolic view marker (``extra="forbid"`` everywhere, closed optical roles).

Physics authority: every ray point is an Engine point (the Engine's own trace, recorded ray by ray), bound to the Engine run by
``engine_run.input_hash`` / ``engine_run.trace_hash``. The only non-Engine coordinates are the explicitly separate
``display_extension_to_mm`` stubs that draw a free (escaped) ray a fixed display length beyond its last Engine point
along the Engine's own exit direction.

Frame: the Project design frame (right-handed, mm, x right, y forward, z up; ``x_global = R(q) x_local + p`` with
Hamilton ``(w, x, y, z)`` quaternions). Element local +Z is the optical/reference axis.

Consumers: the headless GLB exporter (``os_optics.optical_view``) and the browser (Three.js/WebGL) Mode B view read
this same document; its JSON Schema is printed by ``os-optics schema optical-scene`` (the ``V1`` of the class
name is the contract family; the document version is ``schema_version``).

``payload()`` omits ``None`` fields; ``content_hash()`` is osc1 of that payload.

Version: ``schema_version`` 3.0 is the only version. Every public catalog product is drawn with ONE generic body
kind, ``OPTIC_BODY``: the product's public design geometry plus the optical surfaces of the simulation model the run
lowered (surface sequence, surface set or polarization element) and, for a surface set, its solid body. The body
is a bounding cylinder, box or convex polyhedron (the product's design geometry); surfaces carry their full profile (plane,
sphere, conic, even asphere, cylinder) and clear aperture (circle or rectangle, optionally decentred). Every beam path carries the Engine's power at its end and a
display role per chief-ray segment (``PRIMARY`` / ``SECONDARY``), so a viewer draws a weak branch such as a coating
leak or a ghost reflection thinner and fainter than the main beam instead of over it. An observation plane carries one
footprint per Engine crossing: a beam path that meets the same plane more than once (a double pass, a fold back through
the plane, a retro-reflection) has one footprint per crossing, numbered by ``crossing_index``. Each plane states whether
it absorbs the light that reaches it (``absorbing_extent``). A beam whose chief ray fell on a central obstruction says
from which chief-ray segment on that ray is only the beam's reference axis (``chief_reference_from``): the light is the
annulus its pupil rays trace. Earlier scene versions are not read.

Declared optical models (optional fields, absent from every scene without one): an element whose component uses a
Project-declared model carries ``optical_model_id`` instead of an ``optical_sku``. A single model is drawn as an
``OPTIC_BODY`` with ``basis = PROJECT_DECLARED_MODEL``: no catalog ``category`` / ``form``, its stated or derived body
(a body of zero thickness -- ``z_min_mm == z_max_mm`` -- for a thin element without a stated body), and, besides the
catalog's model kinds, ``APERTURE_STOP`` (one ``stop`` plane: the opening as clear aperture, its central obstruction, the
plate outline as edge), ``IDEAL_THIN_LENS`` (one ``lens`` plane with its focal length) and ``TABULATED_THIN_LENS`` (one
``lens`` plane; its focal length depends on the wavelength, so the plane states none). A surface may carry an axicon term
(``linear_sag_coefficient``, optional: absent from every scene without one). A ``COMPOUND`` unit is ONE
element whose geometry is an ``OPTICAL_UNIT``: the unit's envelope and ports in its frame and one ``OPTIC_BODY`` per member
at the member's pose in the unit frame.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from ..base import OmitAbsentFields, StrictContractModel
from ..common.geometry import PoseV1, Vector3
from ..common.hashes import HashProfile, HashRef, Sha256Digest, osc1_digest, require_engine_quantized_params, require_hash_profile
from ..common.identifiers import BuildVersion, InstanceKey, OpenSciId
from ..common.sag_shapes import sag_form, sag_value
from ..design_catalog.enums import CoatingKind, FrameReferenceAxis, ModelFidelity, OpticalCategory, OpticalForm
from ..design_catalog.optical_catalog import (
    FACE_NORMAL_FORMS,
    FORMS_BY_CATEGORY,
    MODEL_KINDS_BY_CATEGORY,
    POLARIZING_PRISM_FORMS,
    PORTS_FORMS,
    RESERVED_MEDIA,
    CatalogVersion,
    MaterialId,
    OpticalSku,
    PolarizationElementKind,
    SagShapeV1,
    design_roles_for,
)
from ..optical_design.design_content import ApertureV1
from ..optical_design.component import ModelId, OpticalRoleV1, require_unit_quaternion
from ..common.meshes import MAX_MESH_TRIANGLES, MAX_MESH_VERTICES
from ..optical_design.optical_model import MAX_FACET_GROUPS, MemberId
from ..optical_design.local_validation import ENGINE_PACKAGE_NAME, EnginePackageName, ValidatedProjectRefV1


OPTICAL_SIMULATION_SCENE_SCHEMA_ID: Literal["opensci.optical_simulation_scene"] = "opensci.optical_simulation_scene"
OPTICAL_SIMULATION_SCENE_SCHEMA_VERSION: Literal["3.0"] = "3.0"
OPTICAL_SIMULATION_VIEW_KIND: Literal["OPTICAL_SIMULATION_VIEW"] = "OPTICAL_SIMULATION_VIEW"
OPTICAL_SIMULATION_VIEW_NOTICE = (
    "Optical Simulation View of the design: optical elements, sources, Engine-traced beam paths and observation "
    "planes. Element bodies are drawn with their design geometry."
)
ENGINE_INPUT_HASH_SUBJECT = "engine_input"
ENGINE_RESULT_HASH_SUBJECT = "engine_result"

Rgb = Annotated[tuple[float, float, float], Field(description="Display sRGB triple in [0, 1].")]

#: The longest Engine beam path id (``opensci_engine.optics.rays.path_ids.PATH_ID_MAX_LENGTH``).
BEAM_PATH_ID_MAX_LENGTH = 256
BeamPathId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=BEAM_PATH_ID_MAX_LENGTH,
        pattern=r"^[^\s/\x00-\x1f\x7f-\x9f][^/\x00-\x1f\x7f-\x9f]*$",
    ),
    Field(description="Engine beam path id (OpticalPath.path_id): '<source_id>#<spectral index>' and one '.<k>' per branch; "
                      "the Engine keeps the branch part at most 96 characters (a deeper branch's part is '~' and a 128-bit "
                      "digest, then '.<k>' again), so an id is at most 256 characters for a source id of at most 128."),
]


def _has_duplicates(ids: list[str] | tuple[str, ...]) -> bool:
    return len(set(ids)) != len(ids)


def _finite(values: tuple[float, ...], where: str) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{where} must be finite")


# ------------------------------------------------------------------------------------------------ geometry


class GeometryStatusV1(StrEnum):
    """What an element body is drawn from."""

    NOMINAL = "NOMINAL"  # design geometry: the public catalog's, or a declared model's (nominal body)
    PLACEHOLDER = "PLACEHOLDER"  # neutral symbolic marker chosen by the view


class GeometryBasisV1(StrEnum):
    """``PUBLIC_CATALOG_DESIGN_GEOMETRY``: a catalog product's public model and design geometry.
    ``PROJECT_DECLARED_MODEL``: an optical model the Project declares and its stated or derived body.
    ``VIEW_PLACEHOLDER``: a symbolic marker."""

    PUBLIC_CATALOG_DESIGN_GEOMETRY = "PUBLIC_CATALOG_DESIGN_GEOMETRY"
    VIEW_PLACEHOLDER = "VIEW_PLACEHOLDER"
    PROJECT_DECLARED_MODEL = "PROJECT_DECLARED_MODEL"


class OpticSurfaceRoleV1(StrEnum):
    """What an optical surface of an ``OPTIC_BODY`` does to the light the run traces (from the published model).

    ``REFRACT``: a transmitting interface (catalog ``traced = TRANSMITTED``); ``REFLECT``: a reflector (``REFLECTED``);
    ``SPLIT``: a splitting surface, both branches traced (``TRANSMITTED_AND_REFLECTED``); ``POLARIZATION_ELEMENT``: the
    plane of a thin Jones-matrix retarder or linear polarizer (catalog ``POLARIZATION_ELEMENT`` model); ``APERTURE_STOP``:
    the plane of a declared aperture stop (light inside the opening passes, light on the plate is blocked);
    ``THIN_LENS``: the plane of a declared ideal thin lens; ``ATTENUATOR``: the plane of a neutral-density attenuator
    (``NEUTRAL_DENSITY`` model: its optical density may vary across the plane); ``FARADAY_ROTATOR``: the plane of a declared
    nonreciprocal polarization rotator; ``GRATING``: the ruled plane of a declared diffraction grating.
    """

    REFRACT = "REFRACT"
    REFLECT = "REFLECT"
    SPLIT = "SPLIT"
    POLARIZATION_ELEMENT = "POLARIZATION_ELEMENT"
    APERTURE_STOP = "APERTURE_STOP"
    THIN_LENS = "THIN_LENS"
    ATTENUATOR = "ATTENUATOR"
    FARADAY_ROTATOR = "FARADAY_ROTATOR"
    GRATING = "GRATING"


#: The Engine / scene id of the one plane of a ``POLARIZATION_ELEMENT`` model (surface-sequence surfaces are S1, S2, ...).
POLARIZATION_ELEMENT_SURFACE_ID = "element"
#: The Engine / scene ids of the one plane of a declared ``APERTURE_STOP`` and ``IDEAL_THIN_LENS`` model.
APERTURE_STOP_SURFACE_ID = "stop"
THIN_LENS_SURFACE_ID = "lens"
#: The Engine / scene id of the one plane of a ``NEUTRAL_DENSITY`` attenuator.
ATTENUATOR_SURFACE_ID = "attenuator"
#: The Engine / scene id of the one plane of a declared ``FARADAY_ROTATOR``.
FARADAY_ROTATOR_SURFACE_ID = "rotator"
#: The Engine / scene id of the ruled plane of a declared ``DIFFRACTION_GRATING``.
GRATING_SURFACE_ID = "grating"
#: The single-plane surface id of each single-plane surface role.
_PLANE_SURFACE_IDS = {"POLARIZATION_ELEMENT": POLARIZATION_ELEMENT_SURFACE_ID, "APERTURE_STOP": APERTURE_STOP_SURFACE_ID,
                      "THIN_LENS": THIN_LENS_SURFACE_ID, "ATTENUATOR": ATTENUATOR_SURFACE_ID, "FARADAY_ROTATOR": FARADAY_ROTATOR_SURFACE_ID,
                      "GRATING": GRATING_SURFACE_ID}
OpticSurfaceId = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(S[1-9][0-9]?|element|stop|lens|attenuator|rotator|grating)$"),
    Field(description="The model's surface label 'S1', 'S2', ... (surface sequence / set), 'element' (polarization element plane), "
                      "'stop' (aperture stop plane), 'lens' (ideal thin lens plane), 'attenuator' (neutral-density plane), 'rotator' "
                      "(Faraday rotator plane) or 'grating' (diffraction grating plane)."),
]
#: Optic-body model kinds: the public catalog's simulation model kinds and the single-plane kinds a Project declares.
OPTIC_MODEL_KINDS = ("SURFACE_SEQUENCE", "SURFACE_SET", "POLARIZATION_ELEMENT", "APERTURE_STOP", "IDEAL_THIN_LENS",
                     "TABULATED_THIN_LENS", "NEUTRAL_DENSITY", "FARADAY_ROTATOR", "DIFFRACTION_GRATING", "FACET_SOLID")
#: The model kinds a public catalog body may carry.
CATALOG_OPTIC_MODEL_KINDS = ("SURFACE_SEQUENCE", "SURFACE_SET", "POLARIZATION_ELEMENT", "NEUTRAL_DENSITY")
#: The surface role of each single-plane model kind.
_PLANE_ROLE_BY_KIND = {"POLARIZATION_ELEMENT": "POLARIZATION_ELEMENT", "APERTURE_STOP": "APERTURE_STOP", "IDEAL_THIN_LENS": "THIN_LENS",
                       "TABULATED_THIN_LENS": "THIN_LENS", "NEUTRAL_DENSITY": "ATTENUATOR", "FARADAY_ROTATOR": "FARADAY_ROTATOR",
                       "DIFFRACTION_GRATING": "GRATING"}
_SEQUENCE_ROLES = frozenset({"REFRACT", "REFLECT", "SPLIT"})
# Per catalog category: surface roles a body may carry, and the roles it must carry at least one of (``None``: no demand).
# A transmissive category never carries a reflector; a mirror reflects (a partial reflector also transmits); a splitter
# splits; a waveplate / polarizer is one Jones plane.
_CATEGORY_SURFACE_ROLES: dict[OpticalCategory, tuple[frozenset[str], frozenset[str] | None]] = {
    OpticalCategory.LENS: (frozenset({"REFRACT", "SPLIT"}), None),
    OpticalCategory.WINDOW: (frozenset({"REFRACT", "SPLIT"}), None),
    OpticalCategory.FILTER: (frozenset({"REFRACT", "SPLIT", "ATTENUATOR"}), None),
    OpticalCategory.MIRROR: (frozenset({"REFRACT", "REFLECT", "SPLIT"}), frozenset({"REFLECT", "SPLIT"})),
    OpticalCategory.BEAMSPLITTER: (frozenset({"REFRACT", "SPLIT"}), frozenset({"SPLIT"})),
    OpticalCategory.WAVEPLATE: (frozenset({"POLARIZATION_ELEMENT"}), frozenset({"POLARIZATION_ELEMENT"})),
    OpticalCategory.POLARIZER: (frozenset({"POLARIZATION_ELEMENT"}), frozenset({"POLARIZATION_ELEMENT"})),
    OpticalCategory.PRISM: (frozenset({"REFRACT", "REFLECT", "SPLIT"}), None),
}
#: Surface roles a body of a polarizing prism carries (its crystal faces refract; an optical contact is one of them).
_PRISM_BODY_SURFACE_ROLES = frozenset({"REFRACT"})
# The Jones element a polarization category publishes.
_POLARIZATION_ELEMENT_BY_CATEGORY = {OpticalCategory.WAVEPLATE: PolarizationElementKind.RETARDER,
                                     OpticalCategory.POLARIZER: PolarizationElementKind.LINEAR_POLARIZER}


class OpticSurfaceV1(StrictContractModel):
    """One optical surface of an ``OPTIC_BODY``, in the element frame (from the simulation model the run used).

    Placement: a coaxial surface (surface sequence, polarization element) has its vertex at ``vertex_z_mm`` on local Z;
    a posed surface (surface set) has its vertex at ``origin_mm``, its axis ``normal`` and its local X ``u_axis`` (local Y
    is ``normal x u_axis``). Profile: ``radius_mm`` the signed vertex radius of curvature (positive: the centre of
    curvature lies on the +axis side of the vertex; absent: a plane), ``conic`` and ``aspheric_coefficients`` (A4, A6,
    ...) of a conic / even asphere, ``linear_sag_coefficient`` the axicon term ``a rho`` (a cone, or on a cylinder a roof),
    ``cylinder_axis`` the local axis a cylinder is straight along, ``sag`` a two-variable sag shape (Zernike, XY polynomial,
    biconic or toroid, as the model publishes it). Clear aperture: a
    circle of ``clear_radius_mm`` or a rectangle of ``clear_half_widths_mm`` (local X / Y), centred on the axis or at
    ``aperture_center_mm``; ``clear_outline_mm`` (a closed ring of local X / Y points, counter-clockwise) draws a clear
    aperture of another shape within it (an ellipse, a D shape or a polygon; round parts as chords within the view's
    drawing tolerance), and ``hole_outlines_mm`` one ring per hole through the element (no surface inside it). ``coating_kind`` is the published coating's kind (absent: an uncoated interface). An aperture
    stop's plane may carry a central ``obstruction_radius_mm`` (a circular opening), an ideal thin lens's plane its
    ``focal_length_mm`` (a tabulated thin lens's plane none: its focal length depends on the wavelength). A surface of a
    surface sequence may state its own mechanical edge, ``edge_radius_mm`` (a round element) or ``edge_half_widths_mm`` (a
    rectangular one), centred on the axis and inside the body outline (absent: it reaches the outline); the edges of two
    surfaces bounding one glass nest. Beyond its edge the surface continues flat at its rim height out to the outline of
    the glass it bounds. A display annotation of the model, not a second optical model: the Engine's surfaces are bound to
    the same model by the view builder.
    """

    surface_id: OpticSurfaceId
    role: OpticSurfaceRoleV1
    vertex_z_mm: float | None = None
    origin_mm: Vector3 | None = None
    normal: Vector3 | None = None
    u_axis: Vector3 | None = None
    radius_mm: float | None = None
    conic: float | None = None
    aspheric_coefficients: tuple[float, ...] | None = Field(None, min_length=1, max_length=8)
    linear_sag_coefficient: float | None = None
    cylinder_axis: Literal["X", "Y"] | None = None
    sag: SagShapeV1 | None = None
    clear_radius_mm: float | None = Field(None, gt=0)
    clear_half_widths_mm: tuple[float, float] | None = None
    aperture_center_mm: tuple[float, float] | None = None
    clear_outline_mm: tuple[tuple[float, float], ...] | None = Field(None, min_length=3)
    hole_outlines_mm: tuple[tuple[tuple[float, float], ...], ...] | None = Field(None, min_length=1)
    edge_radius_mm: float | None = Field(None, gt=0)
    edge_half_widths_mm: tuple[float, float] | None = None
    coating_kind: CoatingKind | None = None
    obstruction_radius_mm: float | None = Field(None, gt=0)
    focal_length_mm: float | None = None

    @model_validator(mode="after")
    def _check(self) -> OpticSurfaceV1:
        numbers = [v for v in (self.vertex_z_mm, self.radius_mm, self.conic, self.clear_radius_mm, self.obstruction_radius_mm,
                               self.focal_length_mm, self.linear_sag_coefficient, self.edge_radius_mm) if v is not None]
        for group in (self.origin_mm, self.normal, self.u_axis, self.aspheric_coefficients, self.clear_half_widths_mm, self.aperture_center_mm,
                      self.edge_half_widths_mm):
            numbers.extend(group or ())
        rings = ([self.clear_outline_mm] if self.clear_outline_mm is not None else []) + list(self.hole_outlines_mm or ())
        for ring in rings:
            if len(ring) < 3:
                raise ValueError(f"surface {self.surface_id}: an outline ring has at least 3 points")
            numbers.extend(value for point in ring for value in point)
        _finite(tuple(numbers), f"surface {self.surface_id}")
        posed = (self.origin_mm, self.normal, self.u_axis)
        if (self.vertex_z_mm is None) == all(v is None for v in posed) or (any(v is None for v in posed) and not all(v is None for v in posed)):
            raise ValueError(f"surface {self.surface_id}: give either vertex_z_mm (a coaxial surface) or origin_mm, normal and u_axis (a posed surface)")
        if (self.clear_radius_mm is None) == (self.clear_half_widths_mm is None):
            raise ValueError(f"surface {self.surface_id}: give exactly one of clear_radius_mm and clear_half_widths_mm")
        if self.clear_half_widths_mm is not None and min(self.clear_half_widths_mm) <= 0:
            raise ValueError(f"surface {self.surface_id}: clear_half_widths_mm must be positive")
        for role, plane_id in _PLANE_SURFACE_IDS.items():
            if (self.surface_id == plane_id) != (self.role.value == role):
                raise ValueError(f"surface {self.surface_id!r}: exactly the {role.lower().replace('_', ' ')} plane has id {plane_id!r}")
        if self.role.value in _PLANE_SURFACE_IDS and (not self.is_plane or self.is_posed):
            raise ValueError(f"a {self.role.value.lower().replace('_', ' ')} is a plane on the element axis (vertex_z_mm)")
        if self.obstruction_radius_mm is not None and (self.role is not OpticSurfaceRoleV1.APERTURE_STOP or self.clear_radius_mm is None
                                                       or self.obstruction_radius_mm >= self.clear_radius_mm):
            raise ValueError(f"surface {self.surface_id}: obstruction_radius_mm is a central disc smaller than an aperture stop's circular opening")
        if (self.focal_length_mm is not None and self.role is not OpticSurfaceRoleV1.THIN_LENS) or self.focal_length_mm == 0:
            raise ValueError(f"surface {self.surface_id}: only a thin lens plane has a focal_length_mm, and it is non-zero")
        if self.linear_sag_coefficient == 0:
            raise ValueError(f"surface {self.surface_id}: omit linear_sag_coefficient for a surface without an axicon term")
        if self.conic is not None and self.radius_mm is None:
            raise ValueError(f"surface {self.surface_id}: a conic profile needs radius_mm")
        if self.cylinder_axis is not None and self.radius_mm is None and self.linear_sag_coefficient is None:
            raise ValueError(f"surface {self.surface_id}: a cylindrical profile needs radius_mm or linear_sag_coefficient")
        if self.is_spherical and self.clear_radius_mm is not None and self.aperture_center_mm is None and abs(self.radius_mm) < self.clear_radius_mm:  # type: ignore[arg-type]
            raise ValueError(f"surface {self.surface_id}: |radius_mm| must be at least clear_radius_mm")
        if self.coating_kind is CoatingKind.UNCOATED:
            raise ValueError("an uncoated surface omits coating_kind")
        if self.edge_radius_mm is not None or self.edge_half_widths_mm is not None:
            if self.edge_radius_mm is not None and self.edge_half_widths_mm is not None:
                raise ValueError(f"surface {self.surface_id}: give at most one of edge_radius_mm and edge_half_widths_mm")
            if self.is_posed or self.role.value in _PLANE_SURFACE_IDS:
                raise ValueError(f"surface {self.surface_id}: only a surface-sequence surface states its own edge")
            if self.edge_half_widths_mm is not None:
                if min(self.edge_half_widths_mm) <= 0:
                    raise ValueError(f"surface {self.surface_id}: edge_half_widths_mm must be positive")
                cx, cy = self.aperture_center_mm or (0.0, 0.0)
                hx, hy = (self.clear_radius_mm, self.clear_radius_mm) if self.clear_radius_mm is not None else self.clear_half_widths_mm  # type: ignore[misc]
                if abs(cx) + hx > self.edge_half_widths_mm[0] + _Z_ALLOWANCE_MM or abs(cy) + hy > self.edge_half_widths_mm[1] + _Z_ALLOWANCE_MM:
                    raise ValueError(f"surface {self.surface_id}: clear aperture exceeds its edge")
            elif self.aperture_extent_mm() > self.edge_radius_mm + _Z_ALLOWANCE_MM:  # type: ignore[operator]
                raise ValueError(f"surface {self.surface_id}: clear aperture exceeds its edge")
        return self

    def edge_half_widths(self, outline: tuple[float, float]) -> tuple[float, float]:
        """Half widths ``(x, y)`` of the surface's edge (a round edge: its radius twice): its own, else ``outline``."""

        if self.edge_radius_mm is not None:
            return self.edge_radius_mm, self.edge_radius_mm
        if self.edge_half_widths_mm is not None:
            return self.edge_half_widths_mm
        return outline

    @property
    def is_plane(self) -> bool:
        return self.radius_mm is None and self.aspheric_coefficients is None and self.linear_sag_coefficient is None and self.sag is None

    @property
    def is_spherical(self) -> bool:
        """A sphere: a radius, no conic, no polynomial, axicon or two-variable terms, not cylindrical."""

        return (self.radius_mm is not None and self.conic is None and self.aspheric_coefficients is None and self.cylinder_axis is None
                and self.linear_sag_coefficient is None and self.sag is None)

    @property
    def is_posed(self) -> bool:
        return self.origin_mm is not None

    def profile_sag(self, rho_sq_mm2: float) -> float | None:
        """Sag at profile coordinate ``rho`` given as ``rho^2`` (``None`` outside the profile's domain)."""

        sag = 0.0
        if self.radius_mm is not None:
            curvature = 1.0 / self.radius_mm
            arg = 1.0 - (1.0 + (self.conic or 0.0)) * curvature * curvature * rho_sq_mm2
            if arg < 0.0:
                return None
            sag = curvature * rho_sq_mm2 / (1.0 + math.sqrt(arg))
        power = rho_sq_mm2 * rho_sq_mm2
        for coefficient in self.aspheric_coefficients or ():
            sag += coefficient * power
            power *= rho_sq_mm2
        if self.linear_sag_coefficient is not None:
            sag += self.linear_sag_coefficient * math.sqrt(rho_sq_mm2)
        return sag

    def sag_at(self, x_mm: float, y_mm: float = 0.0) -> float | None:
        """Sag (offset along the surface axis from the vertex) at the local point ``(x, y)``; for a rotationally symmetric
        profile ``sag_at(r)`` is the sag at transverse radius ``r`` (``None`` outside the profile's domain)."""

        rho_sq = y_mm * y_mm if self.cylinder_axis == "X" else x_mm * x_mm if self.cylinder_axis == "Y" else x_mm * x_mm + y_mm * y_mm
        if self.sag is None:
            return self.profile_sag(rho_sq)
        form = sag_form(self.sag)
        z, ok = sag_value(form, x_mm, y_mm)
        if not ok:
            return None
        if not form.additive:
            return z
        base = self.profile_sag(rho_sq)
        return None if base is None else base + z

    def aperture_extent_mm(self) -> float:
        """Largest distance of a clear-aperture point from the surface axis."""

        cx, cy = self.aperture_center_mm or (0.0, 0.0)
        if self.clear_radius_mm is not None:
            return math.hypot(cx, cy) + self.clear_radius_mm
        hx, hy = self.clear_half_widths_mm  # type: ignore[misc]
        return math.hypot(abs(cx) + hx, abs(cy) + hy)


class SolidFaceV1(StrictContractModel):
    """One bounding plane of a solid body: the half-space ``outward_normal . (x - point_mm) <= 0`` (element frame)."""

    point_mm: Vector3
    outward_normal: Vector3

    @model_validator(mode="after")
    def _check(self) -> SolidFaceV1:
        _finite((*self.point_mm, *self.outward_normal), "solid face")
        return self


class SolidBodyV1(StrictContractModel):
    """The convex solid a surface-set element's light travels in (from the public model): the intersection of ``faces``
    and, for a round plate, the cylinder of ``side_radius_mm`` about local Z, filled with ``medium``."""

    medium: MaterialId
    faces: tuple[SolidFaceV1, ...] = Field(min_length=2, max_length=32)
    side_radius_mm: float | None = Field(None, gt=0)


class PolarizationAxisV1(StrictContractModel):
    """The in-plane axis of a polarization element: local +X of the element frame (a retarder's fast axis, a linear
    polarizer's transmission axis). Turning the element about local Z turns it."""

    element: PolarizationElementKind
    axis_local: Literal["+X"] = "+X"


class OpticGratingV1(StrictContractModel):
    """What a diffraction grating's body draws: ``mode`` (``REFLECTION``: the ruled face at z = 0 with the plate behind it at
    z < 0; ``TRANSMISSION``: the plate before it), the ``period_nm`` and the grating vector in local X / Y (the rules run
    across it)."""

    mode: Literal["REFLECTION", "TRANSMISSION"]
    period_nm: float = Field(gt=0)
    grating_vector_local: tuple[float, float]

    @model_validator(mode="after")
    def _check(self) -> OpticGratingV1:
        _finite((self.period_nm, *self.grating_vector_local), "grating")
        return self


class OpticFacetGroupV1(StrictContractModel):
    """Facets of a ``FACET_SOLID`` body with their own surface: ``REFRACT`` (a coated refracting face, ``coating_kind`` its
    coating), ``REFLECT`` (a mirrored face) or ``ABSORB`` (a blackened face)."""

    triangles: tuple[int, ...] = Field(min_length=1, max_length=MAX_MESH_TRIANGLES)
    role: Literal["REFRACT", "REFLECT", "ABSORB"]
    coating_kind: CoatingKind | None = None


class OpticFacetsV1(StrictContractModel):
    """The flat triangular facets of a ``FACET_SOLID`` body in the element frame: ``vertices_mm``, ``triangles``
    (counter-clockwise seen from outside, closed around the solid of ``medium``), what they stand for (``approximation``,
    with ``max_chord_deviation_mm`` for a tessellation of curved surfaces), the coating kind of the facets outside any group
    (``coating_kind``; absent: bare) and the facet ``groups`` with their own surface. The run traces the facets themselves
    (edge-connected coplanar facets of one group as one planar face). At most ``MAX_MESH_VERTICES`` corners,
    ``MAX_MESH_TRIANGLES`` triangles and ``MAX_FACET_GROUPS`` groups, as the ``FACET_SOLID`` model it draws."""

    vertices_mm: tuple[Vector3, ...] = Field(min_length=4, max_length=MAX_MESH_VERTICES)
    triangles: tuple[tuple[int, int, int], ...] = Field(min_length=4, max_length=MAX_MESH_TRIANGLES)
    approximation: Literal["FACETED_EXACT", "TESSELLATED_FROM_CAD"]
    max_chord_deviation_mm: float | None = Field(None, ge=0)
    medium: MaterialId
    coating_kind: CoatingKind | None = None
    groups: tuple[OpticFacetGroupV1, ...] = Field((), max_length=MAX_FACET_GROUPS)

    @model_validator(mode="after")
    def _check(self) -> OpticFacetsV1:
        _finite(tuple(c for v in self.vertices_mm for c in v), "facets")
        count = len(self.vertices_mm)
        if any(not 0 <= i < count for t in self.triangles for i in t):
            raise ValueError(f"facets: a triangle names a vertex outside 0 .. {count - 1}")
        if any(not 0 <= k < len(self.triangles) for g in self.groups for k in g.triangles):
            raise ValueError(f"facets: a group names a triangle outside 0 .. {len(self.triangles) - 1}")
        if (self.approximation == "TESSELLATED_FROM_CAD") != (self.max_chord_deviation_mm is not None):
            raise ValueError("facets: max_chord_deviation_mm is stated for TESSELLATED_FROM_CAD facets and only for them")
        return self


class OpticBodyGeometryV1(OmitAbsentFields, StrictContractModel):
    """An optical element in the element frame: one body kind for every category, catalog product or declared model.

    The body is the product's public design geometry (``shape``): a bounding ``CYLINDER`` of ``diameter_mm`` about local Z
    from ``z_min_mm`` to ``z_max_mm``, a ``BOX`` centred at ``center_mm`` with ``half_extents_mm`` along local X, Y, Z, or a
    ``CONVEX_POLYHEDRON`` with corners ``vertices_mm``. The optical annotations are the simulation model the run used
    (``model_kind`` / ``fidelity``):

    * ``SURFACE_SEQUENCE``: coaxial ``surfaces`` S1, S2, ... ordered along local Z with the ``media`` between them
      (``media[i]`` fills ``surfaces[i]`` .. ``surfaces[i+1]``; ``AIR`` for an air gap), inside a round outline of
      ``edge_radius_mm`` or a rectangular one of ``edge_half_widths_mm``; outside the element is air;
    * ``SURFACE_SET``: posed ``surfaces`` S1, S2, ... and, when the model has one, its ``solid`` body;
    * ``POLARIZATION_ELEMENT``: one ``element`` plane, its ``edge_radius_mm`` and its in-plane ``polarization`` axis (local
      +X).

    * ``APERTURE_STOP`` (declared): one ``stop`` plane: its opening (clear aperture, optionally decentred, with an
      ``obstruction_radius_mm``) inside the plate outline ``edge_radius_mm`` / ``edge_half_widths_mm``;
    * ``IDEAL_THIN_LENS`` (declared): one ``lens`` plane with its ``focal_length_mm``, clear radius and edge radius;
    * ``TABULATED_THIN_LENS`` (declared): one ``lens`` plane with its clear radius and edge radius (no single focal length);
    * ``NEUTRAL_DENSITY``: one ``attenuator`` plane, its clear aperture inside the outline ``edge_radius_mm`` /
      ``edge_half_widths_mm``;
    * ``FARADAY_ROTATOR`` (declared): one ``rotator`` plane with its clear radius and edge radius, its field axis local +Z;
    * ``DIFFRACTION_GRATING`` (declared): one ``grating`` plane, its clear aperture inside the outline ``edge_radius_mm`` /
      ``edge_half_widths_mm``, and ``grating`` (its mode, period and grating vector);
    * ``FACET_SOLID`` (declared): no ``surfaces``; its ``facets`` (``OpticFacetsV1``) inside the body (the design geometry, by
      default the box around the facets).

    ``reference_axis = FACE_NORMAL`` marks a reflector / plate splitter frame: local +Z is the outward normal of the coated
    face and the substrate lies at z < 0; ``PORTS`` an element placed by its ports (a cube splitter, a prism, an off-axis
    mirror); absent, local +Z is a transmissive element's optical axis and its first surface vertex is the origin.

    ``basis``: ``PUBLIC_CATALOG_DESIGN_GEOMETRY`` (a catalog product: ``category`` and ``form`` given, checked against the
    catalog's rules) or ``PROJECT_DECLARED_MODEL`` (a declared model: no ``category`` / ``form``; a thin element without a
    stated body has a body of zero thickness, ``z_min_mm == z_max_mm`` or a box of zero z extent).
    """

    # Serialized only when given: a document without it keeps its earlier payload and hash.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("facets",)
    # The JSON Schema states the surface rule of ``_check`` too: a FACET_SOLID body has its facets and no surfaces, every
    # other body at least one surface.
    model_config = ConfigDict(json_schema_extra={"allOf": [{
        "if": {"properties": {"model_kind": {"const": "FACET_SOLID"}}, "required": ["model_kind"]},
        "then": {"required": ["facets"], "properties": {"surfaces": {"maxItems": 0}}},
        "else": {"required": ["surfaces"], "properties": {"surfaces": {"minItems": 1}}},
    }]})

    kind: Literal["OPTIC_BODY"] = "OPTIC_BODY"
    shape: Literal["CYLINDER", "BOX", "CONVEX_POLYHEDRON"] = "CYLINDER"
    diameter_mm: float | None = Field(None, gt=0)
    z_min_mm: float | None = None
    z_max_mm: float | None = None
    center_mm: Vector3 | None = None
    half_extents_mm: Vector3 | None = None
    vertices_mm: tuple[Vector3, ...] | None = Field(None, min_length=4, max_length=32)
    category: OpticalCategory | None = None
    form: OpticalForm | None = None
    reference_axis: FrameReferenceAxis | None = None
    model_kind: Literal["SURFACE_SEQUENCE", "SURFACE_SET", "POLARIZATION_ELEMENT", "APERTURE_STOP", "IDEAL_THIN_LENS", "TABULATED_THIN_LENS",
                        "NEUTRAL_DENSITY", "FARADAY_ROTATOR", "DIFFRACTION_GRATING", "FACET_SOLID"]
    fidelity: ModelFidelity | None = None
    edge_radius_mm: float | None = Field(None, gt=0)
    edge_half_widths_mm: tuple[float, float] | None = None
    surfaces: tuple[OpticSurfaceV1, ...] = Field((), description="The model's surfaces (at least one; none for a FACET_SOLID body).")
    media: tuple[MaterialId, ...] = ()
    solid: SolidBodyV1 | None = None
    polarization: PolarizationAxisV1 | None = None
    grating: OpticGratingV1 | None = None
    facets: OpticFacetsV1 | None = None
    status: Literal[GeometryStatusV1.NOMINAL] = GeometryStatusV1.NOMINAL
    basis: Literal[GeometryBasisV1.PUBLIC_CATALOG_DESIGN_GEOMETRY, GeometryBasisV1.PROJECT_DECLARED_MODEL] = GeometryBasisV1.PUBLIC_CATALOG_DESIGN_GEOMETRY

    @property
    def is_declared(self) -> bool:
        return self.basis is GeometryBasisV1.PROJECT_DECLARED_MODEL

    @model_validator(mode="after")
    def _check(self) -> OpticBodyGeometryV1:
        self._check_shape()
        roles = [s.role.value for s in self.surfaces]
        if self.is_declared:
            if self.category is not None or self.form is not None:
                raise ValueError("a declared model's body has no catalog category or form")
        else:
            category = self.category
            if category is None or self.form is None:
                raise ValueError("a public catalog body states its category and form")
            if self.model_kind not in CATALOG_OPTIC_MODEL_KINDS or self.fidelity is None:
                raise ValueError(f"a public catalog body carries a catalog model kind and its fidelity, not {self.model_kind}")
            if self.form not in FORMS_BY_CATEGORY.get(category, frozenset()):
                raise ValueError(f"form {self.form} is not a {category} form")
            prism_body = self.form in POLARIZING_PRISM_FORMS
            if prism_body and self.model_kind != "SURFACE_SET":
                raise ValueError(f"a body of a {self.form} is one of its SURFACE_SET bodies (drawn in its OPTICAL_UNIT)")
            if not prism_body and self.model_kind not in MODEL_KINDS_BY_CATEGORY.get(category, frozenset()):
                raise ValueError(f"a {category} body cannot carry a {self.model_kind} model")
            if (self.form in FACE_NORMAL_FORMS) != (self.reference_axis is FrameReferenceAxis.FACE_NORMAL):
                raise ValueError(f"a {self.form} body {'has' if self.form in FACE_NORMAL_FORMS else 'does not have'} the FACE_NORMAL frame")
            if (self.form in PORTS_FORMS) != (self.reference_axis is FrameReferenceAxis.PORTS):
                raise ValueError(f"a {self.form} body {'is' if self.form in PORTS_FORMS else 'is not'} placed by its ports (PORTS frame)")
            allowed, required = (_PRISM_BODY_SURFACE_ROLES, None) if prism_body else _CATEGORY_SURFACE_ROLES[category]
            if set(roles) - allowed:
                raise ValueError(f"a {category} body cannot carry {sorted(set(roles) - allowed)} surfaces")
            if required is not None and not set(roles) & required:
                raise ValueError(f"a {category} body needs a {'/'.join(sorted(required))} surface")
        if (self.fidelity is None) != (self.model_kind == "APERTURE_STOP"):
            raise ValueError("every body but an aperture stop states its model's fidelity")
        if self.model_kind != "SURFACE_SET" and self.solid is not None:
            raise ValueError("only a SURFACE_SET body carries a solid")
        if (self.grating is not None) != (self.model_kind == "DIFFRACTION_GRATING"):
            raise ValueError("exactly a DIFFRACTION_GRATING body states its grating")
        if (self.facets is not None) != (self.model_kind == "FACET_SOLID"):
            raise ValueError("exactly a FACET_SOLID body states its facets")
        if self.model_kind == "FACET_SOLID":
            if not self.is_declared or self.surfaces or self.media or self.polarization is not None:
                raise ValueError("a FACET_SOLID body is a declared model's: its facets, no surfaces, no media and no polarization axis")
            if self.edge_radius_mm is not None or self.edge_half_widths_mm is not None:
                raise ValueError("a FACET_SOLID body has no edge outline (its facets bound it)")
            return self
        if not self.surfaces:
            raise ValueError(f"a {self.model_kind} body has at least one surface")
        if self.model_kind == "SURFACE_SET":
            if any(not s.is_posed for s in self.surfaces) or self.media or self.polarization is not None:
                raise ValueError("a SURFACE_SET body has posed surfaces S1, S2, ..., no media and no polarization axis")
            if self.edge_radius_mm is not None or self.edge_half_widths_mm is not None:
                raise ValueError("a SURFACE_SET body has no edge outline (its solid bounds it)")
            if [s.surface_id for s in self.surfaces] != [f"S{i}" for i in range(1, len(self.surfaces) + 1)]:
                raise ValueError("surfaces must be labelled S1, S2, ... in order")
            return self
        if any(s.is_posed for s in self.surfaces):
            raise ValueError(f"a {self.model_kind} body has coaxial surfaces (vertex_z_mm)")
        if (self.edge_radius_mm is None) == (self.edge_half_widths_mm is None):
            raise ValueError("give exactly one of edge_radius_mm and edge_half_widths_mm")
        if self.edge_half_widths_mm is not None:
            _finite(self.edge_half_widths_mm, "edge_half_widths_mm")
            if self.model_kind not in ("SURFACE_SEQUENCE", "APERTURE_STOP", "NEUTRAL_DENSITY", "DIFFRACTION_GRATING") or min(self.edge_half_widths_mm) <= 0:
                raise ValueError("a rectangular outline has positive edge_half_widths_mm and belongs to a SURFACE_SEQUENCE, APERTURE_STOP, "
                                 "NEUTRAL_DENSITY or DIFFRACTION_GRATING body")
        if self.shape == "CYLINDER" and self.edge_radius_mm is not None and self.edge_radius_mm > self.diameter_mm / 2:  # type: ignore[operator]
            raise ValueError("edge_radius_mm exceeds the body radius")
        low, high = self._z_range()
        for surface in self.surfaces:
            if self.edge_radius_mm is not None and surface.aperture_extent_mm() > self.edge_radius_mm + _Z_ALLOWANCE_MM:
                raise ValueError(f"surface {surface.surface_id}: clear radius exceeds the edge radius")
            # (a two-variable sag shape is not referenced to its vertex: the surface crosses the axis at its sag there)
            on_axis = surface.vertex_z_mm if surface.sag is None else surface.vertex_z_mm + (surface.sag_at(0.0) or 0.0)  # type: ignore[operator]
            if low is not None and not low - _Z_ALLOWANCE_MM <= on_axis <= high + _Z_ALLOWANCE_MM:  # type: ignore[operator]
                raise ValueError(f"surface {surface.surface_id}: vertex outside the body")
        for surface in self.surfaces:
            if (self.edge_radius_mm is not None and surface.edge_half_widths_mm is not None
                    or self.edge_half_widths_mm is not None and surface.edge_radius_mm is not None):
                raise ValueError(f"surface {surface.surface_id}: its edge has the shape of the body outline")
            outline = self.outline_half_widths()
            edge = surface.edge_half_widths(outline)
            if edge[0] > outline[0] + _Z_ALLOWANCE_MM or edge[1] > outline[1] + _Z_ALLOWANCE_MM:
                raise ValueError(f"surface {surface.surface_id}: its edge exceeds the body outline")
        for k, medium in enumerate(self.media[: len(self.surfaces) - 1]):
            (ax, ay), (bx, by) = self.surface_edge(k), self.surface_edge(k + 1)
            if medium != "AIR" and not (ax <= bx and ay <= by or ax >= bx and ay >= by):
                raise ValueError(f"surfaces {self.surfaces[k].surface_id} and {self.surfaces[k + 1].surface_id} bound one glass: their "
                                 "edges nest (one contains the other on both axes)")
        if self.model_kind == "SURFACE_SEQUENCE":
            if set(roles) - _SEQUENCE_ROLES or self.polarization is not None:
                raise ValueError("a SURFACE_SEQUENCE body has S1, S2, ... surfaces and no polarization axis")
            if [s.surface_id for s in self.surfaces] != [f"S{i}" for i in range(1, len(self.surfaces) + 1)]:
                raise ValueError("surfaces must be labelled S1, S2, ... in order")
            z = [s.vertex_z_mm for s in self.surfaces]
            if any(b <= a for a, b in zip(z, z[1:])):  # type: ignore[operator]
                raise ValueError("surfaces must be ordered by strictly increasing vertex_z_mm")
            if len(self.media) != len(self.surfaces) - 1:
                raise ValueError("media must have exactly one entry per gap between consecutive surfaces")
            if self.reference_axis is FrameReferenceAxis.FACE_NORMAL and self.surfaces[-1].vertex_z_mm != 0.0:
                raise ValueError("a FACE_NORMAL body's coated face is its last surface, at z = 0")
        elif self.model_kind == "POLARIZATION_ELEMENT":
            if len(self.surfaces) != 1 or roles != ["POLARIZATION_ELEMENT"] or self.media or self.polarization is None or self.edge_radius_mm is None:
                raise ValueError("a POLARIZATION_ELEMENT body has one 'element' plane, its edge radius, no media and its polarization axis")
            if not self.is_declared and self.polarization.element is not _POLARIZATION_ELEMENT_BY_CATEGORY[self.category]:  # type: ignore[index]
                raise ValueError(f"a {self.category} body is a {_POLARIZATION_ELEMENT_BY_CATEGORY[self.category]} element")  # type: ignore[index]
        else:
            role = _PLANE_ROLE_BY_KIND[self.model_kind]
            if len(self.surfaces) != 1 or roles != [role] or self.media or self.polarization is not None:
                raise ValueError(f"a {self.model_kind} body has one {_PLANE_SURFACE_IDS[role]!r} plane, no media and no polarization axis")
            if role == "THIN_LENS" and (self.surfaces[0].focal_length_mm is None) != (self.model_kind == "TABULATED_THIN_LENS"):
                raise ValueError("an ideal thin lens plane states its non-zero focal_length_mm; a tabulated thin lens plane states none")
        return self

    def _check_shape(self) -> None:
        thin = self.is_declared  # a declared thin element without a stated body has zero thickness
        cylinder = (self.diameter_mm, self.z_min_mm, self.z_max_mm)
        box = (self.center_mm, self.half_extents_mm)
        given = {"CYLINDER": all(v is not None for v in cylinder), "BOX": all(v is not None for v in box),
                 "CONVEX_POLYHEDRON": self.vertices_mm is not None}
        stray = {"CYLINDER": any(v is not None for v in cylinder), "BOX": any(v is not None for v in box),
                 "CONVEX_POLYHEDRON": self.vertices_mm is not None}
        if not given[self.shape] or any(stray[other] for other in stray if other != self.shape):
            raise ValueError(f"a {self.shape} body gives exactly its own extent fields")
        if self.shape == "CYLINDER":
            _finite(cylinder, "OPTIC_BODY extent")  # type: ignore[arg-type]
            if not (self.z_max_mm > self.z_min_mm or (thin and self.z_max_mm == self.z_min_mm)):  # type: ignore[operator]
                raise ValueError("OPTIC_BODY needs z_max_mm > z_min_mm")
        elif self.shape == "BOX":
            _finite((*self.center_mm, *self.half_extents_mm), "OPTIC_BODY extent")  # type: ignore[misc]
            if min(self.half_extents_mm[:2]) <= 0 or not (self.half_extents_mm[2] > 0 or (thin and self.half_extents_mm[2] == 0)):  # type: ignore[index]
                raise ValueError("OPTIC_BODY half extents must be positive")
        else:
            _finite(tuple(c for v in self.vertices_mm for c in v), "OPTIC_BODY extent")  # type: ignore[union-attr]

    def _z_range(self) -> tuple[float | None, float | None]:
        """The body's local z range (``None`` for a polyhedron: its corners are drawn as given)."""

        if self.shape == "CYLINDER":
            return self.z_min_mm, self.z_max_mm
        if self.shape == "BOX":
            return self.center_mm[2] - self.half_extents_mm[2], self.center_mm[2] + self.half_extents_mm[2]  # type: ignore[index]
        return None, None

    def outline_half_widths(self) -> tuple[float, float]:
        """Half widths ``(x, y)`` of the body's edge outline (a round outline: its radius twice)."""

        if self.edge_radius_mm is not None:
            return self.edge_radius_mm, self.edge_radius_mm
        return self.edge_half_widths_mm  # type: ignore[return-value]

    def surface_edge(self, index: int) -> tuple[float, float]:
        """Half widths of surface ``index``'s edge (its own, else the outline), as ``SurfaceSequenceModelV1.surface_edge_mm``."""

        return self.surfaces[index].edge_half_widths(self.outline_half_widths())

    def glass_outline(self, index: int) -> tuple[float, float]:
        """Half widths of the outline of the glass surface ``index`` bounds (``SurfaceSequenceModelV1.glass_outline_mm``)."""

        edges = [self.surface_edge(index)]
        if index > 0 and self.media[index - 1] not in RESERVED_MEDIA:
            edges.append(self.surface_edge(index - 1))
        if index < len(self.media) and self.media[index] not in RESERVED_MEDIA:
            edges.append(self.surface_edge(index + 1))
        return max(e[0] for e in edges), max(e[1] for e in edges)

    def gap_outline(self, gap: int) -> tuple[float, float]:
        """Half widths of the outline of gap ``gap`` (``SurfaceSequenceModelV1.gap_outline_mm``)."""

        if self.media[gap] not in RESERVED_MEDIA:
            a, b = self.surface_edge(gap), self.surface_edge(gap + 1)
            return max(a[0], b[0]), max(a[1], b[1])
        a, b = self.glass_outline(gap), self.glass_outline(gap + 1)
        return min(a[0], b[0]), min(a[1], b[1])

    @property
    def has_surface_edges(self) -> bool:
        """Some surface states its own edge (the surfaces do not all reach the outline)."""

        return any(s.edge_radius_mm is not None or s.edge_half_widths_mm is not None for s in self.surfaces)

    @property
    def design_roles(self) -> frozenset[str] | None:
        """Project ``optical_role`` values an element of this body's category and form may play (``None`` for a declared
        model's body: the runtime checks its role against the model kind)."""

        if self.category is None or self.form is None:
            return None
        return design_roles_for(self.category, self.form)


# A surface vertex may sit on the design body's end face; allow float noise of the Decimal -> float conversion.
_Z_ALLOWANCE_MM = 1e-9


class EmittingAreaV1(StrictContractModel):
    """The true emitting surface of an extended emitter, centred on the element origin in the local XY plane."""

    shape: Literal["RECTANGLE", "DISC"]
    size_x_mm: float = Field(gt=0, description="Full width along local +X (a disc's diameter), in mm.")
    size_y_mm: float = Field(gt=0, description="Full height along local +Y (a disc's diameter), in mm.")


class SourceEmitterGeometryV1(StrictContractModel):
    """Symbolic emitter: a disc of ``emitter_radius_mm`` at the element origin, facing local +Z, with a short
    ``marker_length_mm`` body behind it (toward local -Z). An extended emitter also states its ``emitting_area``, which is
    drawn at its true shape and size; ``emitter_radius_mm`` then encloses it."""

    kind: Literal["SOURCE_EMITTER"] = "SOURCE_EMITTER"
    emitter_radius_mm: float = Field(gt=0)
    marker_length_mm: float = Field(gt=0)
    emitting_area: EmittingAreaV1 | None = None
    status: Literal[GeometryStatusV1.PLACEHOLDER] = GeometryStatusV1.PLACEHOLDER
    basis: Literal[GeometryBasisV1.VIEW_PLACEHOLDER] = GeometryBasisV1.VIEW_PLACEHOLDER


class ElementMarkerGeometryV1(StrictContractModel):
    """Symbolic thin disc for an optical element that has no public design geometry (e.g. a custom part)."""

    kind: Literal["ELEMENT_MARKER"] = "ELEMENT_MARKER"
    radius_mm: float = Field(gt=0)
    thickness_mm: float = Field(gt=0)
    status: Literal[GeometryStatusV1.PLACEHOLDER] = GeometryStatusV1.PLACEHOLDER
    basis: Literal[GeometryBasisV1.VIEW_PLACEHOLDER] = GeometryBasisV1.VIEW_PLACEHOLDER


class UnitPortV1(StrictContractModel):
    """An external optical port of a ``COMPOUND`` unit in the unit frame: ``IN`` (the nominal input travels along
    ``direction`` through ``point_mm``) or an ``OUT``/``OUT_<NAME>`` exit (from the declared model's ``ports``)."""

    port_id: str = Field(min_length=1, max_length=32, pattern=r"^(IN|OUT|OUT_[A-Z0-9]+)$")
    role: Literal["INPUT", "OUTPUT"]
    point_mm: Vector3
    direction: Vector3

    @model_validator(mode="after")
    def _check(self) -> UnitPortV1:
        _finite((*self.point_mm, *self.direction), f"port {self.port_id}")
        return self


class UnitMemberV1(OmitAbsentFields, StrictContractModel):
    """One member of an ``OPTICAL_UNIT``: its pose in the unit frame (member frame -> unit frame) and its ``OPTIC_BODY`` in the
    member frame. A ``COMPOUND`` unit's member names its declared model (``optical_model_id``; ``basis =
    PROJECT_DECLARED_MODEL``); a body of a catalog polarizing prism (a ``JOINED_BODIES`` model) names none and is drawn with
    the product's public model (``basis = PUBLIC_CATALOG_DESIGN_GEOMETRY``, the product's category and form)."""

    # Serialized only when given: a COMPOUND unit's member always names its model, so its payload is unchanged.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("optical_model_id",)

    member_id: MemberId
    optical_model_id: ModelId | None = None
    label: str = Field(min_length=1, max_length=200)
    pose: PoseV1
    body: OpticBodyGeometryV1

    @model_validator(mode="after")
    def _check(self) -> UnitMemberV1:
        require_unit_quaternion(self.pose, f"member {self.member_id!r} pose")
        if (self.optical_model_id is not None) != self.body.is_declared:
            raise ValueError(f"member {self.member_id!r}: a member of a declared unit names its model and is drawn with it "
                             "(PROJECT_DECLARED_MODEL); a body of a catalog product names none and is drawn with the public model")
        return self


class OpticalUnitGeometryV1(OmitAbsentFields, StrictContractModel):
    """A ``COMPOUND`` unit declared in the Project (an objective, a relay), or a catalog polarizing prism made of joined bodies
    (a ``JOINED_BODIES`` model): ONE element placed as a whole. ``shape`` and its extent fields are the unit's envelope in the
    unit frame (as for ``OPTIC_BODY``: a ``CYLINDER``, a ``BOX`` or a ``CONVEX_POLYHEDRON``; a catalog product's: its public
    design geometry), ``ports`` its external optical ports, ``members`` every member body at its pose in the unit frame; the
    Engine simulates every surface of every member. ``basis``: ``PROJECT_DECLARED_MODEL`` (a declared unit: no category or
    form) or ``PUBLIC_CATALOG_DESIGN_GEOMETRY`` (a catalog product: its ``category`` and ``form``, every member drawn with the
    public model)."""

    # Serialized only when given: a declared unit's payload is unchanged.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("category", "form")

    kind: Literal["OPTICAL_UNIT"] = "OPTICAL_UNIT"
    shape: Literal["CYLINDER", "BOX", "CONVEX_POLYHEDRON"] = "CYLINDER"
    diameter_mm: float | None = Field(None, gt=0)
    z_min_mm: float | None = None
    z_max_mm: float | None = None
    center_mm: Vector3 | None = None
    half_extents_mm: Vector3 | None = None
    vertices_mm: tuple[Vector3, ...] | None = Field(None, min_length=4, max_length=32)
    ports: tuple[UnitPortV1, ...] | None = Field(None, min_length=1)
    members: tuple[UnitMemberV1, ...] = Field(min_length=1, max_length=64)
    status: Literal[GeometryStatusV1.NOMINAL] = GeometryStatusV1.NOMINAL
    basis: Literal[GeometryBasisV1.PROJECT_DECLARED_MODEL, GeometryBasisV1.PUBLIC_CATALOG_DESIGN_GEOMETRY] = GeometryBasisV1.PROJECT_DECLARED_MODEL
    category: OpticalCategory | None = None
    form: OpticalForm | None = None

    @property
    def is_declared(self) -> bool:
        return self.basis is GeometryBasisV1.PROJECT_DECLARED_MODEL

    @property
    def design_roles(self) -> frozenset[str] | None:
        """Project ``optical_role`` values a catalog product drawn as this unit may play (``None`` for a declared unit)."""

        if self.category is None or self.form is None:
            return None
        return design_roles_for(self.category, self.form)

    @model_validator(mode="after")
    def _check(self) -> OpticalUnitGeometryV1:
        cylinder = (self.diameter_mm, self.z_min_mm, self.z_max_mm)
        box = (self.center_mm, self.half_extents_mm)
        given = {"CYLINDER": all(v is not None for v in cylinder), "BOX": all(v is not None for v in box),
                 "CONVEX_POLYHEDRON": self.vertices_mm is not None}
        stray = {"CYLINDER": any(v is not None for v in cylinder), "BOX": any(v is not None for v in box),
                 "CONVEX_POLYHEDRON": self.vertices_mm is not None}
        if not given[self.shape] or any(stray[other] for other in stray if other != self.shape):
            raise ValueError(f"a {self.shape} envelope gives exactly its own extent fields")
        if self.is_declared:
            if self.category is not None or self.form is not None:
                raise ValueError("a declared unit has no catalog category or form")
        else:
            if self.form not in POLARIZING_PRISM_FORMS or self.form not in FORMS_BY_CATEGORY.get(self.category, frozenset()):  # type: ignore[arg-type]
                raise ValueError("a catalog OPTICAL_UNIT is a polarizing prism product: its category and one of its forms")
            if any(m.body.category is not self.category or m.body.form is not self.form for m in self.members):
                raise ValueError("every body of a catalog OPTICAL_UNIT is drawn with the product's category and form")
        if any(m.body.is_declared != self.is_declared for m in self.members):
            raise ValueError("every member of an OPTICAL_UNIT is drawn on the unit's basis")
        if _has_duplicates([m.member_id for m in self.members]):
            raise ValueError("duplicate member_id in an OPTICAL_UNIT")
        if _has_duplicates([p.port_id for p in self.ports or ()]):
            raise ValueError("duplicate port_id in an OPTICAL_UNIT")
        return self


ElementGeometryV1 = Annotated[OpticBodyGeometryV1 | SourceEmitterGeometryV1 | ElementMarkerGeometryV1 | OpticalUnitGeometryV1,
                              Field(discriminator="kind")]


# ------------------------------------------------------------------------------------------------ elements


class SceneElementV1(StrictContractModel):
    """One optical element of the Project (component ``instance_id``), at its nominal pose."""

    element_id: InstanceKey
    optical_role: OpticalRoleV1
    optical_sku: OpticalSku | None = Field(None, description="Public OpenSci optical identity, when the component is a catalog product.")
    label: str = Field(min_length=1, max_length=200, description="Display label: the public catalog display name, the declared model's "
                                                                  "name, or the role.")
    pose: PoseV1
    geometry: ElementGeometryV1
    optical_model_id: ModelId | None = Field(None, description="The Project-declared optical model the element is drawn from (instead of a SKU).")

    @model_validator(mode="after")
    def _check(self) -> SceneElementV1:
        require_unit_quaternion(self.pose, f"element {self.element_id!r} pose")
        geometry = self.geometry
        declared = isinstance(geometry, (OpticalUnitGeometryV1, OpticBodyGeometryV1)) and geometry.is_declared
        if declared:
            if self.optical_model_id is None or self.optical_sku is not None:
                raise ValueError(f"element {self.element_id!r}: a declared model's geometry names its optical_model_id (and no optical_sku)")
        elif self.optical_model_id is not None:
            raise ValueError(f"element {self.element_id!r}: an optical_model_id is drawn with its declared model's geometry")
        if isinstance(geometry, OpticBodyGeometryV1) and not geometry.is_declared and geometry.form in POLARIZING_PRISM_FORMS:
            raise ValueError(f"element {self.element_id!r}: a {geometry.form} product is drawn as the OPTICAL_UNIT of its bodies")
        if isinstance(geometry, (OpticalUnitGeometryV1, OpticBodyGeometryV1)) and not geometry.is_declared:
            if self.optical_sku is None:
                raise ValueError(f"element {self.element_id!r}: public catalog geometry requires its optical_sku")
            if self.optical_role.value not in geometry.design_roles:  # type: ignore[operator]
                raise ValueError(
                    f"element {self.element_id!r}: a {geometry.category} body cannot play optical_role {self.optical_role.value!r} "
                    f"(a {geometry.category} product plays {sorted(geometry.design_roles)})"  # type: ignore[arg-type]
                )
        elif self.optical_sku is not None:
            raise ValueError(f"element {self.element_id!r}: a public catalog product is drawn as its OPTIC_BODY (a polarizing prism: "
                             "its OPTICAL_UNIT)")
        if isinstance(geometry, SourceEmitterGeometryV1) and self.optical_role is not OpticalRoleV1.SOURCE:
            raise ValueError(f"element {self.element_id!r}: a SOURCE_EMITTER body is only for source elements")
        return self


# ------------------------------------------------------------------------------------------- observations


class BeamFootprintV1(StrictContractModel):
    """One Engine arrival of a beam path on an observation plane (from ``ValidationResult.observations``).

    ``crossing_index`` numbers the crossings of this plane by this path in the order the beam meets them: 0 the first, 1
    the second (the beam coming back through the plane after a fold, a double pass or a retro-reflection), and so on.
    """

    path_id: BeamPathId
    crossing_index: int = Field(ge=0, description="0 for the path's first crossing of this plane, 1 for its second, ...")
    chief_uv_mm: tuple[float, float] = Field(description="Chief-ray hit in the plane frame (u = local x, v = local y).")
    beam_radius_u_mm: float | None = Field(None, gt=0, description="Gaussian 1/e^2 radius along the beam's first transverse axis.")
    beam_radius_v_mm: float | None = Field(None, gt=0)
    pupil_rms_radius_mm: float | None = Field(None, ge=0)
    power_W: float | None = Field(None, ge=0, description="The beam's power crossing the plane (W); null where it is not known (an order of a diffraction grating without a stated response).")


class ObservationPlaneKindV1(StrEnum):
    OBSERVATION = "OBSERVATION"
    TARGET = "TARGET"  # at least one Project target evaluates on this plane


class ObservationAbsorptionV1(StrEnum):
    """What an observation plane does to the light that reaches it.

    ``NONE``: a transparent monitor, every beam passes on unchanged. ``ACTIVE_AREA``: a detector that absorbs the beams
    reaching its active area (``aperture``); beams outside it pass on. ``UNBOUNDED_PLANE``: a detector without an
    aperture, the whole plane absorbs.
    """

    NONE = "NONE"
    ACTIVE_AREA = "ACTIVE_AREA"
    UNBOUNDED_PLANE = "UNBOUNDED_PLANE"


class ObservationPlaneV1(StrictContractModel):
    """A Project observation plane in the design frame (a device-attached plane is already composed)."""

    observation_id: OpenSciId
    kind: ObservationPlaneKindV1
    pose: PoseV1 = Field(description="Plane frame: normal = local +Z, u = local +X, v = local +Y.")
    aperture: ApertureV1 | None = None
    display_half_extent_mm: float = Field(gt=0, description="Half size used to draw a plane without an aperture.")
    absorbing_extent: ObservationAbsorptionV1 = Field(description="Whether and where the plane absorbs the light that reaches it.")
    target_ids: tuple[OpenSciId, ...] = ()
    footprints: tuple[BeamFootprintV1, ...] = Field(
        (), description="One per Engine crossing of the plane, ordered by (path_id, crossing_index).")

    @model_validator(mode="after")
    def _check(self) -> ObservationPlaneV1:
        require_unit_quaternion(self.pose, f"observation {self.observation_id!r} pose")
        if (self.kind is ObservationPlaneKindV1.TARGET) != bool(self.target_ids):
            raise ValueError(f"observation {self.observation_id!r}: kind TARGET iff it has target_ids")
        if _has_duplicates(self.target_ids):
            raise ValueError(f"observation {self.observation_id!r}: duplicate target_ids")
        if self.absorbing_extent is ObservationAbsorptionV1.ACTIVE_AREA and self.aperture is None:
            raise ValueError(f"observation {self.observation_id!r}: an ACTIVE_AREA detector needs its aperture")
        if self.absorbing_extent is ObservationAbsorptionV1.UNBOUNDED_PLANE and self.aperture is not None:
            raise ValueError(f"observation {self.observation_id!r}: a plane with an aperture absorbs on its ACTIVE_AREA")
        keys = [(f.path_id, f.crossing_index) for f in self.footprints]
        if _has_duplicates(keys):
            raise ValueError(f"observation {self.observation_id!r}: duplicate footprint (path_id, crossing_index)")
        if keys != sorted(keys):
            raise ValueError(f"observation {self.observation_id!r}: footprints must be ordered by (path_id, crossing_index)")
        return self


# ---------------------------------------------------------------------------------------------- beam paths


class RayEndV1(StrEnum):
    PATH_END = "PATH_END"  # alive when the Engine path ended
    STOPPED_AT_SURFACE = "STOPPED_AT_SURFACE"  # removed at its last point (aperture, grazing, absorber, ...)
    NOT_FOLLOWED = "NOT_FOLLOWED"  # a marginal ray the Engine does not follow past its last point


class RayPolylineV1(StrictContractModel):
    """One Engine ray: ``ray_index`` 0 is the chief ray, ``>= 1`` a pupil sample (``ring_index`` = its pupil ring)."""

    ray_index: int = Field(ge=0)
    ring_index: int | None = Field(None, ge=0)
    points_mm: tuple[Vector3, ...] = Field(min_length=1, description="Engine points, global design frame.")
    end: RayEndV1
    display_extension_to_mm: Vector3 | None = Field(
        None, description="Display-only stub end for a free ray: last point + settings.free_ray_display_length_mm * exit direction."
    )

    @model_validator(mode="after")
    def _check(self) -> RayPolylineV1:
        if (self.ray_index == 0) != (self.ring_index is None):
            raise ValueError("ring_index is set exactly for pupil rays (ray_index >= 1)")
        for point in self.points_mm:
            _finite(point, "points_mm")
        if self.display_extension_to_mm is not None:
            _finite(self.display_extension_to_mm, "display_extension_to_mm")
        return self


#: At a split (beamsplitter, dichroic, partially reflecting surface) a branch that receives less than this share of the
#: power arriving there is SECONDARY: a leak through a coating, a ghost reflection, a weak pick-off.
SECONDARY_BRANCH_POWER_SHARE = 0.2
#: Spectral rejection: an element that passes a line at less than this share of what it passes at its best-passed
#: wavelength (``coating_peak``: a short-pass filter's stop band, a dichroic's rejected band) blocks that line; the light
#: beyond it is SECONDARY and no target makes it a main beam. A wavelength-neutral attenuation (a neutral density filter
#: of any optical density) passes every line at its own best share, so it never blocks.
BLOCKED_POWER_SHARE = 1e-3
#: Metrics of how much light arrives: a target on one of them asks for the light only when it requires a positive
#: amount (a ``minimum`` above zero, or a ``target_value`` above zero by more than its tolerance); otherwise it bounds a
#: leak.
LEAK_BOUND_METRICS = frozenset({"power_W", "coherent_power_W", "mean_irradiance_W_per_mm2", "peak_irradiance_W_per_mm2"})
#: Two wavelengths (nm) closer than this are the same spectral line when a target selects a line (the Engine default of
#: ``interference_wavelength_tol_nm``).
TARGET_LINE_TOLERANCE_NM = 1e-9


def coating_peak(coating: Any, *, reflected: bool) -> float | None:
    """The largest power reflectance (``reflected``) or transmittance a published coating (``PublicCoatingV1``) states at
    any wavelength: its band values, its tabulated spectrum, or for an angle-resolved table (``angular_spectrum``) the
    largest unpolarized value ``(s + p) / 2`` at any of its wavelength and angle nodes (bilinear interpolation never exceeds
    its nodes). None without stated values."""

    field = "reflectance" if reflected else "transmittance"
    table = getattr(coating, "angular_spectrum", None)
    if table is not None:
        s_rows, p_rows = getattr(table, f"{field}_s"), getattr(table, f"{field}_p")
        return max((float(s) + float(p)) / 2.0 for s_row, p_row in zip(s_rows, p_rows) for s, p in zip(s_row, p_row))
    if getattr(coating, "spectrum", None) is not None:
        values = [float(v) for v in getattr(coating.spectrum, field)]
    else:
        values = [float(getattr(band, field)) for band in getattr(coating, "bands", None) or ()]
    return max(values) if values else None


def target_asks_for_light(target: Any) -> bool:
    """Whether a Project target (``TargetV1``) asks for light on its plane: a metric that is no amount of light (a spot
    size, a position) always describes a beam that is there; an amount of light (``LEAK_BOUND_METRICS``) is asked for
    only when the target requires more than zero (``minimum > 0``, or ``target_value`` minus its tolerance above zero).
    ``minimum: 0, maximum: 1e-5`` or ``target_value: 0, tolerance_abs: 1e-5`` bound a leak."""

    if target.metric not in LEAK_BOUND_METRICS:
        return True
    if target.minimum is not None and target.minimum > 0:
        return True
    if target.target_value is None:
        return False
    tolerance = max(target.tolerance_abs or 0.0, (target.tolerance_rel or 0.0) * abs(target.target_value))
    return target.target_value - tolerance > 0


@dataclass(frozen=True)
class PlaneArrivalV1:
    """One beam that reaches an observation plane, as a display-role decision sees it: the caller's ``key`` for it, its
    source and spectral line, its power there (W, or any common scale on one plane), whether it arrives blocked
    (spectrally rejected on its way, ``BLOCKED_POWER_SHARE``) and its Engine path id when known."""

    key: Any
    source_id: str
    wavelength_nm: float
    power: float
    blocked: bool
    path_id: str | None = None


def designed_arrivals(targets: Iterable[Any], arrivals: Mapping[str, Sequence[PlaneArrivalV1]], *,
                      tolerance_nm: float = TARGET_LINE_TOLERANCE_NM) -> set[tuple[str, Any]]:
    """``{(observation id, arrival key)}``: the arrivals that a Project target on their plane asks for (the designed
    signal, drawn as a main beam up to that plane, a weak pick-off included). A blocked arrival never is.

    A target that asks for light (``target_asks_for_light``) selects the arrivals of its sources (``source_id`` /
    ``source_id_b``; every source when unset), of its line (``wavelength_nm``; every line when unset) and of its
    ``path_id`` parameter. A target naming both a source and a line asks for each selected arrival; a target that
    leaves the source or the line open sums what arrives (the Engine sums the power of every selected beam), so it asks
    for each selected arrival that carries at least ``SECONDARY_BRANCH_POWER_SHARE`` of the selected power on its plane,
    never for a minor leak beside it."""

    out: set[tuple[str, Any]] = set()
    for target in targets:
        plane = target.observation_id
        if plane is None or not target_asks_for_light(target):
            continue
        sources = {target.source_id, getattr(target, "source_id_b", None)} - {None}
        path = next((p.value for p in getattr(target, "parameters", ()) if p.key == "path_id"), None)
        selected = [a for a in arrivals.get(plane, ())
                    if (not sources or a.source_id in sources)
                    and (target.wavelength_nm is None or abs(float(a.wavelength_nm) - float(target.wavelength_nm)) <= tolerance_nm)
                    and not (isinstance(path, str) and a.path_id is not None and a.path_id != path)]
        if (sources and target.wavelength_nm is not None) or isinstance(path, str):
            out.update((plane, a.key) for a in selected if not a.blocked)
            continue
        total = sum(max(a.power, 0.0) for a in selected)
        out.update((plane, a.key) for a in selected
                   if not a.blocked and total > 0.0 and a.power >= SECONDARY_BRANCH_POWER_SHARE * total)
    return out


class BeamDisplayRoleV1(StrEnum):
    """How prominently a viewer draws a stretch of a beam path. Display only: it never changes what the Engine computed.

    ``PRIMARY``: a main beam. ``SECONDARY``: light downstream of a split where it received less than
    ``SECONDARY_BRANCH_POWER_SHARE`` of the arriving power (for example a dichroic leak or a ghost reflection), or
    downstream of an element that rejects its line (``BLOCKED_POWER_SHARE`` of the element's best share; a neutral
    attenuator never does), unless the path reaches an observation plane whose Project target asks for it
    (``designed_arrivals``); then its weak stretches up to that plane are the designed signal (a weak pick-off), never
    a blocked one. Viewers draw SECONDARY stretches thinner and fainter, never over a PRIMARY beam.
    """

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"


class BeamPathV1(StrictContractModel):
    """One Engine beam path (``OpticalPath.path_id``): one source, one wavelength, one branch."""

    path_id: BeamPathId
    source_element_id: InstanceKey
    wavelength_nm: float = Field(gt=0)
    spectral_index: int = Field(ge=0)
    parent_path_id: BeamPathId | None = None
    termination: OpenSciId
    display_rgb: Rgb = Field(description="Display colour only (false colour outside the visible band); not physics.")
    chief_ray: RayPolylineV1
    pupil_rays: tuple[RayPolylineV1, ...] = ()
    pupil_ray_count_traced: int = Field(ge=0, description="Pupil rays the Engine traced; pupil_rays may be a display subset.")
    final_power_W: float | None = Field(None, ge=0, description="The Engine's power of this path where it ends (W), quantized like the result hash; null where it is not known.")
    segment_roles: tuple[BeamDisplayRoleV1, ...] = Field(
        description="Display role of each chief-ray segment (points k -> k+1); the pupil rays follow the same steps.")
    display_role: BeamDisplayRoleV1 = Field(description="Display role where the path ends (its display extension and footprints).")
    chief_reference_from: int | None = Field(
        None, ge=0, description="The first chief-ray segment behind a central obstruction that the chief ray fell on: from there "
        "the chief ray is the beam's reference axis, not light (the light is the annulus the pupil rays trace); absent: the "
        "whole chief ray is lit.")

    @model_validator(mode="after")
    def _check(self) -> BeamPathV1:
        if self.chief_ray.ray_index != 0:
            raise ValueError(f"path {self.path_id!r}: chief_ray must have ray_index 0")
        if any(r.ray_index == 0 for r in self.pupil_rays) or _has_duplicates([r.ray_index for r in self.pupil_rays]):
            raise ValueError(f"path {self.path_id!r}: pupil ray indexes must be unique and >= 1")
        if len(self.pupil_rays) > self.pupil_ray_count_traced:
            raise ValueError(f"path {self.path_id!r}: more pupil rays than the Engine traced")
        if not all(0.0 <= c <= 1.0 for c in self.display_rgb):
            raise ValueError("display_rgb components must lie in [0, 1]")
        if len(self.segment_roles) != max(len(self.chief_ray.points_mm) - 1, 0):
            raise ValueError(f"path {self.path_id!r}: segment_roles must give one role per chief-ray segment")
        if self.chief_reference_from is not None and self.chief_reference_from >= len(self.segment_roles):
            raise ValueError(f"path {self.path_id!r}: chief_reference_from must name a chief-ray segment")
        return self


# ------------------------------------------------------------------------------------------ markers, camera


class AxisMarkerV1(StrictContractModel):
    """Optical-axis marker: a segment along an element's local +Z through its origin."""

    element_id: InstanceKey
    kind: Literal["OPTICAL_AXIS"] = "OPTICAL_AXIS"
    start_mm: Vector3
    end_mm: Vector3


class SceneBoundsV1(StrictContractModel):
    min_mm: Vector3
    max_mm: Vector3

    @model_validator(mode="after")
    def _order(self) -> SceneBoundsV1:
        _finite(self.min_mm + self.max_mm, "bounds")
        if any(lo > hi for lo, hi in zip(self.min_mm, self.max_mm, strict=True)):
            raise ValueError("bounds min_mm must not exceed max_mm")
        return self


class DisplayMediumV1(StrictContractModel):
    """A solid medium of the scene's elements, for drawing its glass: the refractive index of the run's own material at
    one wavelength. ``medium_id`` is the Engine material id (``catalog:<MATERIAL_ID>`` for a catalog product's media,
    ``project:<MATERIAL_ID>`` for a Project model's)."""

    medium_id: str = Field(pattern=r"^(catalog|project):[^\s:]+$")
    refractive_index: float = Field(gt=0)
    at_wavelength_nm: float = Field(gt=0, description="The d line (587.56 nm) where the material is valid there, else the middle of its band.")


class CameraHintV1(StrictContractModel):
    """Suggested initial view: look from ``eye_mm`` at ``target_mm`` with ``up`` (design frame)."""

    target_mm: Vector3
    eye_mm: Vector3
    up: Vector3 = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------------------------- provenance


class EngineRunRefV1(StrictContractModel):
    """The Engine run whose rays the scene shows. ``input_hash`` / ``result_hash`` use the same profiles as
    ``LocalValidationRecordV1``, so a scene and a validation record of the same run carry equal digests."""

    engine_package: EnginePackageName = ENGINE_PACKAGE_NAME
    engine_version: BuildVersion
    input_hash: HashRef
    result_hash: HashRef | None = None
    trace_hash: Sha256Digest = Field(description="The run manifest's trace_hash (eng-quant-py1 over the trace events).")
    ray_trace_version: BuildVersion

    @field_validator("input_hash")
    @classmethod
    def _input(cls, value: HashRef) -> HashRef:
        value = require_hash_profile(value, HashProfile.ENGINE_EXACT_PY1, field_name="EngineRunRefV1.input_hash")
        if value.subject != ENGINE_INPUT_HASH_SUBJECT:
            raise ValueError(f"EngineRunRefV1.input_hash subject must be {ENGINE_INPUT_HASH_SUBJECT!r}")
        return value

    @field_validator("result_hash")
    @classmethod
    def _result(cls, value: HashRef | None) -> HashRef | None:
        if value is None:
            return None
        value = require_engine_quantized_params(value, field_name="EngineRunRefV1.result_hash")
        if value.subject != ENGINE_RESULT_HASH_SUBJECT:
            raise ValueError(f"EngineRunRefV1.result_hash subject must be {ENGINE_RESULT_HASH_SUBJECT!r}")
        return value


class PupilRaySelectionV1(StrEnum):
    NONE = "NONE"
    OUTER_RING = "OUTER_RING"  # every pupil ray of the outermost traced ring
    ALL = "ALL"


class OpticalViewSettingsV1(StrictContractModel):
    """Display settings. They choose what to show, never how light propagates."""

    pupil_ray_selection: PupilRaySelectionV1 = PupilRaySelectionV1.OUTER_RING
    free_ray_display_length_mm: float = Field(50.0, gt=0, description="Stub length drawn past the last Engine point of an escaped ray.")
    axis_marker_half_length_mm: float = Field(20.0, gt=0)
    coordinate_decimals: int = Field(6, ge=3, le=9, description="Scene coordinates are rounded to this many decimals (mm).")


class ViewBuilderRefV1(StrictContractModel):
    builder_package: OpenSciId
    builder_version: BuildVersion
    settings: OpticalViewSettingsV1


class OpticalSimulationSceneV1(StrictContractModel):
    schema_id: Literal["opensci.optical_simulation_scene"] = Field(OPTICAL_SIMULATION_SCENE_SCHEMA_ID, alias="schema")
    schema_version: Literal["3.0"] = OPTICAL_SIMULATION_SCENE_SCHEMA_VERSION
    view_kind: Literal["OPTICAL_SIMULATION_VIEW"] = OPTICAL_SIMULATION_VIEW_KIND
    view_notice: Literal[
        "Optical Simulation View of the design: optical elements, sources, Engine-traced beam paths and observation "
        "planes. Element bodies are drawn with their design geometry."
    ] = OPTICAL_SIMULATION_VIEW_NOTICE
    frame: Literal["PROJECT_DESIGN_FRAME"] = "PROJECT_DESIGN_FRAME"
    length_unit: Literal["mm"] = "mm"
    up_axis: Literal["+Z"] = "+Z"
    project: ValidatedProjectRefV1
    catalog_version: CatalogVersion | None = Field(None, description="Public optical catalog release that supplied element geometry.")
    engine_run: EngineRunRefV1
    builder: ViewBuilderRefV1
    elements: tuple[SceneElementV1, ...]
    observation_planes: tuple[ObservationPlaneV1, ...] = ()
    beam_paths: tuple[BeamPathV1, ...] = ()
    axis_markers: tuple[AxisMarkerV1, ...] = ()
    display_media: tuple[DisplayMediumV1, ...] = Field((), description="The Engine materials of the run, ordered by id: what a viewer draws glass with.")
    bounds: SceneBoundsV1
    camera_hint: CameraHintV1

    @model_validator(mode="after")
    def _check(self) -> OpticalSimulationSceneV1:
        element_ids = [e.element_id for e in self.elements]
        for label, ids in (
            ("element_id", element_ids),
            ("observation_id", [o.observation_id for o in self.observation_planes]),
            ("path_id", [p.path_id for p in self.beam_paths]),
        ):
            if _has_duplicates(ids):
                raise ValueError(f"duplicate {label} in OpticalSimulationSceneV1")
        roles = {e.element_id: e.optical_role for e in self.elements}
        path_ids = {p.path_id for p in self.beam_paths}
        for path in self.beam_paths:
            if roles.get(path.source_element_id) is not OpticalRoleV1.SOURCE:
                raise ValueError(f"path {path.path_id!r}: source_element_id {path.source_element_id!r} is not a source element")
            if path.parent_path_id is not None and path.parent_path_id not in path_ids:
                raise ValueError(f"path {path.path_id!r}: unknown parent_path_id {path.parent_path_id!r}")
        for plane in self.observation_planes:
            unknown = [f.path_id for f in plane.footprints if f.path_id not in path_ids]
            if unknown:
                raise ValueError(f"observation {plane.observation_id!r}: footprint for unknown path(s) {unknown}")
        for marker in self.axis_markers:
            if marker.element_id not in roles:
                raise ValueError(f"axis marker for unknown element {marker.element_id!r}")
        media = [m.medium_id for m in self.display_media]
        if media != sorted(set(media)):
            raise ValueError("display_media must be ordered by medium_id, without duplicates")
        if any(e.geometry.basis is GeometryBasisV1.PUBLIC_CATALOG_DESIGN_GEOMETRY for e in self.elements) and self.catalog_version is None:
            raise ValueError("public catalog geometry requires catalog_version")
        return self

    def payload(self) -> dict[str, Any]:
        """Canonical hashed/persisted form (``None`` fields omitted)."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def content_hash(self) -> str:
        return osc1_digest(self.payload())
