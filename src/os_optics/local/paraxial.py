"""Paraxial analysis of placed elements of a Project: thick-lens and mirror ABCD, focal distances, principal planes, afocal
and collimation spacing.

``paraxial_analysis(project, request)`` takes components of a Project that lie on one optical axis (straight, or folded by
mirrors), in the order light meets them, and evaluates each element's paraxial matrix at one vacuum wavelength with the Engine's own
compilation and refractive-index evaluation (``compile.paraxial_matrix``: real surfaces, real thicknesses, the material
dispersion of the element). ``request``:

* ``components``: instance ids (one or more), in the order light meets them. A component the light returns to after a mirror is listed
  again where it meets it (a lens before a mirror: ``[L1, M1, L1]``, a double pass); a mirror is listed once
  (``PARAXIAL_REFLECTOR_REPEATED``: light that returns to a mirror circulates, which ``os_optics.local.cavity`` analyses);
* ``wavelength_nm``: the vacuum wavelength;
* ``axis`` (optional): ``"X"`` / ``"Y"``: the section of the system to analyse, needed when an element has cylindrical
  surfaces. The section axes are one frame for the whole system (``section_frame``): the source's local x / y when
  ``source_id`` is given (the axes of its waist radii), else the first component's local X / Y. Each element contributes
  the power its own surfaces have in that section: a cylinder whose curved axis lies along the section has its power
  there, one turned 90 deg about the optical axis has none there and its power in the other section. A cylinder at any
  other angle to the section axes couples the two sections and is refused (``PARAXIAL_CYLINDER_ROTATED``);
* ``solve`` (optional): ``"afocal"`` (exactly two components): the vertex gap that makes the pair afocal in the analysed
  section, ``C = 0`` of ``M2 T(d) M1``, ``d = -(C2 A1 + D2 C1) / (C1 C2)``, and where to put the second component for it
  (with ``axis``, the reply states whether the other section is afocal at that gap too); ``"collimate"``: the object
  distance of the analysed section's front focal point, from the object (a point source or a Gaussian waist) to the first
  entrance vertex (the meaning below);
* ``source_id`` (optional): a source of the Project, emitting along the optical axis from a point on it; with
  ``solve: "collimate"`` its waist is the object, and the reply adds the output Gaussian beam of each source axis.

The light's direction (``light_direction`` of the reply). The light path decides it, whether or not ``source_id`` is given: the
emission direction of the source ``source_id`` names, else of the Project's source whose emission line passes through the first
element (its reflecting vertex, or its origin) with the element ahead of it, checked as if named (rule ``PROJECT_SOURCE``). Sources
the request did not name that light the first element from different directions, or one that lies beyond its entrance vertex, do
not decide the direction: they are set aside, ``light_direction.notes`` names them (``{"code": "PARAXIAL_LIGHT_DIRECTION_AMBIGUOUS" |
"PARAXIAL_SOURCE_NOT_BEFORE_COMPONENTS", "source_ids"}``), and the direction is assumed; a source that lights the listed elements in
the opposite order is a list that contradicts the Project's light path and stays refused, as every check of a named source does.
With no source deciding it the direction is assumed (rule ``ASSUMED``) as stated below, and a list in reverse order is the same elements met by light travelling the other way.

The Project needs a light source only when the request names ``source_id``: element spacing can be analysed before a
source is placed.

The optical axis runs from the first component to the last; with one component, along the light's direction, assumed
along the element's local +Z. Every element's local Z axis must lie on that axis (turned elements
-- local Z against the light -- are used turned around), each component origin (and the source) on the axis, the
components in increasing order along it. Light travels between the elements in the air of the Project's environment
(``ProjectDocumentV2.effective_environment``: air at 20 C and 101325 Pa unless stated; vacuum at zero pressure): every index
is taken relative to that air at the same vacuum wavelength (``compile.paraxial_matrix``), so the air has index 1 in the
matrices and every distance is a distance in it -- the trace's refraction, with the same absolute indices.

Mirrors. A reflector (``SURFACE_SEQUENCE`` whose reflecting surface is the last one, on the +Z side) reflects the light
arriving on the side its local +Z faces, ``PARAXIAL_REFLECTOR_BACK_FACE`` otherwise. Curved: ``((1, 0), (-2 / R, 1))`` at its
vertex (``R > 0`` is concave, focal length ``R / 2``), its local Z along the axis (``PARAXIAL_CURVED_REFLECTOR_TILTED``
otherwise). Plane: the identity, at any tilt, folding the axis by the law of reflection. A reflector the light reaches
through other surfaces of its element is refused (``PARAXIAL_REFLECTOR_BEHIND_SURFACES``), and so is a ``SURFACE_SET``.
With a reflector in the request the light starts along the light's direction; assumed, it arrives along the first element's
axis for a curved first mirror (normal incidence), as the mirror image of the direction to the next element for a plane
first mirror, or from the first element to the next. The axis is then followed element by
element: each element must lie on the current stretch, a reflection turns the stretch and carries the section axes with it
(``v - 2 (v . n) n``). Every number is on the unfolded axis: ``*_along_axis_mm`` keep increasing past a reflection, and the
design-frame points lie on the real folded path (``axis_path`` lists its straight stretches); a rear focal point lies on the
line of the outgoing direction through the exit vertex, a negative ``bfl_mm`` behind the last mirror.

The double pass. Light a mirror sends back through a transmissive element of the request meets it again, turned around. The list
says so (``components`` above), every element then counting once per crossing; a straight stretch of the axis that crosses the element's plane
inside its outline while the list does not name it on that stretch is refused (``PARAXIAL_ELEMENT_CROSSED_AGAIN``), because the numbers would describe a
system the light does not see. Other components of the Project are not looked at: an analysis of a chosen list of elements leaves the others out.
A unit with a mirror names each member once, so a member its own light crosses again is refused the same way: analyse its members as separate
components.

Compound units. A ``COMPOUND`` unit is composed from its members' poses, every member checked as above and named with its
pose when it leaves the axis (``PARAXIAL_COMPOUND_MEMBER_NOT_COAXIAL``). Members meet the light in pose order along the axis;
a unit containing a mirror takes them in the order it declares them, because light does not meet mirrors in axial order.

Crystals. An element holding a uniaxial crystal (a Project-declared ``birefringence``) is refused
(``PARAXIAL_BIREFRINGENT_ELEMENT``): its ordinary and extraordinary waves have different indices (and the extraordinary wave
diffracts differently along and across the optic axis), so it has no single paraxial matrix. The cavity analysis
(``os_optics.local.cavity``) analyses a cavity holding one for a stated eigen-polarization.

The reply states every number's reference: ``abcd`` is the reduced matrix ``((A, B), (C, D))`` acting on ``(y, n u)``
from the stated input plane to the stated output plane; ``efl_mm = -1 / C``; ``bfl_mm = -A / C`` from the last exit vertex
to the rear focal point (positive: after it); ``ffd_mm = -D / C`` from the front focal point to the first entrance vertex
(positive: the focal point lies before the vertex); principal planes ``(1 - A) / C`` from the exit vertex and ``(D - 1) / C``
from the entrance vertex (positive: downstream). Points are design-frame millimetres. With ``axis``, ``sections`` gives
the system of both sections. ``optical_axis`` is the light's direction at the input plane; ``axis_path`` lists the axis's
straight stretches (direction, ends, unfolded positions, section axes). Each element states ``reflector``, ``turned`` (a mirror:
false), and a compound unit its ``members`` in the order light meets them.

``solve: "collimate"``: a point source at the front focal point leaves the elements collimated (geometric optics). A
Gaussian waist there leaves with the largest output waist and the smallest far-field divergence the elements can give it,
``w_out = M^2 lambda |f| / (pi w_in)``, its waist at the rear focal point; its output Rayleigh range is ``f^2 / z_R,in``.
That beam is collimated (``state: "COLLIMATED"``) when the input Rayleigh range ``z_R,in`` is at most ``|EFL|`` (the output
Rayleigh range is then at least ``|EFL|``); a source whose Rayleigh range is longer is already collimated over the
elements' focal length, and the elements focus it (``state: "FOCUSED"``, note ``INPUT_ALREADY_COLLIMATED``). A source axis
whose section has its front focal point elsewhere is ``"UNCOLLIMATED"`` (note ``OTHER_SECTION_NOT_COLLIMATED``).

This is paraxial ray optics of the elements' prescriptions; it predicts where to put elements. A run of the Project
(``run_optical_project``) gives the traced beams with the Gaussian model, aberrations and apertures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from .._vendor.shared_contracts.optical_design import OpticalRoleV1
from ..contracts.materials import UniaxialMaterial
from ..math.exact_trig import asin
from .catalog import load_project
from .compile import IDENTITY, PARAXIAL_SURFACE_NOT_SUPPORTED, CompiledElement, Matrix, ParaxialSurfaceNotSupported, _multiply, paraxial_matrix, reflecting_surface, surrounding_index
from .lowering import lower_project

class ParaxialErrorCode(StrEnum):
    """The codes of ``ParaxialError`` problems and of the refused analysis's envelope (stable values; a problem holds the
    value as a plain string)."""

    PARAXIAL_REQUEST_INVALID = "PARAXIAL_REQUEST_INVALID"
    PARAXIAL_COMPONENT_UNKNOWN = "PARAXIAL_COMPONENT_UNKNOWN"
    PARAXIAL_ELEMENT_UNSUPPORTED = "PARAXIAL_ELEMENT_UNSUPPORTED"
    PARAXIAL_NOT_COAXIAL = "PARAXIAL_NOT_COAXIAL"
    PARAXIAL_ORDER_INVALID = "PARAXIAL_ORDER_INVALID"
    PARAXIAL_NO_SOLUTION = "PARAXIAL_NO_SOLUTION"
    #: A cylindrical element whose cylinder axes are neither parallel nor perpendicular to the section axes.
    PARAXIAL_CYLINDER_ROTATED = "PARAXIAL_CYLINDER_ROTATED"
    #: A member of a COMPOUND unit that leaves the optical axis (the message names the member and its pose).
    PARAXIAL_COMPOUND_MEMBER_NOT_COAXIAL = "PARAXIAL_COMPOUND_MEMBER_NOT_COAXIAL"
    #: The light reaches a reflector on the side its local +Z does not face.
    PARAXIAL_REFLECTOR_BACK_FACE = "PARAXIAL_REFLECTOR_BACK_FACE"
    #: A curved reflector whose local Z axis does not lie on the optical axis.
    PARAXIAL_CURVED_REFLECTOR_TILTED = "PARAXIAL_CURVED_REFLECTOR_TILTED"
    #: A reflector the light reaches through other surfaces of its element (a second-surface mirror).
    PARAXIAL_REFLECTOR_BEHIND_SURFACES = "PARAXIAL_REFLECTOR_BEHIND_SURFACES"
    #: A straight stretch of the axis crosses a transmissive element the list does not name on that stretch (a lens before a mirror, listed once).
    PARAXIAL_ELEMENT_CROSSED_AGAIN = "PARAXIAL_ELEMENT_CROSSED_AGAIN"
    #: A mirror listed twice (light that returns to a mirror circulates: a cavity).
    PARAXIAL_REFLECTOR_REPEATED = "PARAXIAL_REFLECTOR_REPEATED"
    #: Sources of the Project on the line through the first element emit in different directions (name one with ``source_id``).
    PARAXIAL_LIGHT_DIRECTION_AMBIGUOUS = "PARAXIAL_LIGHT_DIRECTION_AMBIGUOUS"
    #: An element holding a uniaxial crystal: its matrix depends on the light's polarization (module docstring).
    PARAXIAL_BIREFRINGENT_ELEMENT = "PARAXIAL_BIREFRINGENT_ELEMENT"
    #: A surface of an element that has no paraxial matrix (``compile.ParaxialSurfaceNotSupported``).
    PARAXIAL_SURFACE_NOT_SUPPORTED = PARAXIAL_SURFACE_NOT_SUPPORTED
    #: The envelope code of a refused analysis (``json_api.paraxial_analysis``).
    PARAXIAL_ANALYSIS_INVALID = "PARAXIAL_ANALYSIS_INVALID"


#: Error codes of ``ParaxialError`` problems (the values of ``ParaxialErrorCode``, as plain strings).
REQUEST_INVALID = ParaxialErrorCode.PARAXIAL_REQUEST_INVALID.value
COMPONENT_UNKNOWN = ParaxialErrorCode.PARAXIAL_COMPONENT_UNKNOWN.value
ELEMENT_UNSUPPORTED = ParaxialErrorCode.PARAXIAL_ELEMENT_UNSUPPORTED.value
NOT_COAXIAL = ParaxialErrorCode.PARAXIAL_NOT_COAXIAL.value
ORDER_INVALID = ParaxialErrorCode.PARAXIAL_ORDER_INVALID.value
NO_SOLUTION = ParaxialErrorCode.PARAXIAL_NO_SOLUTION.value
CYLINDER_ROTATED = ParaxialErrorCode.PARAXIAL_CYLINDER_ROTATED.value
COMPOUND_MEMBER_NOT_COAXIAL = ParaxialErrorCode.PARAXIAL_COMPOUND_MEMBER_NOT_COAXIAL.value
REFLECTOR_BACK_FACE = ParaxialErrorCode.PARAXIAL_REFLECTOR_BACK_FACE.value
CURVED_REFLECTOR_TILTED = ParaxialErrorCode.PARAXIAL_CURVED_REFLECTOR_TILTED.value
REFLECTOR_BEHIND_SURFACES = ParaxialErrorCode.PARAXIAL_REFLECTOR_BEHIND_SURFACES.value
ELEMENT_CROSSED_AGAIN = ParaxialErrorCode.PARAXIAL_ELEMENT_CROSSED_AGAIN.value
REFLECTOR_REPEATED = ParaxialErrorCode.PARAXIAL_REFLECTOR_REPEATED.value
LIGHT_DIRECTION_AMBIGUOUS = ParaxialErrorCode.PARAXIAL_LIGHT_DIRECTION_AMBIGUOUS.value
#: Note code (``light_direction.notes``): a Project source the request did not name lies beyond the first entrance vertex.
SOURCE_NOT_BEFORE_COMPONENTS = "PARAXIAL_SOURCE_NOT_BEFORE_COMPONENTS"
#: Where the light's direction at the first element comes from (``light_direction.rule``).
SOURCE_LIGHT, PROJECT_SOURCE_LIGHT, ASSUMED_LIGHT = "SOURCE", "PROJECT_SOURCE", "ASSUMED"
BIREFRINGENT_ELEMENT = ParaxialErrorCode.PARAXIAL_BIREFRINGENT_ELEMENT.value
PARAXIAL_ANALYSIS_INVALID = ParaxialErrorCode.PARAXIAL_ANALYSIS_INVALID.value
SOLVES = ("afocal", "collimate")
SECTIONS = ("X", "Y")
REQUEST_KEYS = frozenset({"components", "wavelength_nm", "axis", "solve", "source_id", "configuration_id", "coaxial_tolerance"})
COAXIAL_TOLERANCE_KEYS = frozenset({"lateral_mm", "tilt_rad"})
#: An element axis counts as parallel to the optical axis when |cos| is within this of 1 (1e-9 rad^2 / 2 of tilt).
AXIS_COSINE_TOLERANCE = 1e-12
#: A cylinder axis counts as perpendicular to a section axis when |cos| is at most this (the angle of the tilt
#: tolerance above, about 1.4e-6 rad).
PERPENDICULAR_COSINE_TOLERANCE = math.sqrt(2.0 * AXIS_COSINE_TOLERANCE)
#: A component origin counts as on the optical axis within this lateral distance (mm): the default of the request's
#: ``coaxial_tolerance.lateral_mm``.
LATERAL_TOLERANCE_MM = 1e-6
#: The tilt the default cosine test accepts, as an angle (rad): ``1 - cos(t) <= AXIS_COSINE_TOLERANCE``, ``t = sqrt(2 x 1e-12)``. The
#: reply states it as the tolerance in force where the request states no ``coaxial_tolerance.tilt_rad``.
DEFAULT_TILT_RAD = math.sqrt(2.0 * AXIS_COSINE_TOLERANCE)
#: Shortest distance (mm) between two component origins that defines the optical axis direction.
MIN_AXIS_LENGTH_MM = 1e-9
#: |C| below this (1/mm) is an afocal system: its focal quantities are infinite and reported as null.
AFOCAL_C_PER_MM = 1e-15
#: Two sections share their front focal point when their front focal distances agree within this (relative to
#: max(1 mm, |distance|)).
SAME_FOCUS_REL = 1e-9
NM_TO_MM = 1e-6
#: Output-beam states of ``solve: "collimate"`` (module docstring).
COLLIMATED, FOCUSED, UNCOLLIMATED = "COLLIMATED", "FOCUSED", "UNCOLLIMATED"
CONVENTIONS = {
    "matrix": "reduced paraxial ABCD ((A, B), (C, D)) on (y, n u): y ray height (mm), u ray angle (rad), n refractive index "
              "relative to the air of the Project's environment (absolute index / n_air at the same vacuum wavelength): that air "
              "(n = 1) lies between the elements",
    "efl_mm": "-1 / C",
    "bfl_mm": "-A / C: from the output plane to the rear focal point, positive after it (along the outgoing direction)",
    "ffd_mm": "-D / C: from the front focal point to the input plane, positive when the focal point lies before it",
    "principal_planes": "rear_principal_plane_mm = (1 - A) / C from the output plane, front_principal_plane_mm = (D - 1) / C from "
                        "the input plane; both positive downstream",
    "positions": "design-frame millimetres; along_axis_mm = distance along the axis from the first component's origin (a compound "
                 "unit with mirrors: its first member's); optical_axis is the light's direction at the input plane",
    "unfolded_axis": "a reflection is unfolded: along_axis_mm, vertex gaps, ABCD and the focal distances are taken along the light's "
                     "path, past a mirror they keep increasing; design-frame points lie on the real folded path, whose straight "
                     "stretches axis_path lists",
    "mirrors": "a mirror of vertex radius R is ((1, 0), (-2 / R, 1)) at its vertex (R > 0: centre of curvature on the side the light "
               "comes from, concave, focal length R / 2), a plane mirror the identity; a reflection carries the section axes with "
               "it, v - 2 (v . n) n",
    "sections": "section X / Y: the source's local x / y (source_id given), else the first component's local X / Y (section_frame, "
                "design-frame unit vectors, at each stretch of axis_path after reflections); each element contributes the power its "
                "surfaces have along that section",
    "light_direction": "the light's direction at the first element (notes: Project sources set aside because they disagree or lie beyond "
                       "the first entrance vertex): the emission direction of the source named by source_id (rule SOURCE), "
                       "else of the Project's source whose emission line passes through the first element (PROJECT_SOURCE), else assumed "
                       "(ASSUMED): from the first element to the last, a curved first mirror met on its reflecting side along its axis, a "
                       "plane first mirror from the direction it reflects towards the next element",
    "model": "paraxial ray optics of the elements' prescriptions at the stated vacuum wavelength (the Engine's refractive-index "
             "evaluation); Gaussian output beams by the complex q through the same matrix, with the beam's own Rayleigh range "
             "pi w0^2 / (M^2 lambda)",
}


class ParaxialError(ValueError):
    """A request the analysis cannot answer: ``problems`` (each ``{"code", "message", "location"}``)."""

    def __init__(self, problems: Sequence[Mapping[str, Any]]) -> None:
        super().__init__("; ".join(p["message"] for p in problems))
        self.problems = tuple(dict(p) for p in problems)

    def as_dict(self) -> dict[str, Any]:
        return {"code": PARAXIAL_ANALYSIS_INVALID, "problems": list(self.problems)}


def _refuse(code: str, message: str, location: str | None = None) -> ParaxialError:
    return ParaxialError([{"code": code, "message": message, "location": location}])


Vector = tuple[float, float, float]


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _add(a: Vector, b: Vector, scale: float = 1.0) -> Vector:
    return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])


def _unit(v: Vector) -> Vector:
    length = math.sqrt(_dot(v, v))
    return (v[0] / length, v[1] / length, v[2] / length)


def _local_axes(quaternion: Sequence[float]) -> tuple[Vector, Vector, Vector]:
    """The component's local +X, +Y and +Z axes in the design frame (the rotation of a unit quaternion ``(w, x, y, z)``)."""

    w, x, y, z = (float(c) for c in quaternion)
    norm = w * w + x * x + y * y + z * z
    ex = (1.0 - 2.0 * (y * y + z * z) / norm, 2.0 * (x * y + w * z) / norm, 2.0 * (x * z - w * y) / norm)
    ey = (2.0 * (x * y - w * z) / norm, 1.0 - 2.0 * (x * x + z * z) / norm, 2.0 * (y * z + w * x) / norm)
    ez = (2.0 * (x * z + w * y) / norm, 2.0 * (y * z - w * x) / norm, 1.0 - 2.0 * (x * x + y * y) / norm)
    return _unit(ex), _unit(ey), _unit(ez)


def _local_z(quaternion: Sequence[float]) -> Vector:
    """The component's local +Z axis in the design frame."""

    return _local_axes(quaternion)[2]


def _in_frame(axes: tuple[Vector, Vector, Vector], v: Sequence[float]) -> Vector:
    """The design-frame vector of ``v`` given in the frame ``axes``."""

    return _add(_add(_add((0.0, 0.0, 0.0), axes[0], float(v[0])), axes[1], float(v[1])), axes[2], float(v[2]))


def _mirrored(v: Vector, normal: Vector) -> Vector:
    """``v`` reflected in a plane of unit ``normal``."""

    return _unit(_add(v, normal, -2.0 * _dot(v, normal)))


def _across(v: Vector, axis: Vector) -> Vector:
    """The part of ``v`` perpendicular to the unit ``axis``."""

    return _add(v, axis, -_dot(v, axis))


def _lateral(point: Vector, origin: Vector, axis: Vector) -> tuple[float, float]:
    """``(along, lateral distance)`` of ``point`` from the line through ``origin`` along the unit ``axis``."""

    offset = _add(point, origin, -1.0)
    along = _dot(offset, axis)
    lateral = _add(offset, axis, -along)
    return along, math.sqrt(_dot(lateral, lateral))


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _refuse(REQUEST_INVALID, f"{name} is a number, not {value!r}", name)
    try:
        number = float(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        raise _refuse(REQUEST_INVALID, f"{name} is a number, not {value!r}", name) from None
    if not math.isfinite(number) or number <= 0.0:
        raise _refuse(REQUEST_INVALID, f"{name} must be a positive number, not {value!r}", name)
    return number


def _focal(matrix: Matrix) -> dict[str, float | None]:
    (a, _b), (c, d) = matrix
    if abs(c) < AFOCAL_C_PER_MM:
        return {"efl_mm": None, "bfl_mm": None, "ffd_mm": None, "front_principal_plane_mm": None, "rear_principal_plane_mm": None}
    return {"efl_mm": -1.0 / c, "bfl_mm": -a / c, "ffd_mm": -d / c, "front_principal_plane_mm": (d - 1.0) / c,
            "rear_principal_plane_mm": (1.0 - a) / c}


def _matrix_json(matrix: Matrix) -> list[list[float]]:
    return [[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]]


@dataclass(frozen=True)
class CoaxialTolerance:
    """How far an element may lie off the optical axis (``lateral_mm``) and how far its axis may be tilted to it (``tilt_rad``) and
    still be analysed on the nominal axis (the request's ``coaxial_tolerance``). ``tilt_rad`` ``None`` is the Engine's default,
    compared on the direction cosine as ``1 - cos <= AXIS_COSINE_TOLERANCE`` (``DEFAULT_TILT_RAD`` as an angle); a stated tilt is
    compared as an angle. The analysis stays the nominal-axis one: positions are projections onto the axis, matrices the elements'
    own. A tilt changes an element's matrix at second order in the angle; a decentre leaves it and deviates the axis by the decentre
    times the element's power, which is not reported."""

    lateral_mm: float = LATERAL_TOLERANCE_MM
    tilt_rad: float | None = None

    @property
    def stated(self) -> bool:
        return self.lateral_mm != LATERAL_TOLERANCE_MM or self.tilt_rad is not None

    def off(self, distance_mm: float) -> bool:
        return distance_mm > self.lateral_mm

    def tilted(self, a: Vector, b: Vector) -> bool:
        """Whether the unit vectors ``a`` and ``b``, which should point the same way, are further apart than the tolerance."""

        cosine = _dot(a, b)
        if self.tilt_rad is None:
            return 1.0 - cosine > AXIS_COSINE_TOLERANCE
        return cosine <= 0.0 or _angle(a, b) > self.tilt_rad

    def differ(self, a: Vector, b: Vector) -> bool:
        """``tilted`` for two directions compared with each other (the default as ``cos < 1 - AXIS_COSINE_TOLERANCE``)."""

        cosine = _dot(a, b)
        if self.tilt_rad is None:
            return cosine < 1.0 - AXIS_COSINE_TOLERANCE
        return cosine <= 0.0 or _angle(a, b) > self.tilt_rad

    def as_dict(self) -> dict[str, float]:
        return {"lateral_mm": self.lateral_mm, "tilt_rad": DEFAULT_TILT_RAD if self.tilt_rad is None else self.tilt_rad}

    def lateral_text(self) -> str:
        return f"coaxial_tolerance.lateral_mm is {self.lateral_mm:g} mm"

    def tilt_text(self) -> str:
        return f"coaxial_tolerance.tilt_rad is {self.as_dict()['tilt_rad']:.6g} rad"


def _angle(a: Vector, b: Vector) -> float:
    """The angle (rad, 0 to pi / 2) between the lines of the unit vectors ``a`` and ``b``, from their cross product (the same
    arithmetic in every runtime: ``math.exact_trig.asin``)."""

    cross = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    return float(asin(min(1.0, math.sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]))))


def coaxial_tolerance(stated: Any) -> CoaxialTolerance:
    """The request's ``coaxial_tolerance`` (``{"lateral_mm"?, "tilt_rad"?}``; absent: the Engine's)."""

    if stated is None:
        return CoaxialTolerance()
    if not isinstance(stated, Mapping) or set(stated) - COAXIAL_TOLERANCE_KEYS:
        raise _refuse(REQUEST_INVALID, f"coaxial_tolerance is an object; it takes {sorted(COAXIAL_TOLERANCE_KEYS)}", "coaxial_tolerance")
    values: dict[str, float] = {}
    for key in COAXIAL_TOLERANCE_KEYS:
        value = stated.get(key)
        if value is None:
            continue
        name = f"coaxial_tolerance.{key}"
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise _refuse(REQUEST_INVALID, f"{name} is a number, not {value!r}", name)
        try:
            number = float(Decimal(str(value)))
        except (InvalidOperation, ValueError):
            raise _refuse(REQUEST_INVALID, f"{name} is a number, not {value!r}", name) from None
        if not math.isfinite(number) or number < 0.0 or (key == "tilt_rad" and number >= math.pi / 2.0):
            raise _refuse(REQUEST_INVALID, f"{name} is not negative" + (" and below pi / 2" if key == "tilt_rad" else "") + f", not {value!r}", name)
        values[key] = number
    return CoaxialTolerance(values.get("lateral_mm", LATERAL_TOLERANCE_MM), values.get("tilt_rad"))


@dataclass(frozen=True)
class _Stretch:
    """One straight stretch of the optical axis: from ``start_mm`` (at ``start_along_mm`` on the unfolded axis) along
    ``direction``, with the section axes there."""

    start_along_mm: float
    start_mm: Vector
    direction: Vector
    section_x: Vector
    section_y: Vector


class _AxisWalk:
    """The optical axis as far as the analysis has followed it: its stretches, the last one current."""

    def __init__(self, point: Vector, direction: Vector, section_x: Vector, section_y: Vector) -> None:
        self.stretches = [_Stretch(0.0, point, direction, section_x, section_y)]
        self.air: Any = None  # the Engine material of the environment's air (``LoweredProject.air``)

    @property
    def current(self) -> _Stretch:
        return self.stretches[-1]

    def locate(self, point: Vector) -> tuple[float, float]:
        """``(unfolded position, lateral distance)`` of ``point`` from the current stretch's line."""

        along, off = _lateral(point, self.current.start_mm, self.current.direction)
        return self.current.start_along_mm + along, off

    def point(self, along_mm: float) -> Vector:
        return _add(self.current.start_mm, self.current.direction, along_mm - self.current.start_along_mm)

    def reflect(self, along_mm: float, normal: Vector) -> None:
        here = self.current
        self.stretches.append(_Stretch(along_mm, self.point(along_mm), _mirrored(here.direction, normal),
                                       _mirrored(here.section_x, normal), _mirrored(here.section_y, normal)))


@dataclass(frozen=True)
class _Placed:
    instance_id: str
    element: CompiledElement
    origin: Vector
    turned: bool
    matrices: dict[str, Matrix]  # by system section "X" / "Y" (equal for a rotationally symmetric element)
    element_sections: dict[str, str] | None  # system section -> the element's local section (cylindrical elements only)
    entrance_mm: float  # along the unfolded axis, from the first component's origin
    exit_mm: float
    entrance_point: Vector
    exit_point: Vector
    direction_in: Vector  # of the light arriving at the element
    direction_out: Vector  # of the light leaving it
    stretch: int = 0  # the stretch of the axis it was met on
    reflector: bool = False  # the element reflects the light (a compound unit: a member does)
    members: tuple[_Placed, ...] = ()  # a compound unit's members, in the order light meets them
    decentre_mm: float | None = None  # measured: its origin (a mirror: its reflecting vertex) off the axis; a compound unit: its members'
    tilt_rad: float | None = None  # measured: its local Z against the axis; None for a plane mirror, which folds the axis at any tilt


def _vertex_span(element: CompiledElement) -> tuple[float, float]:
    """Local z of the first and the last vertex (0, 0 for a thin element)."""

    surfaces = getattr(element.model, "surfaces", None)
    if not surfaces or not hasattr(surfaces[0], "vertex_z_mm"):
        return 0.0, 0.0
    return float(Decimal(surfaces[0].vertex_z_mm)), float(Decimal(surfaces[-1].vertex_z_mm))


def _cylindrical(element: CompiledElement) -> bool:
    surfaces = getattr(element.model, "surfaces", None) or ()
    return any(getattr(s, "is_cylindrical", False) or getattr(s, "has_section_powers", False) for s in surfaces)


def _turn(matrix: Matrix) -> Matrix:
    """The element used turned around (light along its local -Z): ``((D, B), (C, A))``."""

    (a, b), (c, d) = matrix
    return ((d, b), (c, a))


def _gaussian_out(matrix: Matrix, distance_mm: float, waist_mm: float, m2: float, wavelength_mm: float) -> dict[str, float]:
    """The output beam of a waist ``distance_mm`` before the input plane, through ``matrix``: the beam's own q (its Rayleigh
    range ``pi w0^2 / (M^2 lambda)``, so the radii are the beam's second-moment radii)."""

    rayleigh = math.pi * waist_mm * waist_mm / (m2 * wavelength_mm)
    q_in = complex(distance_mm, rayleigh)
    (a, b), (c, d) = matrix
    q_out = (a * q_in + b) / (c * q_in + d)
    waist_out = math.sqrt(m2 * wavelength_mm * q_out.imag / math.pi)
    return {"input_rayleigh_range_mm": rayleigh, "waist_radius_mm": waist_out, "waist_after_output_plane_mm": -q_out.real,
            "rayleigh_range_mm": q_out.imag, "far_field_half_angle_rad": waist_out / q_out.imag}


def _system(placed: Sequence[_Placed], gaps: list[float], section: str) -> Matrix:
    system = IDENTITY
    for k, item in enumerate(placed):
        if k:
            system = _multiply(((1.0, gaps[k - 1]), (0.0, 1.0)), system)
        system = _multiply(item.matrices[section], system)
    return system


def _where(instance_id: str, element: CompiledElement) -> str:
    return f"component {instance_id}" if element.member_id is None else f"component {instance_id} member {element.member_id}"


def _who(instance_id: str, element: CompiledElement) -> str:
    return repr(instance_id) if element.member_id is None else f"member {element.member_id!r} of {instance_id!r}"


def _off_axis(instance_id: str, element: CompiledElement, problem: str) -> ParaxialError:
    """``{element} {problem}``: a component's refusal, or a compound member's with the member's pose in the unit."""

    if element.member_id is None:
        return _refuse(NOT_COAXIAL, f"{instance_id!r} {problem}", _where(instance_id, element))
    pose = element.member_pose
    return _refuse(COMPOUND_MEMBER_NOT_COAXIAL, f"{_who(instance_id, element)} (pose in the unit: position_mm {list(pose.position_mm)}, "
                   f"quaternion_wxyz {list(pose.quaternion_wxyz)}) {problem}", _where(instance_id, element))


def _section_matrices(element: CompiledElement, local_x: Vector, walk: _AxisWalk, instance_id: str, axis: str | None, wavelength_nm: float,
                      frame_reference: str) -> tuple[dict[str, Matrix], dict[str, str] | None]:
    """The element's own matrix per system section, and the section mapping of a cylindrical element."""

    where = _where(instance_id, element)
    element_sections = None
    try:
        if _cylindrical(element) and axis is not None:
            c = _dot(local_x, walk.current.section_x)
            if 1.0 - abs(c) <= AXIS_COSINE_TOLERANCE:
                element_sections = {"X": "X", "Y": "Y"}
            elif abs(c) <= PERPENDICULAR_COSINE_TOLERANCE:
                element_sections = {"X": "Y", "Y": "X"}
            else:
                angle = math.degrees(math.acos(min(1.0, abs(c))))
                raise _refuse(CYLINDER_ROTATED, f"the cylinder axes of {_who(instance_id, element)} are turned {angle:.6g} deg about the optical "
                              f"axis from the section axes (the local X / Y of {frame_reference}): its power couples the two sections. "
                              "The paraxial analysis takes cylinders parallel or perpendicular to the section axes", where)
            return {section: paraxial_matrix(element, wavelength_nm, axis=element_sections[section], air=walk.air) for section in SECTIONS}, element_sections
        matrix = paraxial_matrix(element, wavelength_nm, axis=axis, air=walk.air)
        return {section: matrix for section in SECTIONS}, None
    except ValueError as exc:
        if isinstance(exc, ParaxialError):
            raise
        code = PARAXIAL_SURFACE_NOT_SUPPORTED if isinstance(exc, ParaxialSurfaceNotSupported) else ELEMENT_UNSUPPORTED
        raise _refuse(code, str(exc).removeprefix(f"{PARAXIAL_SURFACE_NOT_SUPPORTED}: "), where) from None


def _place_reflector(walk: _AxisWalk, instance_id: str, element: CompiledElement, origin: Vector, axes: tuple[Vector, Vector, Vector],
                     axis: str | None, wavelength_nm: float, frame_reference: str, tolerance: CoaxialTolerance) -> _Placed:
    """A mirror on the current stretch: checks it, and turns the walk at its vertex."""

    where, who = _where(instance_id, element), _who(instance_id, element)
    surface = reflecting_surface(element)
    if surface is not element.model.surfaces[-1]:
        raise _refuse(REFLECTOR_BEHIND_SURFACES, f"{who} surface {surface.surface_id} is a reflector the light reaches through other surfaces of "
                      "the element: the paraxial analysis takes a reflector on the element's outer +Z face", where)
    normal = axes[2]
    vertex = _add(origin, normal, float(Decimal(surface.vertex_z_mm)))
    along, off = walk.locate(vertex)
    if tolerance.off(off):
        raise _off_axis(instance_id, element, f"has its reflecting vertex {off:.6g} mm off the optical axis ({tolerance.lateral_text()})")
    direction = walk.current.direction
    cosine = _dot(direction, normal)
    if cosine > -PERPENDICULAR_COSINE_TOLERANCE:
        raise _refuse(REFLECTOR_BACK_FACE, f"{who} reflects on the side its local +Z faces, and the light along the optical axis "
                      f"({', '.join(f'{v:.6g}' for v in direction)}) does not arrive on that side: turn it around, or check the order of the components", where)
    back = (-direction[0], -direction[1], -direction[2])
    tilt = None if surface.is_plane else _angle(normal, back)
    # (a tilted curved mirror has no single 2 x 2 matrix -- its tangential and sagittal powers differ -- so the Engine's fixed
    # normal-incidence test decides, whatever coaxial_tolerance.tilt_rad says)
    if not surface.is_plane and 1.0 + cosine > AXIS_COSINE_TOLERANCE:
        raise _refuse(CURVED_REFLECTOR_TILTED, f"the local Z axis of the curved reflector {who} is tilted by "
                      f"{math.degrees(math.acos(max(-1.0, min(1.0, -cosine)))):.6g} deg to the optical axis ({tilt:.6g} rad): the paraxial "
                      "analysis takes a curved reflector only at normal incidence, whatever coaxial_tolerance says, and a plane mirror at "
                      "any tilt", where)
    matrices, element_sections = _section_matrices(element, axes[0], walk, instance_id, axis, wavelength_nm, frame_reference)
    point = walk.point(along)
    met_on = len(walk.stretches) - 1
    walk.reflect(along, normal)
    return _Placed(instance_id, element, origin, False, matrices, element_sections, along, along, point, point, direction, walk.current.direction,
                   stretch=met_on, reflector=True, decentre_mm=off, tilt_rad=tilt)


def _place_element(walk: _AxisWalk, instance_id: str, element: CompiledElement, origin: Vector, axes: tuple[Vector, Vector, Vector],
                   axis: str | None, wavelength_nm: float, frame_reference: str, tolerance: CoaxialTolerance) -> _Placed:
    """One element (a component, or a compound member) on the current stretch of the axis."""

    if element.model_kind in ("SURFACE_SET", "FACET_SOLID"):
        raise _refuse(ELEMENT_UNSUPPORTED, f"{element.identity.name} is a {element.model_kind} element: its surfaces need not be coaxial, so it has "
                      "no paraxial matrix", _where(instance_id, element))
    if reflecting_surface(element) is not None:
        return _place_reflector(walk, instance_id, element, origin, axes, axis, wavelength_nm, frame_reference, tolerance)
    direction = walk.current.direction
    cosine = _dot(axes[2], direction)
    facing = direction if cosine >= 0.0 else (-direction[0], -direction[1], -direction[2])
    tilt = _angle(axes[2], facing)
    if tolerance.tilted(axes[2], facing):
        raise _off_axis(instance_id, element, f"has its local Z axis tilted by {math.degrees(math.acos(min(1.0, abs(cosine)))):.6g} deg to the "
                        f"optical axis ({tilt:.6g} rad; {tolerance.tilt_text()}): a transmissive element lies with its local Z along the axis, "
                        "and only a mirror folds it")
    along, off = walk.locate(origin)
    if tolerance.off(off):
        raise _off_axis(instance_id, element, f"is {off:.6g} mm off the optical axis ({tolerance.lateral_text()})")
    turned = cosine < 0.0
    matrices, element_sections = _section_matrices(element, axes[0], walk, instance_id, axis, wavelength_nm, frame_reference)
    z_first, z_last = _vertex_span(element)
    sign = -1.0 if turned else 1.0
    entrance = along + sign * (z_last if turned else z_first)
    exit_ = along + sign * (z_first if turned else z_last)
    return _Placed(instance_id, element, origin, turned, {section: _turn(m) if turned else m for section, m in matrices.items()}, element_sections,
                   entrance, exit_, walk.point(entrance), walk.point(exit_), direction, direction, stretch=len(walk.stretches) - 1,
                   decentre_mm=off, tilt_rad=tilt)


def _member_stations(element: CompiledElement, origin: Vector, axes: tuple[Vector, Vector, Vector]
                     ) -> list[tuple[CompiledElement, Vector, tuple[Vector, Vector, Vector]]]:
    """A compound unit's members in declared order: ``(member, origin, local axes)`` in the design frame."""

    stations = []
    for member in element.members:
        pose = member.member_pose
        own = _local_axes(pose.quaternion_wxyz)
        stations.append((member, _add(origin, _in_frame(axes, pose.position_mm)), (_in_frame(axes, own[0]), _in_frame(axes, own[1]), _in_frame(axes, own[2]))))
    return stations


def _holds_crystal(element: CompiledElement) -> bool:
    """Whether a medium of the element is a uniaxial crystal (``UniaxialMaterial``)."""

    return any(isinstance(m, UniaxialMaterial) for m in element.materials)


def _has_reflector(element: CompiledElement) -> bool:
    return any(reflecting_surface(body) is not None for body in element.bodies())


def _gaps(placed: Sequence[_Placed], instance_ids: Sequence[str], noun: str) -> list[float]:
    """The free-space gaps along the unfolded axis; a next element that begins before the previous one ends is refused."""

    gaps: list[float] = []
    for k, item in enumerate(placed):
        if k:
            gap = item.entrance_mm - placed[k - 1].exit_mm
            if gap < 0.0:
                raise _refuse(ORDER_INVALID, f"{_who(instance_ids[k], item.element)} begins {-gap:.6g} mm before "
                              f"{_who(instance_ids[k - 1], placed[k - 1].element)} ends: list the {noun} in the order light meets them",
                              _where(instance_ids[k], item.element))
            gaps.append(gap)
    return gaps


def _place_compound(walk: _AxisWalk, instance_id: str, element: CompiledElement, origin: Vector, axes: tuple[Vector, Vector, Vector],
                    axis: str | None, wavelength_nm: float, frame_reference: str, tolerance: CoaxialTolerance) -> _Placed:
    """A compound unit: its members' matrices composed with the gaps between them. Pose order along the axis; a unit with a
    mirror in the order it declares its members (module docstring)."""

    placed = [_place_element(walk, instance_id, member, member_origin, member_axes, axis, wavelength_nm, frame_reference, tolerance)
              for member, member_origin, member_axes in _member_stations(element, origin, axes)]
    reflecting = _has_reflector(element)
    if not reflecting:
        placed.sort(key=lambda item: item.entrance_mm)
    gaps = _gaps(placed, [instance_id] * len(placed), "members")
    matrices = {section: _system(placed, gaps, section) for section in SECTIONS}
    head, tail = placed[0], placed[-1]
    turned = not reflecting and _dot(axes[2], head.direction_in) < 0.0
    return _Placed(instance_id, element, origin, turned, matrices, None, head.entrance_mm, tail.exit_mm,
                   head.entrance_point, tail.exit_point, head.direction_in, tail.direction_out, reflector=reflecting, members=tuple(placed))


def _outline_radius(element: CompiledElement) -> float:
    """The largest distance (mm) from its axis of an element's outline: its edge radius, or the corner of its edge rectangle (0 when it states neither)."""

    model = element.model
    edge = getattr(model, "edge_radius_mm", None)
    if edge is not None:
        return float(Decimal(edge))
    widths = getattr(model, "edge_half_widths_mm", None)
    return 0.0 if widths is None else math.hypot(float(Decimal(widths[0])), float(Decimal(widths[1])))


def _crossing_along(walk: _AxisWalk, k: int, item: _Placed, source_along: float | None) -> float | None:
    """Where stretch ``k`` of the axis crosses the plane of a transmissive element inside its outline (unfolded position), else ``None``. The first stretch reaches back to the
    source, ``source_along`` on the axis, where the request names one: the light meets what lies between them."""

    stretch = walk.stretches[k]
    normal = item.direction_in
    cosine = _dot(stretch.direction, normal)
    if abs(cosine) <= PERPENDICULAR_COSINE_TOLERANCE:
        return None  # (the stretch runs along the element's plane)
    distance = _dot(_add(item.origin, stretch.start_mm, -1.0), normal) / cosine
    along = stretch.start_along_mm + distance
    end = walk.stretches[k + 1].start_along_mm if k + 1 < len(walk.stretches) else math.inf
    begin = min(stretch.start_along_mm, source_along) if k == 0 and source_along is not None else stretch.start_along_mm
    if not begin + LATERAL_TOLERANCE_MM < along < end - LATERAL_TOLERANCE_MM:
        return None
    off = _add(_add(stretch.start_mm, stretch.direction, distance), item.origin, -1.0)
    return along if math.sqrt(_dot(off, off)) <= _outline_radius(item.element) else None


def _check_crossings(walk: _AxisWalk, placed: Sequence[_Placed], ids: Sequence[str], source_along: float | None, *, stretches: int | None = None) -> None:
    """A folded axis that crosses a transmissive element of the request where the list does not name it: the numbers would leave that crossing out (module docstring).
    ``stretches``: check only the first this many stretches of the axis."""

    count = len(walk.stretches) if stretches is None else stretches
    if len(walk.stretches) == 1:
        return
    bodies = [(top, member) for top in placed for member in (top.members or (top,))]
    met = {(member.instance_id, member.element.member_id, member.stretch) for _, member in bodies}
    seen: set[tuple[str, str | None]] = set()
    for top, member in bodies:
        body = (member.instance_id, member.element.member_id)
        if member.reflector or body in seen:
            continue
        seen.add(body)
        for k in range(count):
            if (*body, k) in met:
                continue
            along = _crossing_along(walk, k, member, source_along)
            if along is None:
                continue
            who, where = _who(top.instance_id, member.element), _where(top.instance_id, member.element)
            if member.element.member_id is not None:
                raise _refuse(ELEMENT_CROSSED_AGAIN, f"the light crosses {who} again {along:.6g} mm along the axis, and a unit names each member once: list the "
                              "unit's members as separate components, each where the light meets it", where)
            position = sum(1 for p in placed if p.exit_mm <= along)
            raise _refuse(ELEMENT_CROSSED_AGAIN, f"the light crosses {who} again {along:.6g} mm along the axis, and the list does not name it there, so the numbers "
                          f"would leave that crossing out: list it where the light meets it, components {[*ids[:position], top.instance_id, *ids[position:]]}", where)


def _first_direction(stations: Sequence[tuple[CompiledElement, Vector, tuple[Vector, Vector, Vector]]]) -> Vector:
    """The direction of the light at the first element of a system with a mirror and no source (module docstring)."""

    first, origin, axes = stations[0]
    surface = reflecting_surface(first)
    to_next = None
    if len(stations) > 1:
        span = _add(stations[1][1], origin, -1.0)
        if math.sqrt(_dot(span, span)) < MIN_AXIS_LENGTH_MM:
            raise _refuse(ORDER_INVALID, "the first and the second element are at the same position", "components")
        to_next = _unit(span)
    if surface is None:
        return to_next if to_next is not None else axes[2]
    if surface.is_plane and to_next is not None:
        return _mirrored(to_next, axes[2])
    return (-axes[2][0], -axes[2][1], -axes[2][2])


def _element_json(p: _Placed, section: str, sectioned: bool) -> dict[str, Any]:
    return {
        **({"instance_id": p.instance_id} if p.element.member_id is None else {"member_id": p.element.member_id}),
        "model": p.element.optical_sku or p.element.optical_model_id, "model_kind": p.element.model_kind, "turned": p.turned,
        "reflector": p.reflector, "input_plane": "entrance vertex", "output_plane": "exit vertex",
        "entrance_vertex_mm": list(p.entrance_point), "exit_vertex_mm": list(p.exit_point),
        "entrance_vertex_along_axis_mm": p.entrance_mm, "exit_vertex_along_axis_mm": p.exit_mm,
        "vertex_span_mm": p.exit_mm - p.entrance_mm, "abcd": _matrix_json(p.matrices[section]), **_focal(p.matrices[section]),
        **({} if p.members else {"decentre_mm": p.decentre_mm, "tilt_rad": p.tilt_rad}),
        **({} if p.element_sections is None or not sectioned else {"element_section": p.element_sections[section]}),
        **({"members": [_element_json(m, section, sectioned) for m in p.members]} if p.members else {}),
    }


def _axis_path(walk: _AxisWalk, placed: Sequence[_Placed]) -> list[dict[str, Any]]:
    """The straight stretches of the axis, from the first entrance vertex to the last exit vertex."""

    path = []
    for k, stretch in enumerate(walk.stretches):
        start = (placed[0].entrance_mm, placed[0].entrance_point) if k == 0 else (stretch.start_along_mm, stretch.start_mm)
        end = ((placed[-1].exit_mm, placed[-1].exit_point) if k == len(walk.stretches) - 1
               else (walk.stretches[k + 1].start_along_mm, walk.stretches[k + 1].start_mm))
        path.append({"direction": list(stretch.direction), "start_mm": list(start[1]), "end_mm": list(end[1]),
                     "start_along_axis_mm": start[0], "end_along_axis_mm": end[0],
                     "section_x": list(stretch.section_x), "section_y": list(stretch.section_y)})
    return path


@dataclass(frozen=True)
class _Layout:
    """Components placed on the optical axis: what ``paraxial_analysis`` and the cavity analysis read."""

    ids: tuple[str, ...]
    placed: tuple[_Placed, ...]
    gaps: tuple[float, ...]
    walk: _AxisWalk
    optical_axis: Vector
    first: Vector
    frame_x: Vector
    frame_y: Vector
    frame_reference: str
    source: Any  # the source named by source_id (its waist is the object of solve "collimate"), else None
    light: dict[str, Any]  # where the light's direction at the first element comes from (``light_direction`` of the reply)
    lowered: Any
    compiled: dict[str, CompiledElement]
    tolerance: CoaxialTolerance = CoaxialTolerance()


def _project_light(document: Any, first: Vector, listed: Sequence[Vector], tolerance: CoaxialTolerance) -> tuple[list[Any], list[dict[str, Any]]]:
    """``(the Project's sources that light the listed elements along the line through the first one, notes)``: the sources whose
    emission line passes through the first element (its reflecting vertex, or its origin) and that have a listed element ahead of
    them (a source with every listed element behind it lights none of them). Sources that emit along that line in different
    directions do not decide the light's direction: none is taken, and a note names them (``light_direction.notes``; the request
    did not name a source, so the direction is then assumed)."""

    found = []
    for component in document.components:
        if component.optical_role is not OpticalRoleV1.SOURCE or component.source_model is None:
            continue
        origin = tuple(float(v) for v in component.nominal_pose.position_mm)
        emission = _local_z(component.nominal_pose.quaternion_wxyz)
        if not tolerance.off(_lateral(first, origin, emission)[1]) and any(_lateral(point, origin, emission)[0] >= 0.0 for point in (first, *listed)):
            found.append(component)
    directions = [_local_z(c.nominal_pose.quaternion_wxyz) for c in found]
    if any(tolerance.differ(d, directions[0]) for d in directions[1:]):
        return [], [{"code": LIGHT_DIRECTION_AMBIGUOUS, "source_ids": [c.instance_id for c in found]}]
    return found, []


def _light_record(rule: str, direction: Vector, sources: Sequence[Any], notes: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    meaning = {
        SOURCE_LIGHT: "the emission direction of the source named by source_id",
        PROJECT_SOURCE_LIGHT: "the emission direction of the Project's source whose emission line passes through the first element: the light "
                              "path of the Project, the same as with source_id",
        ASSUMED_LIGHT: "no source of the Project decides the light's direction at the first element (none lights it along the line through "
                       "it, or those that do disagree or lie beyond its entrance vertex: notes), so the direction is assumed: from the first "
                       "element to the last; a curved first mirror met on its reflecting side along its axis; a plane first mirror from the "
                       "direction it reflects towards the next element. A list in reverse order then describes the same elements met by "
                       "light travelling the other way",
    }[rule]
    return {"rule": rule, "source_ids": [c.instance_id for c in sources], "direction": list(direction), "meaning": meaning,
            "notes": [dict(note) for note in notes]}


def _lay_out(project: Any, ids: Sequence[str], wavelength_nm: float, axis: str | None, *, source_id: str | None, catalog: Any,  # noqa: C901 - one flat check list
             cavity: bool = False, tolerance: CoaxialTolerance = CoaxialTolerance(), set_aside: Sequence[Mapping[str, Any]] | None = None) -> _Layout:
    """Place the components on the optical axis (module docstring). ``cavity``: the components are a linear cavity listed from one
    end mirror to the other (``os_optics.local.cavity``): the light starts at the first mirror, met on its reflecting side, and
    the last stretch of the axis (back from the last mirror) is the return pass, so it is not checked for crossings. ``set_aside``:
    notes of Project sources the request did not name that do not decide the light's direction; given, the direction is assumed."""

    lowered = lower_project(project, catalog, check_solids=False, require_source=source_id is not None)
    document = load_project(project)
    components = {c.instance_id: c for c in document.components}
    compiled = {e.instance_id: e for e in lowered.elements}
    for instance_id in ids:
        if instance_id not in components:
            raise _refuse(COMPONENT_UNKNOWN, f"no component {instance_id!r} in the Project", f"component {instance_id}")
        if instance_id not in compiled:
            raise _refuse(ELEMENT_UNSUPPORTED, f"component {instance_id!r} ({components[instance_id].optical_role.value}) is not an "
                          "optical element with a simulation model", f"component {instance_id}")
    for k, instance_id in enumerate(ids):
        last = next((j for j in range(k - 1, -1, -1) if ids[j] == instance_id), None)
        if last is None:
            continue
        if cavity:
            raise _refuse(REQUEST_INVALID, f"components names {instance_id!r} twice: a cavity lists each element once, from one end mirror to the other; "
                          "the return pass is part of the round trip", "components")
        if _has_reflector(compiled[instance_id]):
            raise _refuse(REFLECTOR_REPEATED, f"components names the mirror {instance_id!r} twice: a light path meets each mirror of the analysis once. "
                          "Light that returns to a mirror after another one circulates between them: analyse the mirrors as a cavity "
                          "(cavity_analysis, 'os-optics project cavity')", f"component {instance_id}")
        if not any(_has_reflector(compiled[i]) for i in ids[last:k]):
            raise _refuse(REQUEST_INVALID, f"components names {instance_id!r} twice with no mirror between: a component is listed again only where a mirror sends the light "
                          "back to it", "components")
    source = None
    if source_id is not None:
        source = components.get(source_id)
        if source is None or source.optical_role is not OpticalRoleV1.SOURCE or source.source_model is None:
            raise _refuse(COMPONENT_UNKNOWN, f"no source {source_id!r} in the Project", "source_id")
        if source.source_model.model_kind != "GAUSSIAN_BEAM":
            raise _refuse(REQUEST_INVALID, f"source {source_id!r} is an extended emitter, which has no output beam waist: "
                          "leave source_id out for the elements' own paraxial solution", "source_id")

    origins = {i: tuple(float(v) for v in components[i].nominal_pose.position_mm) for i in ids}
    axes_of = {i: _local_axes(components[i].nominal_pose.quaternion_wxyz) for i in ids}
    first = origins[ids[0]]
    frame_label = ids[0]
    folded = any(_has_reflector(compiled[i]) for i in ids)
    stations = []
    if folded:
        for i in ids:
            unit_with_mirror = compiled[i].is_compound and _has_reflector(compiled[i])
            stations += _member_stations(compiled[i], origins[i], axes_of[i]) if unit_with_mirror else [(compiled[i], origins[i], axes_of[i])]
        if stations[0][0].member_id is not None:
            first = stations[0][1]
            frame_label = f"{ids[0]} member {stations[0][0].member_id}"
        lead = reflecting_surface(stations[0][0])
        if lead is not None:
            first = _add(stations[0][1], stations[0][2][2], float(Decimal(lead.vertex_z_mm)))
    # the light path: the named source, else the Project's source on the line through the first element (the same verdict either way)
    notes: list[Mapping[str, Any]] = list(set_aside or ())
    if source is not None:
        lights = [source]
    elif cavity or set_aside is not None:
        lights = []
    else:
        lights, notes = _project_light(document, first, [origins[i] for i in ids], tolerance)
    light = lights[0] if lights else None
    light_axes = None if light is None else _local_axes(light.nominal_pose.quaternion_wxyz)
    if folded:
        optical_axis = light_axes[2] if light_axes is not None else _first_direction(stations)
    elif len(ids) > 1:
        span = _add(origins[ids[-1]], first, -1.0)
        if math.sqrt(_dot(span, span)) < MIN_AXIS_LENGTH_MM:
            raise _refuse(ORDER_INVALID, "the first and the last component are at the same position", "components")
        optical_axis = _unit(span)
    elif light_axes is not None:
        optical_axis = light_axes[2]
    else:
        optical_axis = axes_of[ids[0]][2]
    rule = SOURCE_LIGHT if source is not None else (PROJECT_SOURCE_LIGHT if lights else ASSUMED_LIGHT)
    named = "" if source is not None else " (the Project's source on the line through the first element)"

    if light is not None and light_axes is not None:
        cosine = _dot(light_axes[2], optical_axis)
        if tolerance.tilted(light_axes[2], optical_axis):
            against = (-optical_axis[0], -optical_axis[1], -optical_axis[2])
            if cosine < 0.0 and not tolerance.tilted(light_axes[2], against):
                message = f"source {light.instance_id!r}{named} emits against the optical axis (from {ids[0]!r} to {ids[-1]!r})"
            else:
                message = (f"the emission direction of source {light.instance_id!r}{named} is tilted by "
                           f"{math.degrees(math.acos(max(-1.0, min(1.0, cosine)))):.6g} deg to the optical axis ({tolerance.tilt_text()})")
            raise _refuse(NOT_COAXIAL, message, "source_id")
        _along, off = _lateral(tuple(float(v) for v in light.nominal_pose.position_mm), first, optical_axis)
        if tolerance.off(off):
            raise _refuse(NOT_COAXIAL, f"source {light.instance_id!r} is {off:.6g} mm off the optical axis (the line through "
                          f"{ids[0]!r} along the optical axis; {tolerance.lateral_text()}): the paraxial analysis is for a source on that axis",
                          "source_id")
    if source is not None and light_axes is not None:
        frame_x, frame_y, frame_reference = light_axes[0], light_axes[1], f"source {source.instance_id}"
    else:
        own = stations[0][2] if folded else axes_of[ids[0]]
        frame_x, frame_y, frame_reference = own[0], own[1], f"component {frame_label}"
        if folded and reflecting_surface(stations[0][0]) is not None and _dot(optical_axis, own[2]) < -PERPENDICULAR_COSINE_TOLERANCE:
            # the axes of a first plane mirror lie in its tilted plane: the section axes are their parts across the light
            frame_x = _unit(_across(frame_x, optical_axis))
            frame_y = _unit(_across(_across(frame_y, optical_axis), frame_x))

    walk = _AxisWalk(first, optical_axis, frame_x, frame_y)
    walk.air = lowered.air
    placed: list[_Placed] = []
    for instance_id in ids:
        place = _place_compound if compiled[instance_id].is_compound else _place_element
        try:
            placed.append(place(walk, instance_id, compiled[instance_id], origins[instance_id], axes_of[instance_id], axis, wavelength_nm, frame_reference,
                                tolerance))
        except ParaxialError as exc:
            if light is None or exc.problems[0]["code"] != REFLECTOR_BACK_FACE:
                raise
            problem = dict(exc.problems[0], message=f"{exc.problems[0]['message']}. The light's direction is the emission direction of source "
                                                    f"{light.instance_id!r}{named}")
            raise ParaxialError([problem]) from None
    gaps = _gaps(placed, list(ids), "components")
    source_along = None
    if light is not None:
        source_along, _off = _lateral(tuple(float(v) for v in light.nominal_pose.position_mm), first, optical_axis)
        if source_along > placed[0].entrance_mm and source is None:
            # a source the request did not name, beyond the first entrance vertex: it does not light the components in the listed
            # order, so it is set aside with a note and the layout is made with the assumed direction
            return _lay_out(project, ids, wavelength_nm, axis, source_id=None, catalog=catalog, cavity=cavity, tolerance=tolerance,
                            set_aside=[{"code": SOURCE_NOT_BEFORE_COMPONENTS, "source_ids": [c.instance_id for c in lights]}])
        if source_along > placed[0].entrance_mm:
            raise _refuse(ORDER_INVALID, f"source {light.instance_id!r}{named} lies {source_along - placed[0].entrance_mm:.6g} mm beyond the "
                          f"entrance vertex of {placed[0].instance_id!r} along the optical axis: its light does not reach the components "
                          "in the listed order", "source_id")
    _check_crossings(walk, placed, list(ids), source_along, stretches=len(walk.stretches) - 1 if cavity else None)
    return _Layout(tuple(ids), tuple(placed), tuple(gaps), walk, optical_axis, first, frame_x, frame_y, frame_reference, source,
                   _light_record(rule, optical_axis, lights, notes), lowered, compiled, tolerance)


def paraxial_analysis(project: Any, request: Mapping[str, Any], *, catalog: Any = None) -> dict[str, Any]:
    """The paraxial analysis of placed components (module docstring). ``project``: a Project (as ``load_project`` takes it);
    ``catalog``: a catalog release instead of the one the Project pins. Raises ``ParaxialError`` for a request it cannot
    answer and ``LocalRunError`` for a Project the runtime cannot lower."""

    if not isinstance(request, Mapping):
        raise _refuse(REQUEST_INVALID, "a request is a JSON object with components and wavelength_nm")
    unknown = sorted(set(request) - REQUEST_KEYS)
    if unknown:
        raise _refuse(REQUEST_INVALID, f"unknown request fields {unknown}; a request takes {sorted(REQUEST_KEYS)}")
    ids = request.get("components")
    if not isinstance(ids, (list, tuple)) or not ids or not all(isinstance(i, str) for i in ids):
        raise _refuse(REQUEST_INVALID, "components is a non-empty list of instance ids, in the order light meets them", "components")
    wavelength_nm = _number(request.get("wavelength_nm"), "wavelength_nm")
    axis, solve, source_id = request.get("axis"), request.get("solve"), request.get("source_id")
    if axis not in (None, *SECTIONS):
        raise _refuse(REQUEST_INVALID, f"axis is 'X', 'Y' or absent, not {axis!r}", "axis")
    if solve not in (None, *SOLVES):
        raise _refuse(REQUEST_INVALID, f"solve is one of {list(SOLVES)} or absent, not {solve!r}", "solve")
    if solve == "afocal" and len(ids) != 2:
        raise _refuse(REQUEST_INVALID, "solve 'afocal' takes exactly two components", "components")
    if source_id is not None and not isinstance(source_id, str):
        raise _refuse(REQUEST_INVALID, "source_id is a string", "source_id")
    configuration_id = request.get("configuration_id")
    if configuration_id is not None:
        # (one configuration of a Project that declares several: analysed as its own Project, configured(id))
        from .configurations import configured_project

        project = configured_project(project, configuration_id)

    tolerance = coaxial_tolerance(request.get("coaxial_tolerance"))
    layout = _lay_out(project, ids, wavelength_nm, axis, source_id=source_id, catalog=catalog, tolerance=tolerance)
    for item in layout.placed:
        crystal = next((part for part in (item.members or (item,)) if _holds_crystal(part.element)), None)
        if crystal is not None:
            raise _refuse(BIREFRINGENT_ELEMENT, f"{_who(item.instance_id, crystal.element)} holds a uniaxial crystal: light polarized "
                          "across its optic axis meets n_o and light polarized along it n_e (an extraordinary wave also diffracts "
                          "differently along and across the axis), so the element has no single paraxial matrix; trace the design, "
                          "or analyse a cavity holding it with os_optics.local.cavity and its polarization",
                          _where(item.instance_id, crystal.element))
    placed, gaps, walk, optical_axis, first, source = list(layout.placed), list(layout.gaps), layout.walk, layout.optical_axis, layout.first, layout.source
    frame_x, frame_y, frame_reference = layout.frame_x, layout.frame_y, layout.frame_reference
    section = axis or "X"
    systems = {s: _system(placed, gaps, s) for s in SECTIONS}
    system = systems[section]

    reply: dict[str, Any] = {
        # (the configuration analysed, of a Project that declares several: stated only when one was named)
        **({"configuration_id": configuration_id} if configuration_id is not None else {}),
        "wavelength_nm": wavelength_nm, "section_axis": axis, "optical_axis": list(optical_axis), "conventions": dict(CONVENTIONS),
        "section_frame": {"reference": frame_reference, "x": list(frame_x), "y": list(frame_y)}, "axis_path": _axis_path(walk, placed),
        "light_direction": layout.light, "coaxial_tolerance": tolerance.as_dict(),
        "elements": [_element_json(p, section, axis is not None) for p in placed],
    }
    focal = _focal(system)
    head, tail = placed[0], placed[-1]
    reply["system"] = {
        "input_plane": f"entrance vertex of {head.instance_id}", "output_plane": f"exit vertex of {tail.instance_id}",
        "input_plane_mm": list(head.entrance_point), "output_plane_mm": list(tail.exit_point), "vertex_gaps_mm": gaps, "abcd": _matrix_json(system),
        "afocal": focal["efl_mm"] is None, **focal,
        "front_focal_point_mm": None if focal["ffd_mm"] is None else list(_add(head.entrance_point, head.direction_in, -focal["ffd_mm"])),
        "rear_focal_point_mm": None if focal["bfl_mm"] is None else list(_add(tail.exit_point, tail.direction_out, focal["bfl_mm"])),
    }
    if axis is not None:
        reply["sections"] = {s: {"abcd": _matrix_json(m), "afocal": _focal(m)["efl_mm"] is None, **_focal(m)} for s, m in systems.items()}
    if solve == "afocal":
        reply["solution"] = _afocal(placed, section, axis is not None)
    elif solve == "collimate":
        # (a Gaussian beam in the environment's air: its wavelength there, lambda / n_air)
        in_air_nm = wavelength_nm / surrounding_index(layout.lowered.air, wavelength_nm)
        reply["solution"] = _collimate(placed, systems, section, optical_axis, first, source, in_air_nm, axis)
    return reply


def _afocal(placed: list[_Placed], section: str, sectioned: bool) -> dict[str, Any]:
    one, two = placed
    (a1, _b1), (c1, _d1) = one.matrices[section]
    (_a2, _b2), (c2, d2) = two.matrices[section]
    in_section = f" in section {section}" if sectioned else ""
    if abs(c1) < AFOCAL_C_PER_MM or abs(c2) < AFOCAL_C_PER_MM:
        raise _refuse(NO_SOLUTION, f"an afocal pair needs two elements with optical power{in_section}; "
                      f"{one.instance_id if abs(c1) < AFOCAL_C_PER_MM else two.instance_id!r} has none{in_section}", "components")
    gap = -(c2 * a1 + d2 * c1) / (c1 * c2)
    if gap < 0.0:
        raise _refuse(NO_SOLUTION, f"the afocal vertex gap would be {gap:.6g} mm: these two elements cannot be afocal without "
                      "overlapping", "components")
    current = two.entrance_mm - one.exit_mm
    move = gap - current
    free = ((1.0, gap), (0.0, 1.0))
    system = _multiply(two.matrices[section], _multiply(free, one.matrices[section]))
    solution: dict[str, Any] = {
        "kind": "AFOCAL", "section": section if sectioned else None, "vertex_gap_mm": gap, "current_vertex_gap_mm": current,
        "component": two.instance_id, "move_along_axis_mm": move, "move_direction": list(two.direction_in),
        "position_mm": list(_add(two.origin, two.direction_in, move)),
        "abcd": _matrix_json(system), "beam_size_magnification": system[0][0], "angular_magnification": system[1][1],
        "meaning": f"move {two.instance_id} by move_along_axis_mm along move_direction, the light's direction arriving at it (to "
                   f"position_mm): the pair is then afocal"
                   f"{in_section} (system C = 0): in geometric optics a collimated input leaves collimated, its diameter scaled by "
                   "|beam_size_magnification| (negative: inverted image). A Gaussian beam's output waist position depends on where "
                   "its input waist lies",
        "notes": [],
    }
    if sectioned:
        other = "Y" if section == "X" else "X"
        other_system = _multiply(two.matrices[other], _multiply(free, one.matrices[other]))
        other_focal = _focal(other_system)
        solution["other_section"] = {"section": other, "abcd": _matrix_json(other_system), "afocal": other_focal["efl_mm"] is None,
                                     **other_focal}
        if other_focal["efl_mm"] is not None:
            solution["notes"].append({
                "code": "OTHER_SECTION_NOT_AFOCAL",
                "message": f"at this gap section {other} is not afocal (EFL {other_focal['efl_mm']:.6g} mm): a collimated input leaves "
                           f"collimated in section {section} only"})
    return solution


def _beam_state(solved: bool, input_rayleigh_mm: float, efl_mm: float | None) -> str:
    if not solved:
        return UNCOLLIMATED
    return COLLIMATED if efl_mm is not None and input_rayleigh_mm <= abs(efl_mm) else FOCUSED


def _collimate(placed: list[_Placed], systems: dict[str, Matrix], section: str, optical_axis: Vector, first: Vector, source: Any,
               wavelength_nm: float, axis: str | None) -> dict[str, Any]:
    focal = _focal(systems[section])
    in_section = "" if axis is None else f" in section {axis}"
    if focal["ffd_mm"] is None:
        raise _refuse(NO_SOLUTION, f"the elements are afocal together{in_section}: no finite object distance collimates the output", "components")
    distance = focal["ffd_mm"]
    entrance = placed[0].entrance_mm
    solution: dict[str, Any] = {
        "kind": "COLLIMATE", "section": axis, "object_distance_mm": distance,
        "object_point_mm": list(_add(first, optical_axis, entrance - distance)),
        "meaning": f"the object object_distance_mm before the entrance vertex, at the front focal point{in_section}: a point source "
                   "there leaves the elements collimated (geometric optics); a Gaussian waist there leaves with the largest output "
                   "waist and the smallest far-field divergence the elements give it, which is a collimated beam only when the "
                   "source's Rayleigh range is at most |EFL| (give source_id: output_beam.<axis>.state)",
    }
    if source is None:
        return solution
    model = source.source_model
    waist_point = _add(tuple(float(v) for v in source.nominal_pose.position_mm), _local_z(source.nominal_pose.quaternion_wxyz),
                       float(model.waist_position_mm))
    waist_along = _dot(_add(waist_point, first, -1.0), optical_axis)
    current = entrance - waist_along
    move = distance - current
    wavelength_mm = wavelength_nm * NM_TO_MM
    beams: dict[str, Any] = {}
    notes: list[dict[str, str]] = []
    for beam_axis, beam_section, waist, m2 in (("x", "X", model.waist_radius_mm, model.m2),
                                               ("y", "Y", model.waist_radius_y_mm or model.waist_radius_mm, model.m2_y or model.m2)):
        own = _focal(systems[beam_section])
        solved = own["ffd_mm"] is not None and abs(own["ffd_mm"] - distance) <= SAME_FOCUS_REL * max(1.0, abs(distance))
        beam = {"section": beam_section, "input_waist_radius_mm": float(waist), "m2": float(m2),
                **_gaussian_out(systems[beam_section], distance, float(waist), float(m2), wavelength_mm)}
        beam["state"] = _beam_state(solved, beam["input_rayleigh_range_mm"], own["efl_mm"])
        beams[beam_axis] = beam
        if beam["state"] == FOCUSED:
            notes.append({"code": "INPUT_ALREADY_COLLIMATED",
                          "message": f"source axis {beam_axis}: its Rayleigh range ({beam['input_rayleigh_range_mm']:.6g} mm) is longer than "
                                     f"|EFL| ({abs(own['efl_mm']):.6g} mm), so the beam is already collimated over the elements' focal "
                                     f"length; at this distance they focus it to a waist of {beam['waist_radius_mm']:.6g} mm "
                                     f"{beam['waist_after_output_plane_mm']:.6g} mm after the last exit vertex"})
        elif beam["state"] == UNCOLLIMATED:
            where = ("its section has no power" if own["ffd_mm"] is None else
                     f"its section's front focal point is {own['ffd_mm']:.6g} mm before the entrance vertex")
            notes.append({"code": "OTHER_SECTION_NOT_COLLIMATED",
                          "message": f"source axis {beam_axis} (section {beam_section}) is not collimated: {where}"})
    solution.update({
        "source_id": source.instance_id, "source_waist_mm": list(waist_point), "current_object_distance_mm": current,
        "move_along_axis_mm": move, "components": list(dict.fromkeys(p.instance_id for p in placed)),
        "positions_mm": {p.instance_id: list(_add(p.origin, optical_axis, move)) for p in placed},
        "output_beam": beams, "notes": notes,
        "output_beam_meaning": "the source's Gaussian beam after the elements moved by move_along_axis_mm, per source axis through its "
                               "own section (x: section X, y: section Y): radii are the beam's second-moment radii (Rayleigh range "
                               "pi w0^2 / (M^2 lambda)); waist_after_output_plane_mm from the last exit vertex (positive: after it); "
                               "state COLLIMATED (waist at the front focal point, input Rayleigh range <= |EFL|), FOCUSED (input "
                               "Rayleigh range > |EFL|: the elements focus the beam) or UNCOLLIMATED (this section's front focal "
                               "point lies elsewhere)",
    })
    return solution


__all__ = ["COMPOUND_MEMBER_NOT_COAXIAL", "CONVENTIONS", "CURVED_REFLECTOR_TILTED", "CYLINDER_ROTATED", "ELEMENT_CROSSED_AGAIN", "LIGHT_DIRECTION_AMBIGUOUS",
           "PARAXIAL_ANALYSIS_INVALID", "ParaxialError", "REFLECTOR_BACK_FACE", "REFLECTOR_BEHIND_SURFACES", "REFLECTOR_REPEATED", "paraxial_analysis"]
