"""Native hollow retroreflectors: compound units (``COMPOUND``) of flat rectangular mirrors at fixed mutual angles.

Every member is a flat mirror (``SURFACE_SEQUENCE``, one ``REFLECTED`` surface, face-normal frame) with the declared
coating of ``mirror.flat`` and a drawn substrate behind its face. Neighbouring faces leave a seam of ``seam_mm`` along their
common edge (each face starts that far from the other mirror's plane), so no two mirror bodies touch; light that reaches a
seam passes between the mirrors. Unit frame: local +Z is the retroreflector's axis, pointing from the apex toward the light;
light enters travelling along -Z and leaves along +Z (ports ``IN`` and ``OUT`` at the aperture plane).

* ``mirror.hollow_roof``: two mirrors at 90 deg meeting along a ridge on local X (the apex line at ``z = 0``), each
  inclined at 45 deg to the axis; the aperture is ``aperture_width_mm`` along the ridge by ``aperture_height_mm`` across it
  (each face reaches from the seam to the aperture's edge).
  Light whose direction lies in the YZ plane comes back antiparallel; a ray entering at height ``y`` leaves at ``-y``
  (the transverse coordinate across the ridge is reversed, the one along it kept): the image is inverted in one axis.
  Reflection matrix ``diag(1, -1, -1)`` in the unit frame (two plane reflections).
* ``retroreflector.hollow_corner``: three mutually orthogonal square mirrors (a cube corner) of edge ``mirror_edge_mm``
  (each in its quadrant from ``seam_mm`` to ``seam_mm + mirror_edge_mm`` along both of its axes),
  the axis along the cube diagonal; any ray that meets all three mirrors leaves antiparallel to its incoming direction
  (``-I``, three plane reflections) and offset by its point reflection through the axis (Hecht, *Optics*, 5th ed., sec.
  5.5.3; Born & Wolf, *Principles of Optics*, 7th ed., sec. 4.9). Corner frame: the mirrors lie in the planes ``x = 0``,
  ``y = 0``, ``z = 0`` with normals into the octant ``x, y, z >= 0``; unit axes ``X = (1, -1, 0) / sqrt 2``,
  ``Y = (1, 1, -2) / sqrt 6``, ``Z = (1, 1, 1) / sqrt 3``.

Member poses are exact decimals rounded to floats for the Project pose (positions to 1e-6 mm).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any

from .._vendor.shared_contracts.design_catalog import CoatingKind, FactStatus, ModelFidelity, SimulationApproximation
from .common import BuildContext, NativeBuild, aoi_range, band, box, declared_coating, default_model_id, face_frame, finish, optical_model
from .errors import invalid
from .lenses import coating_approximations
from .numbers import HALF, LENGTH_QUANTUM_MM, ONE, TWO, WORK_CONTEXT, ZERO, ceil_to, length, quantize, text
from .plates import MIRROR_THICKNESS_QUANTUM_MM, _reflectance

Vec = tuple[Decimal, Decimal, Decimal]
#: Drawn substrate thickness of a member mirror when none is given: a tenth of the face's longer side, at least 1 mm.
SUBSTRATE_FRACTION = Decimal("0.1")
#: Gap between neighbouring mirror faces along their common edge when none is given (mm).
DEFAULT_SEAM_MM = Decimal("0.01")
#: Longest unit model id that leaves room for the member suffixes (``-A`` .. ``-C``).
MAX_UNIT_ID = 60


def _sqrt(value: int) -> Decimal:
    with localcontext(WORK_CONTEXT):
        return Decimal(value).sqrt()


def _matvec(m: tuple[Vec, Vec, Vec], v: Vec) -> Vec:
    with localcontext(WORK_CONTEXT):
        return tuple(m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3))  # type: ignore[return-value]


def quaternion_of(columns: tuple[Vec, Vec, Vec]) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """The unit quaternion ``(w, x, y, z)`` of the rotation whose matrix has the given columns (images of the member's local
    X, Y, Z axes), by Shepperd's method in decimal arithmetic."""

    (m00, m10, m20), (m01, m11, m21), (m02, m12, m22) = columns
    with localcontext(WORK_CONTEXT):
        trace = m00 + m11 + m22
        if trace > 0:
            s = (trace + ONE).sqrt() * TWO
            q = (s / 4, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
        elif m00 > m11 and m00 > m22:
            s = (ONE + m00 - m11 - m22).sqrt() * TWO
            q = ((m21 - m12) / s, s / 4, (m01 + m10) / s, (m02 + m20) / s)
        elif m11 > m22:
            s = (ONE + m11 - m00 - m22).sqrt() * TWO
            q = ((m02 - m20) / s, (m01 + m10) / s, s / 4, (m12 + m21) / s)
        else:
            s = (ONE + m22 - m00 - m11).sqrt() * TWO
            q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4)
        if q[0] < 0:
            q = tuple(-c for c in q)  # type: ignore[assignment]
        return q  # type: ignore[return-value]


def _mirror_member(model_id: str, name: str, half_x: Decimal, half_y: Decimal, thickness: Decimal, coating, low: Decimal, high: Decimal):
    """A flat rectangular mirror (face-normal frame: the face in z = 0, +Z toward the light) that reflects over its whole face."""

    surface = {"surface_id": "S1", "vertex_z_mm": "0", "clear_half_widths_mm": [text(half_x), text(half_y)], "coating_id": coating.coating_id,
               "traced": "REFLECTED", "status": FactStatus.DESIGN_BASIS.value}
    approximations = {SimulationApproximation.EDGE_AS_OPAQUE_RIM, SimulationApproximation.IDEAL_SURFACE_FIGURE,
                      SimulationApproximation.TRANSMITTED_LIGHT_NOT_TRACED} | coating_approximations([coating])
    physics = {"model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SPECIFICATION_ENVELOPE.value,
               "surfaces": [surface], "media": [], "edge_half_widths_mm": [text(half_x), text(half_y)],
               "valid_band_nm": band(low, high, FactStatus.DESIGN_BASIS), "approximations": sorted(a.value for a in approximations),
               "status": FactStatus.DESIGN_BASIS.value}
    return optical_model(model_id, name, physics, frame=face_frame(rotational=False), geometry=box(half_x, half_y, -thickness, ZERO))


def _coating(values: Mapping[str, Any]):
    kind = CoatingKind(values.get("coating_kind") or "METALLIC_REFLECTOR")
    r, t, polarized, lossless = _reflectance(values, kind)
    low, high = values["valid_band_nm"]
    coating = declared_coating(kind, low, high, r, t, polarized=polarized, aoi=aoi_range(values.get("valid_aoi_deg"), "valid_aoi_deg"),
                               basis="Declared mirror coating" + (" (transmittance not stated: taken as 0 for a metal film, 1 - R "
                                                                  "for a dielectric stack)" if lossless else ""),
                               parameter="reflectance")
    return coating, r, low, high


def _member(member_id: str, model_id: str, position: Vec, columns: tuple[Vec, Vec, Vec]) -> dict[str, Any]:
    q = quaternion_of(columns)
    return {"member_id": member_id, "model_id": model_id,
            "pose": {"position_mm": [float(quantize(c, LENGTH_QUANTUM_MM)) for c in position], "quaternion_wxyz": [float(c) for c in q]}}


def _ports(z: Decimal) -> list[dict[str, Any]]:
    point = ["0", "0", length(z)]
    return [{"port_id": "IN", "role": "INPUT", "point_mm": point, "direction": ["0", "0", "-1"], "status": FactStatus.DESIGN_BASIS.value},
            {"port_id": "OUT", "role": "OUTPUT", "point_mm": point, "direction": ["0", "0", "1"], "status": FactStatus.DERIVED.value}]


def _envelope(corners: list[Vec]):
    with localcontext(WORK_CONTEXT):
        half_x = max(abs(c[0]) for c in corners)
        half_y = max(abs(c[1]) for c in corners)
        return box(half_x, half_y, min(c[2] for c in corners), max(c[2] for c in corners))


def _box_corners(position: Vec, columns: tuple[Vec, Vec, Vec], half_x: Decimal, half_y: Decimal, thickness: Decimal) -> list[Vec]:
    """The corners of a member's drawn substrate (``[-hx, hx] x [-hy, hy] x [-t, 0]`` in its frame) in the unit frame."""

    out = []
    with localcontext(WORK_CONTEXT):
        for sx in (-half_x, half_x):
            for sy in (-half_y, half_y):
                for sz in (-thickness, ZERO):
                    out.append(tuple(position[i] + columns[0][i] * sx + columns[1][i] * sy + columns[2][i] * sz for i in range(3)))
    return out  # type: ignore[return-value]


def _thickness(values: Mapping[str, Any], longest: Decimal) -> Decimal:
    given = values.get("substrate_thickness_mm")
    return given if given is not None else max(ONE, ceil_to(SUBSTRATE_FRACTION * longest, MIRROR_THICKNESS_QUANTUM_MM))


def _unit_id(values: Mapping[str, Any], *parts: object) -> str:
    return (values.get("model_id") or default_model_id(*parts))[:MAX_UNIT_ID]  # type: ignore[arg-type]


def build_hollow_roof(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    width, height = values["aperture_width_mm"], values["aperture_height_mm"]
    coating, r, low, high = _coating(values)
    root2 = _sqrt(2)
    with localcontext(WORK_CONTEXT):
        seam = values.get("seam_mm") or DEFAULT_SEAM_MM
        face = height / root2 - seam  # along the slope: from the seam to the aperture's edge (0, +-h/2, h/2)
        if face <= 0:
            raise invalid("the seam is wider than the faces: use a larger aperture_height_mm or a smaller seam_mm", "seam_mm")
        half_x, half_y = width / TWO, face / TWO
        a = (seam + face / TWO) / root2  # the face centre is (0, +-a, a)
        c = ONE / root2
    thickness = _thickness(values, max(width, face))
    unit_id = _unit_id(values, "hollow-roof", "w", width, "h", height)
    faces = (("A", (ZERO, a, a), ((ONE, ZERO, ZERO), (ZERO, c, c), (ZERO, -c, c))),     # normal (0, -1, 1)/sqrt 2: faces the y < 0 side
             ("B", (ZERO, -a, a), ((ONE, ZERO, ZERO), (ZERO, c, -c), (ZERO, c, c))))    # normal (0, 1, 1)/sqrt 2
    members, models, corners = [], [], []
    for member_id, position, columns in faces:
        model_id = f"{unit_id}-{member_id}"
        models.append(_mirror_member(model_id, f"Hollow roof mirror {member_id}", half_x, half_y, thickness, coating, low, high))
        members.append(_member(member_id, model_id, position, columns))
        corners += _box_corners(position, columns, half_x, half_y, thickness)
    with localcontext(WORK_CONTEXT):
        top = height / TWO
    physics = {"model_kind": "COMPOUND", "model_version": "1", "members": members, "envelope": _envelope(corners), "ports": _ports(top)}
    model = optical_model(unit_id, values.get("display_name") or f"Hollow roof mirror {text(width)} x {text(height)} mm", physics)
    design = {"aperture_mm": [text(width), text(height)], "face_mm": [text(width), length(face)], "seam_mm": text(seam), "reflectance": text(r),
              "coating_id": coating.coating_id, "drawn_substrate_thickness_mm": text(thickness), "ridge": "local X at z = 0",
              "reflection_matrix": "diag(1, -1, -1): a ray at height y returns at -y, antiparallel in the YZ plane",
              "equations": "two plane mirrors at 90 deg: R = R_B R_A = diag(1, -1, -1) in the unit frame"}
    return finish("mirror.hollow_roof", model, members=tuple(models), coatings=[coating], role="mirror", values=values, design=design)


def build_hollow_corner(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    edge = values["mirror_edge_mm"]
    coating, r, low, high = _coating(values)
    root2, root3, root6 = _sqrt(2), _sqrt(3), _sqrt(6)
    with localcontext(WORK_CONTEXT):
        # rows: the unit axes in corner coordinates (R maps corner coordinates to unit coordinates)
        rotation = ((ONE / root2, -ONE / root2, ZERO), (ONE / root6, ONE / root6, -TWO / root6), (ONE / root3, ONE / root3, ONE / root3))
        seam = values.get("seam_mm") or DEFAULT_SEAM_MM
        h = edge * HALF
        m = seam + h  # each face is centred m from the two planes it borders
    thickness = _thickness(values, edge)
    unit_id = _unit_id(values, "hollow-corner", "e", edge)
    e = ((ONE, ZERO, ZERO), (ZERO, ONE, ZERO), (ZERO, ZERO, ONE))
    # each mirror in the corner frame: (centre, local X, local Y, normal Z), right-handed, the normal into the octant
    faces = (("A", (m, m, ZERO), (e[0], e[1], e[2])), ("B", (ZERO, m, m), (e[1], e[2], e[0])), ("C", (m, ZERO, m), (e[2], e[0], e[1])))
    members, models, corners = [], [], []
    for member_id, centre, axes in faces:
        position = _matvec(rotation, centre)
        columns = tuple(_matvec(rotation, axis) for axis in axes)
        model_id = f"{unit_id}-{member_id}"
        models.append(_mirror_member(model_id, f"Hollow corner mirror {member_id}", h, h, thickness, coating, low, high))
        members.append(_member(member_id, model_id, position, columns))  # type: ignore[arg-type]
        corners += _box_corners(position, columns, h, h, thickness)  # type: ignore[arg-type]
    with localcontext(WORK_CONTEXT):
        top = TWO * (seam + edge) / root3  # the far corners (s + e, s + e, 0) etc. lie at height 2 (s + e) / sqrt 3
    physics = {"model_kind": "COMPOUND", "model_version": "1", "members": members, "envelope": _envelope(corners), "ports": _ports(top)}
    model = optical_model(unit_id, values.get("display_name") or f"Hollow corner retroreflector {text(edge)} mm", physics)
    design = {"mirror_edge_mm": text(edge), "seam_mm": text(seam), "reflectance": text(r), "coating_id": coating.coating_id,
              "drawn_substrate_thickness_mm": text(thickness), "aperture_plane_z_mm": length(top),
              "axis": "local +Z = cube diagonal (1, 1, 1)/sqrt 3 of the corner frame",
              "reflection_matrix": "-I: every ray meeting all three mirrors returns antiparallel",
              "equations": "three mutually orthogonal plane mirrors: R = R_C R_B R_A = -I"}
    return finish("retroreflector.hollow_corner", model, members=tuple(models), coatings=[coating], role="mirror", values=values, design=design)


__all__ = ["build_hollow_corner", "build_hollow_roof", "quaternion_of"]
