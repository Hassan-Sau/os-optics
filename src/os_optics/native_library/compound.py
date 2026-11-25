"""Native compound units (``COMPOUND`` models): telescopes and a 4f relay built from native lens classes.

A unit's members are ordinary native lens models (``lens.ideal_thin``, ``lens.singlet`` or ``lens.achromat``, given as
``{"class_id": ..., "parameters": {...}}``) and, optionally, a native stop; the unit is placed as one component, and every
member surface is traced. The unit frame's +Z is its optical axis; members sit on it, unrotated.

Spacing (paraxial, at ``design_wavelength_nm`` for refracting members): with ``BFL`` the back and ``FFL`` the front focal
length of each lens (``paraxial.py``), the second lens is placed so that its front focal point coincides with the first
lens's rear focal point: ``z_2 = z_1 + last_1 + BFL_1 + FFL_2``. The unit is then afocal (its system matrix has ``C = 0``).

* ``compound.keplerian_telescope``: two positive lenses (real internal focus); beam magnification ``M = -f_2 / f_1``
  (inverted). An optional pinhole at the internal focus makes it a spatial filter. Unit origin: the first lens's frame.
* ``compound.galilean_telescope``: lenses of opposite signs (no internal focus); ``M = -f_2 / f_1`` (upright).
* ``compound.relay_4f``: two positive lenses with the object plane at the unit origin (the first lens's front focal
  plane), the Fourier plane between them and the image plane at the second lens's rear focal plane (port ``OUT``);
  lateral magnification ``m = -f_2 / f_1``, telecentric. An optional Fourier-plane stop (low-pass opening, central DC
  block or both: a band-pass annulus) filters spatial frequencies geometrically.

(Hecht, *Optics*, 5th ed., sec. 5.7.7 (telescopes) and 11.3 (4f spatial filtering); Goodman, *Introduction to Fourier
Optics*, 3rd ed., sec. 5.2.)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal, localcontext
from typing import Any

from .._vendor.shared_contracts.design_catalog import FactStatus
from .common import BuildContext, NativeBuild, cylinder, default_model_id, finish, optical_model
from .errors import infeasible, invalid
from .materials import design_air_index, design_index
from .numbers import LENGTH_QUANTUM_MM, ONE, TWO, WORK_CONTEXT, ZERO, length, quantize, text, to_decimal
from .paraxial import FirstOrder, ParaxialSurface, multiply, sequence_first_order, sequence_matrix, transfer
from .stops import stop_model

NestedBuilder = Callable[[Mapping[str, Any], str, BuildContext, str], NativeBuild]
#: Drawn thickness of a stop inside a compound unit (a thin foil / plate) when none is given.
COMPOUND_STOP_THICKNESS_MM = Decimal("0.1")
#: Longest unit model id that still leaves room for the member suffixes (``-L1``, ``-STOP``).
MAX_UNIT_ID = 58


def _matrix_and_order(build: NativeBuild, wavelength: Decimal | None, parameter: str) -> tuple[tuple[Decimal, ...], FirstOrder]:
    physics = build.optical_model.model
    if physics.model_kind == "IDEAL_THIN_LENS":
        focal = Decimal(physics.focal_length_mm)
        with localcontext(WORK_CONTEXT):
            matrix = (ONE, ZERO, -ONE / focal, ONE)
        return matrix, FirstOrder(efl=focal, bfl=focal, ffl=focal, front_principal_z=ZERO, rear_principal_z=ZERO, first_z=ZERO, last_z=ZERO)
    if any(s.cylinder_axis is not None for s in physics.surfaces):
        raise invalid("the lenses of a compound unit are rotationally symmetric (no cylinder_axis)", parameter)
    if wavelength is None:
        raise invalid("design_wavelength_nm is required to space refracting lenses", "design_wavelength_nm")
    materials = {m.material_id: m for m in build.model_materials}

    def index(medium: str) -> Decimal:
        # (relative to the design air, as ``ResolvedMaterial.index``: the air itself is 1, a VACUUM gap 1 / n_air)
        if medium == "AIR":
            return ONE
        if medium == "VACUUM":
            return ONE / design_air_index(wavelength)
        return design_index(materials[medium], materials[medium].dispersion, wavelength)

    media = ["AIR", *physics.media, "AIR"]
    try:
        with localcontext(WORK_CONTEXT):
            surfaces = [ParaxialSurface(Decimal(s.vertex_z_mm), ONE / Decimal(s.radius_mm) if s.radius_mm else ZERO, index(media[k]),
                                        index(media[k + 1])) for k, s in enumerate(physics.surfaces)]
    except ValueError as exc:
        raise invalid(str(exc), "design_wavelength_nm") from None
    return sequence_matrix(surfaces), sequence_first_order(surfaces)


def _extent(model) -> tuple[Decimal, Decimal, Decimal]:
    """(diameter, z_min, z_max) of a member model's body in its own frame."""

    geometry = model.design_geometry
    return Decimal(geometry.diameter_mm), Decimal(geometry.z_min_mm), Decimal(geometry.z_max_mm)  # type: ignore[union-attr]


def _wavelength(values: Mapping[str, Any], builds: list[NativeBuild]) -> Decimal | None:
    given = values.get("design_wavelength_nm")
    if given is not None:
        return given
    stated = {p.wavelength_nm for b in builds if (p := getattr(b.optical_model.model, "paraxial", None)) is not None}
    if len(stated) > 1:
        raise invalid("the lenses are designed at different wavelengths: give design_wavelength_nm", "design_wavelength_nm")
    return Decimal(stated.pop()) if stated else None


def _member(member_id: str, model, z: Decimal) -> dict[str, Any]:
    return {"member_id": member_id, "model_id": model.model_id,
            "pose": {"position_mm": [0.0, 0.0, float(quantize(z, LENGTH_QUANTUM_MM))], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}


def _port(port_id: str, z: Decimal) -> dict[str, Any]:
    return {"port_id": port_id, "role": "INPUT" if port_id == "IN" else "OUTPUT", "point_mm": ["0", "0", length(z)],
            "direction": ["0", "0", "1"], "status": FactStatus.DERIVED.value}


def _place_pair(values: Mapping[str, Any], context: BuildContext, nested: NestedBuilder, unit_id: str, first_z: Decimal | None):
    """Build both lenses and place them afocally; ``first_z`` None puts L1's front focal point at the unit origin."""

    lens1 = nested(values["lens1"], f"{unit_id}-L1", context, "lens1")
    lens2 = nested(values["lens2"], f"{unit_id}-L2", context, "lens2")
    wavelength = _wavelength(values, [lens1, lens2])
    m1, order1 = _matrix_and_order(lens1, wavelength, "lens1")
    m2, order2 = _matrix_and_order(lens2, wavelength, "lens2")
    with localcontext(WORK_CONTEXT):
        # member positions as the unit places them (``_member``: 1e-6 mm), so every reported matrix is the placed unit's
        z1 = quantize(order1.ffl - order1.first_z, LENGTH_QUANTUM_MM) if first_z is None else first_z
        focus = z1 + order1.rear_focus_z
        z2 = quantize(focus + order2.ffl - order2.first_z, LENGTH_QUANTUM_MM)
        d1, lo1, hi1 = _extent(lens1.optical_model)
        d2, lo2, hi2 = _extent(lens2.optical_model)
        if z1 + hi1 > z2 + lo2:
            raise infeasible("the two lenses would overlap at the afocal spacing: use longer focal lengths or thinner lenses", "lens2")
        if first_z is None and z1 + lo1 < 0:
            raise infeasible("the first lens reaches past the object plane (its front focal length is shorter than its body)", "lens1")
        gap = (z2 + order2.first_z) - (z1 + order1.last_z)
        system = multiply(m2, multiply(transfer(gap), m1))
    return {"lens1": lens1, "lens2": lens2, "wavelength": wavelength, "order1": order1, "order2": order2, "z1": z1, "z2": z2,
            "focus": focus, "system": system, "extents": ((d1, z1 + lo1, z1 + hi1), (d2, z2 + lo2, z2 + hi2))}


def _stop_member(values: Mapping[str, Any], unit_id: str, z: Decimal, lens_reach: Decimal, prefix: str, placed):
    """The stop model at the internal focus (None when no stop is asked for)."""

    opening = values.get(f"{prefix}_diameter_mm")
    obstruction = values.get(f"{prefix.replace('stop', 'obstruction')}_diameter_mm") if prefix == "fourier_stop" else None
    if opening is None and obstruction is None:
        return None
    thickness = values.get(f"{prefix}_thickness_mm") or COMPOUND_STOP_THICKNESS_MM
    (_d1, _lo1, hi1), (_d2, lo2, _hi2) = placed["extents"]
    if not (hi1 <= z and z + thickness <= lo2):
        raise infeasible("the stop does not fit between the lenses at the internal focus", f"{prefix}_diameter_mm")
    plate = lens_reach
    if opening is not None and obstruction is not None:
        if not obstruction < opening:
            raise invalid("the central obstruction must be smaller than the opening", "fourier_obstruction_diameter_mm")
        kwargs = {"opening_radius": opening / TWO, "obstruction_radius": obstruction / TWO, "edge_radius": max(plate, opening) / TWO}
    elif opening is not None:
        kwargs = {"opening_radius": opening / TWO, "edge_radius": max(plate, opening) / TWO}
    else:
        carrier = max(plate, TWO * obstruction)  # type: ignore[operator]
        kwargs = {"opening_radius": carrier / TWO, "obstruction_radius": obstruction / TWO, "edge_radius": carrier / TWO}  # type: ignore[operator]
    return stop_model(f"{unit_id}-STOP", "Stop at the internal focus", thickness=thickness, **kwargs)


def _unit(class_id: str, values: Mapping[str, Any], unit_id: str, placed, stop, stop_z: Decimal | None,
          ports: list[dict[str, Any]], name: str, design: dict[str, Any]) -> NativeBuild:
    lens1, lens2 = placed["lens1"], placed["lens2"]
    members = [_member("L1", lens1.optical_model, placed["z1"]), _member("L2", lens2.optical_model, placed["z2"])]
    extents = list(placed["extents"])
    member_models = [lens1.optical_model, lens2.optical_model]
    if stop is not None:
        members.insert(1, _member("STOP", stop, stop_z))  # type: ignore[arg-type]
        d, lo, hi = _extent(stop)
        extents.append((d, stop_z + lo, stop_z + hi))  # type: ignore[operator]
        member_models.insert(1, stop)
    envelope = cylinder(max(e[0] for e in extents), min(e[1] for e in extents), max(e[2] for e in extents))
    physics = {"model_kind": "COMPOUND", "model_version": "1", "members": members, "envelope": envelope, "ports": ports}
    model = optical_model(unit_id, values.get("display_name") or name, physics)
    return finish(class_id, model, members=tuple(member_models), materials=[*lens1.model_materials, *lens2.model_materials],
                  coatings=[*lens1.model_coatings, *lens2.model_coatings], role="other_optical", values=values, design=design)


def _focal(order: FirstOrder) -> str:
    return length(order.efl)


def _unit_id(values: Mapping[str, Any], prefix: str, f1: Decimal, f2: Decimal) -> str:
    unit_id = values.get("model_id") or default_model_id(prefix, "f", quantize(f1, Decimal("0.001")), quantize(f2, Decimal("0.001")))
    return unit_id[:MAX_UNIT_ID]


def _lens_focal(spec: Mapping[str, Any]) -> Decimal:
    """The focal length a nested lens asks for (for the default unit id; the nested build checks it)."""

    value = to_decimal(spec.get("parameters", {}).get("focal_length_mm"))
    return ZERO if value is None else value


def build_telescope(values: Mapping[str, Any], context: BuildContext, nested: NestedBuilder, *, keplerian: bool) -> NativeBuild:
    class_id = "compound.keplerian_telescope" if keplerian else "compound.galilean_telescope"
    unit_id = _unit_id(values, "keplerian" if keplerian else "galilean", _lens_focal(values["lens1"]), _lens_focal(values["lens2"]))
    placed = _place_pair(values, context, nested, unit_id, ZERO)
    f1, f2 = placed["order1"].efl, placed["order2"].efl
    if keplerian and not (f1 > 0 and f2 > 0):
        raise invalid("a Keplerian telescope has two positive lenses", "lens2")
    if not keplerian and not f1 * f2 < 0:
        raise invalid("a Galilean telescope has one positive and one negative lens", "lens2")
    stop = None
    if keplerian:
        reach = max(placed["extents"][0][0], placed["extents"][1][0])
        stop = _stop_member(values, unit_id, placed["focus"], reach, "focal_stop", placed)
    elif values.get("focal_stop_diameter_mm") is not None:
        raise invalid("a Galilean telescope has no internal focus for a stop", "focal_stop_diameter_mm")
    with localcontext(WORK_CONTEXT):
        magnification = -f2 / f1
    a, _b, c, d = placed["system"]
    (_d1, start, _e1), (_d2, _s2, end) = placed["extents"]
    design = {
        "focal_lengths_mm": [_focal(placed["order1"]), _focal(placed["order2"])],
        "design_wavelength_nm": text(placed["wavelength"]) if placed["wavelength"] is not None else None,
        "lens_vertex_positions_mm": [length(placed["z1"]), length(placed["z2"])],
        "beam_magnification": text(magnification, Decimal("0.000000001")),
        "system_matrix_A_D_C": [text(a, Decimal("1e-12")), text(d, Decimal("1e-12")), text(c, Decimal("1e-15"))],
        "equations": "afocal spacing z2 = z1 + last1 + BFL1 + FFL2; beam magnification -f2/f1 = A of the system matrix",
    }
    if stop is not None:
        design["focal_stop_z_mm"] = length(placed["focus"])
    kind = "Keplerian" if keplerian else "Galilean"
    name = f"{kind} telescope f1 = {_focal(placed['order1'])} mm, f2 = {_focal(placed['order2'])} mm"
    return _unit(class_id, values, unit_id, placed, stop, placed["focus"], [_port("IN", start), _port("OUT", end)], name, design)


def build_relay(values: Mapping[str, Any], context: BuildContext, nested: NestedBuilder) -> NativeBuild:
    unit_id = _unit_id(values, "relay4f", _lens_focal(values["lens1"]), _lens_focal(values["lens2"]))
    placed = _place_pair(values, context, nested, unit_id, None)
    order1, order2 = placed["order1"], placed["order2"]
    if not (order1.efl > 0 and order2.efl > 0):
        raise invalid("a 4f relay has two positive lenses", "lens2")
    reach = max(placed["extents"][0][0], placed["extents"][1][0])
    stop = _stop_member(values, unit_id, placed["focus"], reach, "fourier_stop", placed)
    with localcontext(WORK_CONTEXT):
        image_z = quantize(placed["z2"] + order2.rear_focus_z, LENGTH_QUANTUM_MM)  # the OUT port as written
        magnification = -order2.efl / order1.efl
        to_image = multiply(transfer(image_z - placed["z2"] - order2.last_z), placed["system"])
        full = multiply(to_image, transfer(placed["z1"] + order1.first_z))
    a, b, c, _d = full
    design = {
        "focal_lengths_mm": [_focal(order1), _focal(order2)],
        "design_wavelength_nm": text(placed["wavelength"]) if placed["wavelength"] is not None else None,
        "lens_vertex_positions_mm": [length(placed["z1"]), length(placed["z2"])], "fourier_plane_z_mm": length(placed["focus"]),
        "image_plane_z_mm": length(image_z), "lateral_magnification": text(magnification, Decimal("0.000000001")),
        "object_to_image_matrix_A_B_C": [text(a, Decimal("1e-12")), text(b, Decimal("1e-12")), text(c, Decimal("1e-15"))],
        "equations": "object plane at L1's front focal point; afocal spacing; image at L2's rear focal point; m = -f2/f1",
    }
    name = f"4f relay f1 = {_focal(order1)} mm, f2 = {_focal(order2)} mm"
    return _unit("compound.relay_4f", values, unit_id, placed, stop, placed["focus"], [_port("IN", ZERO), _port("OUT", image_z)], name, design)


__all__ = ["COMPOUND_STOP_THICKNESS_MM", "build_relay", "build_telescope"]
