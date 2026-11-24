"""Native stops: geometric apertures in an opaque plate (``APERTURE_STOP`` models).

Every stop is an opaque plate at the plane ``z = 0`` of its frame (light travels along +Z) with one opening; a ray that
reaches the plate inside the opening passes unchanged, a ray that reaches the plate's opaque part is stopped, and light
beyond the plate's outline passes by. The stop is geometric: diffraction at the opening (the Airy pattern of a pinhole,
the sinc pattern of a slit) is not modelled.

* ``aperture.circular``: a round opening (an iris, a field or aperture stop, a pinhole);
* ``aperture.slit``: a rectangular opening of width (local X) and height (local Y); rotate the component about local Z
  to turn it;
* ``aperture.annular``: a round opening with an opaque central disc (an annular pupil, a centrally obstructed aperture);
* ``aperture.central_obstruction``: an opaque disc held in a clear carrier: light outside the disc passes (a beam block,
  a dark-field / Fourier-plane DC stop, a secondary-mirror obstruction).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any

from .._vendor.shared_contracts.design_catalog import FactStatus
from .common import THIN_ELEMENT_THICKNESS_MM, BuildContext, NativeBuild, cylinder, default_model_id, finish, optical_model
from .errors import invalid
from .numbers import TWO, WORK_CONTEXT, ZERO, ceil_to, text

#: Outer diameter of a stop plate when none is given: at least this (one inch), and at least twice the opening (rounded up
#: to ``PLATE_QUANTUM_MM``).
DEFAULT_PLATE_DIAMETER_MM = Decimal("25.4")
PLATE_QUANTUM_MM = Decimal("0.1")


def _plate(values: Mapping[str, Any], reach: Decimal, parameter: str = "plate_diameter_mm") -> Decimal:
    """The plate diameter: given (at least ``reach``, the diameter about the plate centre that the opening's far edge
    needs), or the default (one inch, or twice the reach)."""

    plate = values.get(parameter)
    if plate is None:
        return max(DEFAULT_PLATE_DIAMETER_MM, ceil_to(TWO * reach, PLATE_QUANTUM_MM))
    if plate < reach:
        decentred = values.get("opening_center_mm") not in (None, (ZERO, ZERO))
        what = "the far edge of the decentred opening it holds" if decentred else "the opening it holds"
        raise invalid(f"the plate ({text(plate)} mm) must reach {what}: a diameter of at least {text(reach, Decimal('0.001'))} mm",
                      parameter)
    return plate


def _center(values: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    center = values.get("opening_center_mm")
    return (ZERO, ZERO) if center is None else center


def stop_model(model_id: str, display_name: str, *, opening_radius: Decimal | None = None, half_widths: tuple[Decimal, Decimal] | None = None,
               center: tuple[Decimal, Decimal] = (ZERO, ZERO), obstruction_radius: Decimal | None = None, edge_radius: Decimal,
               thickness: Decimal):
    physics: dict[str, Any] = {"model_kind": "APERTURE_STOP", "model_version": "1", "edge_radius_mm": text(edge_radius),
                               "plane_z_mm": "0", "status": FactStatus.DESIGN_BASIS.value}
    if opening_radius is not None:
        physics["opening_radius_mm"] = text(opening_radius)
    else:
        physics["opening_half_widths_mm"] = [text(v) for v in half_widths]  # type: ignore[union-attr]
    if center != (ZERO, ZERO):
        physics["opening_center_mm"] = [text(v) for v in center]
    if obstruction_radius is not None:
        physics["obstruction_radius_mm"] = text(obstruction_radius)
    return optical_model(model_id, display_name, physics, geometry=cylinder(TWO * edge_radius, ZERO, thickness), parameter="plate_diameter_mm")


def _finish(class_id: str, model, values, design) -> NativeBuild:
    return finish(class_id, model, role="aperture", values=values, design=design)


def build_circular(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    opening = values["opening_diameter_mm"]
    cx, cy = _center(values)
    with localcontext(WORK_CONTEXT):  # the opening's far edge lies at |centre| + its radius from the plate centre
        reach = TWO * ((cx * cx + cy * cy).sqrt() + opening / TWO)
    plate = _plate(values, reach)
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("aperture", "d", opening)
    model = stop_model(model_id, values.get("display_name") or f"Circular aperture {text(opening)} mm", opening_radius=opening / TWO,
                       center=(cx, cy), edge_radius=plate / TWO, thickness=thickness)
    design = {"opening_diameter_mm": text(opening), "plate_diameter_mm": text(plate), "drawn_thickness_mm": text(thickness),
              "transmission_rule": "geometric: a ray passes when its crossing of z = 0 lies inside the opening"}
    return _finish("aperture.circular", model, values, design)


def build_slit(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    width, height = values["slit_width_mm"], values["slit_height_mm"]
    cx, cy = _center(values)
    hx, hy = width / TWO + abs(cx), height / TWO + abs(cy)
    with localcontext(WORK_CONTEXT):
        reach = TWO * (hx * hx + hy * hy).sqrt()
    plate = _plate(values, reach)
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("slit", "w", width, "h", height)
    model = stop_model(model_id, values.get("display_name") or f"Slit {text(width)} x {text(height)} mm",
                       half_widths=(width / TWO, height / TWO), center=(cx, cy), edge_radius=plate / TWO, thickness=thickness)
    design = {"slit_width_mm": text(width), "slit_height_mm": text(height), "plate_diameter_mm": text(plate),
              "drawn_thickness_mm": text(thickness), "orientation": "width along local X, height along local Y"}
    return _finish("aperture.slit", model, values, design)


def build_annular(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    outer, inner = values["opening_diameter_mm"], values["obstruction_diameter_mm"]
    if not inner < outer:
        raise invalid("the central obstruction must be smaller than the opening", "obstruction_diameter_mm")
    plate = _plate(values, outer)
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("annulus", "d", outer, "obs", inner)
    model = stop_model(model_id, values.get("display_name") or f"Annular aperture {text(inner)}-{text(outer)} mm", opening_radius=outer / TWO,
                       obstruction_radius=inner / TWO, edge_radius=plate / TWO, thickness=thickness)
    design = {"opening_diameter_mm": text(outer), "obstruction_diameter_mm": text(inner), "plate_diameter_mm": text(plate),
              "open_area_fraction_of_opening": text((outer * outer - inner * inner) / (outer * outer)),
              "drawn_thickness_mm": text(thickness)}
    return _finish("aperture.annular", model, values, design)


def build_central_obstruction(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    disc = values["obstruction_diameter_mm"]
    carrier = values.get("carrier_diameter_mm")
    carrier = max(DEFAULT_PLATE_DIAMETER_MM, ceil_to(TWO * disc, PLATE_QUANTUM_MM)) if carrier is None else carrier
    if not disc < carrier:
        raise invalid("the carrier must be larger than the opaque disc it holds", "carrier_diameter_mm")
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("obstruction", "d", disc)
    # A clear carrier of the given diameter with an opaque centre: the opening is the whole carrier and the plate has no
    # opaque rim (edge = opening), so only the disc stops light.
    model = stop_model(model_id, values.get("display_name") or f"Central obstruction {text(disc)} mm", opening_radius=carrier / TWO,
                       obstruction_radius=disc / TWO, edge_radius=carrier / TWO, thickness=thickness)
    design = {"obstruction_diameter_mm": text(disc), "carrier_diameter_mm": text(carrier), "drawn_thickness_mm": text(thickness),
              "transmission_rule": "geometric: a ray is stopped when its crossing of z = 0 lies on the disc"}
    return _finish("aperture.central_obstruction", model, values, design)


__all__ = ["DEFAULT_PLATE_DIAMETER_MM", "build_annular", "build_central_obstruction", "build_circular", "build_slit", "stop_model"]
