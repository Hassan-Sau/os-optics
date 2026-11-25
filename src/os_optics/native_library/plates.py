"""Native plates, mirrors and plate splitters (``SURFACE_SEQUENCE`` models).

* ``plate.window``: a plane-parallel plate of one material (a window, a substrate, a sample slab of declared index and
  absorption); refraction, dispersion, bulk absorption and the bare Fresnel loss of uncoated faces are traced.
* ``plate.layered``: plane-parallel layers of different materials in contact (a cuvette: wall / liquid / wall, a cemented
  plate stack); every interface refracts between its two media.
* ``mirror.flat`` / ``mirror.curved``: one reflecting surface with a declared reflectance (a curved mirror of focal
  length ``f`` has vertex radius ``R = 2 f``; ``f > 0`` concave, ``f < 0`` convex; optional conic constant, ``-1`` a
  paraboloid, which images an on-axis point at infinity perfectly). The substrate behind the face is drawn, not traced;
  its thickness is measured at the vertex. A convex face falls behind its vertex towards the edge: by default the
  substrate is that much thicker (its edge keeps the default thickness), and a given ``substrate_thickness_mm`` that does
  not exceed that depth is refused (``NATIVE_DESIGN_INFEASIBLE``).
* ``splitter.plate``: one coated plane that reflects ``R`` and transmits ``T`` of the power (optionally per s / p
  polarization); its substrate is not modelled (an ideal thin splitter).

Declared coatings are constant over their band; their angle-of-incidence range defaults to 0-90 deg (the same values at
every angle). A metallic mirror transmits nothing (``T = 0``) and a dielectric mirror or a splitter is lossless
(``T = 1 - R``) unless a transmittance is given.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from .._vendor.shared_contracts.design_catalog import CoatingKind, FactStatus, ModelFidelity, SimulationApproximation
from .common import (
    DEFAULT_CLEAR_FRACTION,
    THIN_ELEMENT_THICKNESS_MM,
    BuildContext,
    NativeBuild,
    aoi_range,
    band,
    box,
    cylinder,
    declared_coating,
    default_model_id,
    face_frame,
    finish,
    optical_model,
    resolve_ar_coating,
    transmissive_frame,
)
from .errors import infeasible, invalid
from .lenses import coating_approximations, model_band, radius_text, refractive_approximations, written_curvature
from .numbers import LENGTH_QUANTUM_MM, ONE, TWO, ZERO, ceil_to, length, sag, text

#: Drawn substrate thickness of a mirror when none is given: a quarter of its diameter, at least 1 mm, rounded up to 0.1 mm.
MIRROR_THICKNESS_FRACTION = Decimal("0.25")
MIRROR_THICKNESS_QUANTUM_MM = Decimal("0.1")
MIRROR_KINDS = ("METALLIC_REFLECTOR", "DIELECTRIC_HIGH_REFLECTOR")


def _outline(values: Mapping[str, Any]) -> dict[str, Any]:
    """Round (``diameter_mm``) or rectangular (``width_mm`` x ``height_mm``) outline and its clear aperture."""

    diameter, width, height = values.get("diameter_mm"), values.get("width_mm"), values.get("height_mm")
    fraction = values.get("clear_aperture_fraction") or DEFAULT_CLEAR_FRACTION
    if (diameter is None) == (width is None and height is None) or (width is None) != (height is None):
        raise invalid("give either diameter_mm (a round plate) or both width_mm and height_mm (a rectangular plate)", "diameter_mm")
    if diameter is not None:
        return {"round": True, "edge": {"edge_radius_mm": text(diameter / TWO)}, "clear": {"clear_radius_mm": text(fraction * diameter / TWO)},
                "diameter": diameter}
    return {"round": False, "edge": {"edge_half_widths_mm": [text(width / TWO), text(height / TWO)]},
            "clear": {"clear_half_widths_mm": [text(fraction * width / TWO), text(fraction * height / TWO)]}, "width": width, "height": height}


def _body(outline: dict[str, Any], z_min: Decimal, z_max: Decimal):
    if outline["round"]:
        return cylinder(outline["diameter"], z_min, z_max)
    return box(outline["width"] / TWO, outline["height"] / TWO, z_min, z_max)


def _stack(class_id: str, values: Mapping[str, Any], context: BuildContext, layers: list[tuple[Any, Decimal]], default_id: str,
           default_name: str) -> NativeBuild:
    outline = _outline(values)
    materials = [context.resolver.resolve(spec, "material" if class_id == "plate.window" else f"layers[{k}].material")
                 for k, (spec, _t) in enumerate(layers)]
    coating_spec = values.get("ar_coating")
    low, high = model_band(values, materials, None, ())
    coating = resolve_ar_coating(coating_spec, low, high, context, "ar_coating") if coating_spec is not None else None
    if coating is not None:
        low, high = model_band(values, materials, coating, ())
    surfaces = []
    z = ZERO
    for k in range(len(layers) + 1):
        surface: dict[str, Any] = {"surface_id": f"S{k + 1}", "vertex_z_mm": text(z), **outline["clear"], "traced": "TRANSMITTED",
                                   "status": FactStatus.DESIGN_BASIS.value}
        if coating is not None and k in (0, len(layers)):
            surface["coating_id"] = coating.coating_id
        surfaces.append(surface)
        if k < len(layers):
            z += layers[k][1]
    physics = {
        "model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value,
        "surfaces": surfaces, "media": [m.material_id for m in materials], **outline["edge"],
        "valid_band_nm": band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS),
        "approximations": refractive_approximations(materials, [coating] if coating else []), "status": FactStatus.DESIGN_BASIS.value,
    }
    model = optical_model(values.get("model_id") or default_id, values.get("display_name") or default_name, physics,
                          frame=transmissive_frame(outline["round"]), geometry=_body(outline, ZERO, z))
    design = {"total_thickness_mm": text(z), "layers": [{"material_id": m.material_id, "thickness_mm": text(t)} for m, (_s, t) in zip(materials, layers)]}
    return finish(class_id, model, materials=[m.material for m in materials], coatings=[coating] if coating else [], role="window",
                  values=values, design=design)


def build_window(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    thickness = values["thickness_mm"]
    spec = values["material"]
    label = spec if isinstance(spec, str) else spec.get("catalog") or spec.get("material_id") or "declared"
    return _stack("plate.window", values, context, [(spec, thickness)], default_model_id("window", str(label), "t", thickness),
                  f"Window {label} t = {text(thickness)} mm")


def build_layered(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    layers = [(layer["material"], layer["thickness_mm"]) for layer in values["layers"]]
    total = sum((t for _m, t in layers), ZERO)
    return _stack("plate.layered", values, context, layers, default_model_id("layered-plate", len(layers), "t", total),
                  f"Layered plate ({len(layers)} layers, {text(total)} mm)")


# ----------------------------------------------------------------------------------------------- reflectors


def _reflectance(values: Mapping[str, Any], kind: CoatingKind) -> tuple[Decimal, Decimal, tuple[Decimal, ...] | None, bool]:
    """(R, T, (R_s, R_p, T_s, T_p) or None, lossless default used)."""

    r, r_s, r_p = values.get("reflectance"), values.get("reflectance_s"), values.get("reflectance_p")
    t, t_s, t_p = values.get("transmittance"), values.get("transmittance_s"), values.get("transmittance_p")
    resolved = (r_s, r_p, t_s, t_p)
    if any(v is not None for v in resolved):
        if r_s is None or r_p is None:
            raise invalid("a polarization-resolved coating states reflectance_s and reflectance_p", "reflectance_s")
        if r is not None or t is not None:
            raise invalid("give reflectance / transmittance or their s / p values, not both", "reflectance")
        if (t_s is None) != (t_p is None):
            raise invalid("give both transmittance_s and transmittance_p, or neither", "transmittance_s")
        lossless = t_s is None
        if lossless:
            t_s, t_p = (ZERO, ZERO) if kind is CoatingKind.METALLIC_REFLECTOR else (ONE - r_s, ONE - r_p)
        if r_s + t_s > ONE or r_p + t_p > ONE:  # type: ignore[operator]
            raise invalid("reflectance + transmittance cannot exceed 1 for either polarization", "transmittance_s")
        return (r_s + r_p) / TWO, (t_s + t_p) / TWO, (r_s, r_p, t_s, t_p), lossless  # type: ignore[operator]
    if r is None:
        raise invalid("required: the reflectance (or reflectance_s and reflectance_p)", "reflectance")
    lossless = t is None
    if lossless:
        t = ZERO if kind is CoatingKind.METALLIC_REFLECTOR else ONE - r
    if r + t > ONE:
        raise invalid("reflectance + transmittance cannot exceed 1", "transmittance")
    return r, t, None, lossless


def _mirror(class_id: str, values: Mapping[str, Any], context: BuildContext, curvature: Decimal, conic: Decimal) -> NativeBuild:
    diameter = values["diameter_mm"]
    clear = values.get("clear_aperture_diameter_mm")
    clear = DEFAULT_CLEAR_FRACTION * diameter if clear is None else clear
    if clear > diameter:
        raise invalid("the clear aperture cannot exceed the outer diameter", "clear_aperture_diameter_mm")
    kind = CoatingKind(values.get("coating_kind") or "METALLIC_REFLECTOR")
    r, t, polarized, lossless = _reflectance(values, kind)
    low, high = values["valid_band_nm"]
    coating = declared_coating(kind, low, high, r, t, polarized=polarized, aoi=aoi_range(values.get("valid_aoi_deg"), "valid_aoi_deg"),
                               basis="Declared mirror coating" + (" (transmittance not stated: taken as 0 for a metal film, 1 - R "
                                                                  "for a dielectric stack)" if lossless else ""),
                               parameter="reflectance")
    edge_radius = diameter / TWO
    written = written_curvature(curvature)
    edge_sag = sag(written, conic, edge_radius)
    if edge_sag is None or (written != 0 and ONE - (ONE + conic) * written * written * edge_radius * edge_radius <= 0):
        raise infeasible("the mirror surface is not defined over its outer diameter: use a longer focal length or a smaller diameter",
                         "focal_length_mm")
    surface: dict[str, Any] = {"surface_id": "S1", "vertex_z_mm": "0", "clear_radius_mm": text(clear / TWO), "coating_id": coating.coating_id,
                               "traced": "REFLECTED", "status": (FactStatus.DERIVED if curvature else FactStatus.DESIGN_BASIS).value}
    radius = radius_text(curvature)
    if radius is not None:
        surface["radius_mm"] = radius
        if conic != 0:
            surface["conic"] = text(conic)
    approximations = {SimulationApproximation.EDGE_AS_OPAQUE_RIM, SimulationApproximation.IDEAL_SURFACE_FIGURE,
                      SimulationApproximation.TRANSMITTED_LIGHT_NOT_TRACED} | coating_approximations([coating])
    physics = {"model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SPECIFICATION_ENVELOPE.value,
               "surfaces": [surface], "media": [], "edge_radius_mm": text(edge_radius),
               "valid_band_nm": band(low, high, FactStatus.DESIGN_BASIS), "approximations": sorted(a.value for a in approximations),
               "status": FactStatus.DESIGN_BASIS.value}
    # The substrate thickness is measured at the vertex (z = 0 to the back face at -thickness). A convex face falls behind
    # its vertex towards the edge (edge_sag < 0), so the substrate must be thicker than that depth or the face would leave
    # the body (no material at the edge): the default adds the depth, a given thickness must exceed it.
    depth = max(ZERO, -edge_sag)
    thickness = values.get("substrate_thickness_mm")
    if thickness is None:
        thickness = max(ONE, ceil_to(MIRROR_THICKNESS_FRACTION * diameter, MIRROR_THICKNESS_QUANTUM_MM))
        if depth > 0:
            thickness = ceil_to(thickness + depth, MIRROR_THICKNESS_QUANTUM_MM)
    elif thickness <= depth:
        least = ceil_to(depth, LENGTH_QUANTUM_MM)  # the smallest stated thickness that is accepted
        least = least + LENGTH_QUANTUM_MM if least == depth else least
        raise infeasible(f"the convex face lies {length(depth)} mm behind its vertex at the outer diameter, so a substrate "
                         f"{text(thickness)} mm thick at the vertex leaves no material at the edge: give substrate_thickness_mm of at "
                         f"least {text(least)} mm (or omit it to have it derived)", "substrate_thickness_mm")
    focal = values.get("focal_length_mm")
    model_id = values.get("model_id") or (default_model_id("mirror-flat", "d", diameter) if focal is None
                                          else default_model_id("mirror", "f", focal, "d", diameter))
    name = values.get("display_name") or ("Flat mirror" if focal is None else f"{'Concave' if focal > 0 else 'Convex'} mirror f = {text(focal)} mm")
    model = optical_model(model_id, name, physics, frame=face_frame(), geometry=cylinder(diameter, -thickness, max(ZERO, edge_sag)))
    design = {"reflectance": text(r), "transmittance": text(t), "coating_id": coating.coating_id, "drawn_substrate_thickness_mm": text(thickness)}
    if focal is not None:
        design.update(focal_length_mm=text(focal), radius_mm=radius, conic=text(conic), edge_sag_mm=text(edge_sag))
    return finish(class_id, model, coatings=[coating], role="mirror", values=values, design=design)


def build_flat_mirror(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    return _mirror("mirror.flat", values, context, ZERO, ZERO)


def build_curved_mirror(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    focal = values["focal_length_mm"]
    conic = values.get("conic") or ZERO
    return _mirror("mirror.curved", values, context, ONE / (TWO * focal), conic)


def build_plate_splitter(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    diameter = values["diameter_mm"]
    clear = values.get("clear_aperture_diameter_mm")
    clear = DEFAULT_CLEAR_FRACTION * diameter if clear is None else clear
    if clear > diameter:
        raise invalid("the clear aperture cannot exceed the outer diameter", "clear_aperture_diameter_mm")
    r, t, polarized, lossless = _reflectance(values, CoatingKind.BEAMSPLITTER)
    low, high = values["valid_band_nm"]
    coating = declared_coating(CoatingKind.BEAMSPLITTER, low, high, r, t, polarized=polarized,
                               aoi=aoi_range(values.get("valid_aoi_deg"), "valid_aoi_deg"),
                               basis="Declared beamsplitter coating" + (" (lossless: T = 1 - R)" if lossless else ""), parameter="reflectance")
    surface = {"surface_id": "S1", "vertex_z_mm": "0", "clear_radius_mm": text(clear / TWO), "coating_id": coating.coating_id,
               "traced": "TRANSMITTED_AND_REFLECTED", "status": FactStatus.DESIGN_BASIS.value}
    approximations = {SimulationApproximation.EDGE_AS_OPAQUE_RIM, SimulationApproximation.IDEAL_SURFACE_FIGURE,
                      SimulationApproximation.SUBSTRATE_NOT_MODELLED} | coating_approximations([coating])
    physics = {"model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SPECIFICATION_ENVELOPE.value,
               "surfaces": [surface], "media": [], "edge_radius_mm": text(diameter / TWO),
               "valid_band_nm": band(low, high, FactStatus.DESIGN_BASIS), "approximations": sorted(a.value for a in approximations),
               "status": FactStatus.DESIGN_BASIS.value}
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("splitter", "R", r, "d", diameter)
    model = optical_model(model_id, values.get("display_name") or f"Plate beamsplitter R = {text(r)}", physics, frame=face_frame(split=True),
                          geometry=cylinder(diameter, -thickness, ZERO))
    design = {"reflectance": text(r), "transmittance": text(t), "coating_id": coating.coating_id, "drawn_thickness_mm": text(thickness)}
    if polarized is not None:
        design["reflectance_s_p"] = [text(polarized[0]), text(polarized[1])]
        design["transmittance_s_p"] = [text(polarized[2]), text(polarized[3])]
    return finish("splitter.plate", model, coatings=[coating], role="splitter", values=values, design=design)


__all__ = ["MIRROR_KINDS", "build_curved_mirror", "build_flat_mirror", "build_layered", "build_plate_splitter", "build_window"]
