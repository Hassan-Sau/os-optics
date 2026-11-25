"""Native axicons and wedge prisms: refracting elements of one material with a cone or a tilted plane face.

``lens.axicon`` -- a ``SURFACE_SEQUENCE`` cone lens: a plane face and a cone of base angle ``alpha`` (the angle between
the cone and the plane face) with its apex on the axis. ``PLANE_FIRST`` (the usual orientation, and the public catalog's
``AXICON`` products'): plane ``S1`` at ``z = 0``, cone ``S2`` with its apex at ``z = t`` (the centre thickness) and the sag
``z = t - tan(alpha) rho`` (``linear_sag_coefficient = -tan alpha``); ``CONE_FIRST``: cone ``S1`` with its apex at ``z = 0``
pointing at the incoming light (``+tan alpha``), plane ``S2`` at ``z = t``. The edge is ``t - R tan(alpha)`` thick. A ray
parallel to the axis leaves a plane-first axicon at ``beta = asin(n sin alpha) - alpha`` toward the axis (Snell's law at
the cone; a cone-first axicon: ``beta = asin(n sin(alpha - asin(sin(alpha) / n)))``); the rays of a beam of radius ``h``
cross the axis ``h / tan(beta)`` behind it, the length of the Bessel zone (J. H. McLeod, J. Opt. Soc. Am. 44, 592 (1954)).
The model has no focal length and no paraxial block; a Gaussian beam behind the cone is not a Gaussian beam (the run
states it), while rays, power and geometric spot metrics are traced.

``prism.wedge`` -- a round wedge prism (``SURFACE_SET`` with a round convex body): a plane entrance face normal to local
+Z and an exit face tilted by the wedge angle ``alpha`` about local Y, thick end toward local -X (the public catalog's
``WEDGE_PRISM`` convention). Frame origin: the body's centroid on the axis (``BODY_CENTER``), at the height
``(t^2 + tan^2(alpha) R^2 / 4) / (2 t)`` above the entrance face (``t`` the centre thickness, ``R`` the radius). A ray along
+Z is deviated toward the thick end by ``delta = asin(n sin alpha) - alpha`` (``(n - 1) alpha`` for a thin prism; Hecht,
*Optics*, 5th ed., sec. 5.5.1); port ``OUT`` is that direction at the design wavelength, from the exit face's centre.

``lens.grin_rod`` -- a radial gradient-index rod lens (``SURFACE_SEQUENCE``: plane end faces ``S1`` at ``z = 0`` and ``S2`` at
``z = L``, the medium a declared gradient-index material, ``n(r)^2 = n0^2 (1 - (g r)^2)``). Its length is given, or follows from
the pitch ``P`` at the design wavelength (``L = 2 pi P / g``). Its paraxial matrix in air is
``((cos gL, sin gL / (n0 g)), (-n0 g sin gL, cos gL))`` (E. W. Marchand, *Gradient Index Optics*, 1978, ch. 3): the effective
focal length ``1 / (n0 g sin gL)`` and the back focal distance from the exit face ``cos(gL) / (n0 g sin gL)`` (0 at a quarter
pitch: a collimated beam focuses on the exit face).

Every value is exact decimal arithmetic (``trig.py`` for the angles).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any

from .._vendor.shared_contracts.design_catalog import (
    FactStatus,
    FrameOrigin,
    FrameReferenceAxis,
    ModelFidelity,
    OpticalFrameV1,
    OpticalSymmetry,
    SimulationApproximation,
)
from .common import DEFAULT_CLEAR_FRACTION, BuildContext, NativeBuild, band, cylinder, default_model_id, finish, optical_model, resolve_ar_coating, transmissive_frame
from .errors import infeasible, invalid
from .lenses import coating_approximations, edge_rule, model_band
from .numbers import LENGTH_QUANTUM_MM, ONE, RATIO_QUANTUM, THICKNESS_QUANTUM_MM, TWO, WORK_CONTEXT, ZERO, ceil_to, floor_to, length, text
from .trig import asin, degrees, pi, radians, sin_cos_deg, tan_deg

AXICON_ORIENTATIONS = ("PLANE_FIRST", "CONE_FIRST")
FOUR = Decimal(4)
#: Resolution of angles written in design reports (degrees).
ANGLE_QUANTUM_DEG = Decimal("0.000000001")


def _outline(values: Mapping[str, Any]) -> tuple[Decimal, Decimal, Decimal]:
    diameter = values["diameter_mm"]
    clear = values.get("clear_aperture_diameter_mm")
    clear = DEFAULT_CLEAR_FRACTION * diameter if clear is None else clear
    if clear > diameter:
        raise invalid("the clear aperture cannot exceed the outer diameter", "clear_aperture_diameter_mm")
    return diameter, diameter / TWO, clear / TWO


def _thickness(values: Mapping[str, Any], diameter: Decimal, drop: Decimal, what: str) -> tuple[Decimal, Decimal]:
    """(centre thickness, edge / thin-end thickness): the given centre thickness (its thin edge must be positive), else the
    thinnest multiple of 1 um whose thin edge is at least ``max(0.5 mm, 0.05 D)``."""

    given = values.get("center_thickness_mm")
    if given is not None:
        if given - drop <= 0:
            raise infeasible(f"the {what} would be {length(given - drop)} mm thick: give a centre thickness above {length(drop)} mm "
                             "(or omit it to have it derived)", "center_thickness_mm")
        return given, given - drop
    thickness = ceil_to(edge_rule(diameter) + drop, THICKNESS_QUANTUM_MM)
    return thickness, thickness - drop


def _index(material, wavelength: Decimal | None, parameter: str = "design_wavelength_nm") -> Decimal | None:
    if wavelength is None:
        return None
    try:
        return material.index(wavelength)
    except ValueError as exc:
        raise invalid(str(exc), parameter) from None


# ---------------------------------------------------------------------------------------------------------- axicon


def axicon_deflection_deg(alpha_deg: Decimal, index: Decimal, orientation: str) -> Decimal:
    """Angle (deg) to the axis of a ray that entered parallel to it; ``ValueError`` under total internal reflection."""

    with localcontext(WORK_CONTEXT):
        alpha = radians(alpha_deg)
        sin_a, _ = sin_cos_deg(alpha_deg)
        if orientation == "PLANE_FIRST":
            inside = index * sin_a
            if inside >= ONE:
                raise ValueError("the light is totally reflected at the cone")
            return degrees(asin(inside) - alpha)
        inner = alpha - asin(sin_a / index)  # the ray's angle to the axis inside the glass
        s, _ = sin_cos_deg(degrees(inner))
        if index * s >= ONE:
            raise ValueError("the light is totally reflected at the plane exit face")
        return degrees(asin(index * s))


def build_axicon(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    alpha = values["cone_angle_deg"]
    orientation = values.get("orientation") or "PLANE_FIRST"
    material = context.resolver.resolve(values["material"], "material")
    diameter, edge_radius, clear_radius = _outline(values)
    with localcontext(WORK_CONTEXT):
        slope = tan_deg(alpha)
        drop = edge_radius * slope
    thickness, edge = _thickness(values, diameter, drop, "edge")
    wavelength = values.get("design_wavelength_nm")
    low, high = model_band(values, [material], None, (wavelength,) if wavelength is not None else ())
    coating_spec = values.get("ar_coating")
    coating = resolve_ar_coating(coating_spec, low, high, context, "ar_coating") if coating_spec is not None else None
    if coating is not None:
        low, high = model_band(values, [material], coating, (wavelength,) if wavelength is not None else ())
    coefficient = text(-slope if orientation == "PLANE_FIRST" else slope, RATIO_QUANTUM)
    surfaces = []
    for sid, z, cone in (("S1", ZERO, orientation == "CONE_FIRST"), ("S2", thickness, orientation == "PLANE_FIRST")):
        surface: dict[str, Any] = {"surface_id": sid, "vertex_z_mm": text(z), "clear_radius_mm": text(clear_radius), "traced": "TRANSMITTED",
                                   "status": (FactStatus.DERIVED if cone else FactStatus.DESIGN_BASIS).value}
        if cone:
            surface["linear_sag_coefficient"] = coefficient
        if coating is not None:
            surface["coating_id"] = coating.coating_id
        surfaces.append(surface)
    approximations = {SimulationApproximation.NOMINAL_PRESCRIPTION, SimulationApproximation.IDEAL_SURFACE_FIGURE,
                      SimulationApproximation.EDGE_AS_OPAQUE_RIM, SimulationApproximation.GHOST_REFLECTIONS_NOT_TRACED}
    if material.from_catalog:
        approximations.add(SimulationApproximation.CATALOG_MATERIAL_DISPERSION)
    approximations |= coating_approximations([coating] if coating else [])
    physics = {
        "model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value,
        "surfaces": surfaces, "media": [material.material_id], "edge_radius_mm": text(edge_radius),
        "valid_band_nm": band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS),
        "approximations": sorted(a.value for a in approximations), "status": FactStatus.DERIVED.value,
    }
    model_id = values.get("model_id") or default_model_id("axicon", material.material_id, "a", alpha, "d", diameter)
    name = values.get("display_name") or f"Axicon {material.material_id} {text(alpha)} deg, D {text(diameter)} mm"
    model = optical_model(model_id, name, physics, frame=transmissive_frame(), geometry=cylinder(diameter, ZERO, thickness),
                          parameter="cone_angle_deg")
    design: dict[str, Any] = {
        "cone_angle_deg": text(alpha), "orientation": orientation, "linear_sag_coefficient": coefficient,
        "center_thickness_mm": text(thickness), "edge_thickness_mm": length(edge),
        "equations": "cone sag z = z0 -/+ tan(alpha) rho; plane first: beta = asin(n sin alpha) - alpha; cone first: "
                     "beta = asin(n sin(alpha - asin(sin(alpha) / n))); Bessel zone length = h / tan(beta) for a beam of radius h",
    }
    if wavelength is not None:
        index = _index(material, wavelength)
        try:
            beta = axicon_deflection_deg(alpha, index, orientation)  # type: ignore[arg-type]
        except ValueError as exc:
            raise infeasible(f"{exc} at {text(wavelength)} nm: use a smaller cone angle", "cone_angle_deg") from None
        with localcontext(WORK_CONTEXT):
            zone = clear_radius / tan_deg(beta)
        design.update(design_wavelength_nm=text(wavelength), refractive_index=text(index, RATIO_QUANTUM),  # type: ignore[arg-type]
                      deflection_deg=text(beta, ANGLE_QUANTUM_DEG), bessel_zone_length_mm=length(zone))
    return finish("lens.axicon", model, materials=[material.material], coatings=[coating] if coating else [], role="lens",
                  values=values, design=design)


# ------------------------------------------------------------------------------------------------------- GRIN rod


def build_grin_rod(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    material = context.resolver.resolve(values["material"], "material")
    gradient = material.material.gradient
    if gradient is None:
        raise invalid("a GRIN rod's material is a gradient-index material: a material document with gradient "
                      '({"kind": "RADIAL_QUADRATIC", "gradient_constant_per_mm": g} or gradient_rows)', "material")
    diameter, edge_radius, clear_radius = _outline(values)
    wavelength = values["design_wavelength_nm"]
    n0 = _index(material, wavelength)
    assert n0 is not None
    g = gradient.gradient_at(wavelength)
    given_length, pitch = values.get("length_mm"), values.get("pitch")
    if (given_length is None) == (pitch is None):
        raise invalid("state exactly one of length_mm (the rod length) and pitch (P = g L / (2 pi) at the design wavelength)", "length_mm")
    with localcontext(WORK_CONTEXT):
        rod_length = given_length if given_length is not None else ceil_to(TWO * pi() * pitch / g, THICKNESS_QUANTUM_MM)
        phase = g * rod_length
        s, c = sin_cos_deg(degrees(phase))
        rod_pitch = phase / (TWO * pi())
    low, high = model_band(values, [material], None, (wavelength,))
    coating_spec = values.get("ar_coating")
    coating = resolve_ar_coating(coating_spec, low, high, context, "ar_coating") if coating_spec is not None else None
    if coating is not None:
        low, high = model_band(values, [material], coating, (wavelength,))
    surfaces = []
    for sid, z in (("S1", ZERO), ("S2", rod_length)):
        surface: dict[str, Any] = {"surface_id": sid, "vertex_z_mm": text(z), "clear_radius_mm": text(clear_radius), "traced": "TRANSMITTED",
                                   "status": FactStatus.DESIGN_BASIS.value}
        if coating is not None:
            surface["coating_id"] = coating.coating_id
        surfaces.append(surface)
    approximations = {SimulationApproximation.NOMINAL_PRESCRIPTION, SimulationApproximation.IDEAL_SURFACE_FIGURE,
                      SimulationApproximation.EDGE_AS_OPAQUE_RIM, SimulationApproximation.GHOST_REFLECTIONS_NOT_TRACED}
    approximations |= coating_approximations([coating] if coating else [])
    physics = {
        "model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value,
        "surfaces": surfaces, "media": [material.material_id], "edge_radius_mm": text(edge_radius),
        "valid_band_nm": band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS),
        "approximations": sorted(a.value for a in approximations), "status": FactStatus.DERIVED.value,
    }
    model_id = values.get("model_id") or default_model_id("grin", material.material_id, "l", rod_length, "d", diameter)
    name = values.get("display_name") or f"GRIN rod {material.material_id} L {text(rod_length)} mm, D {text(diameter)} mm"
    model = optical_model(model_id, name, physics, frame=transmissive_frame(), geometry=cylinder(diameter, ZERO, rod_length),
                          parameter="length_mm")
    design: dict[str, Any] = {
        "design_wavelength_nm": text(wavelength), "index_on_axis": text(n0, RATIO_QUANTUM), "gradient_constant_per_mm": text(g, RATIO_QUANTUM),
        "length_mm": text(rod_length), "pitch": text(rod_pitch, RATIO_QUANTUM), "period_mm": length(TWO * pi() / g),
        "matrix": [[text(c, RATIO_QUANTUM), text(s / (n0 * g), RATIO_QUANTUM)], [text(-n0 * g * s, RATIO_QUANTUM), text(c, RATIO_QUANTUM)]],
        "equations": "n(r)^2 = n0^2 (1 - (g r)^2); pitch P = g L / (2 pi); paraxial matrix ((cos gL, sin gL / (n0 g)), (-n0 g sin gL, cos gL)); "
                     "f = 1 / (n0 g sin gL); back focal distance from the exit face cos(gL) / (n0 g sin gL)",
    }
    if s != 0:
        with localcontext(WORK_CONTEXT):
            design.update(effective_focal_length_mm=length(ONE / (n0 * g * s)), back_focal_distance_mm=length(c / (n0 * g * s)))
    return finish("lens.grin_rod", model, materials=[material.material], coatings=[coating] if coating else [], role="lens",
                  values=values, design=design)


# ---------------------------------------------------------------------------------------------------------- wedge


def _unit(vector: tuple[Decimal, Decimal, Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    with localcontext(WORK_CONTEXT):
        norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
        return vector[0] / norm, vector[1] / norm, vector[2] / norm


def _triple(vector, quantum: Decimal = RATIO_QUANTUM) -> list[str]:
    return [text(v, quantum) for v in vector]


def build_wedge(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    alpha = values["wedge_angle_deg"]
    wavelength = values["design_wavelength_nm"]
    material = context.resolver.resolve(values["material"], "material")
    diameter, radius, clear_radius = _outline(values)
    with localcontext(WORK_CONTEXT):
        tan_a = tan_deg(alpha)
        sin_a, cos_a = sin_cos_deg(alpha)
        drop = radius * tan_a  # the exit face falls by R tan(alpha) from the centre to the thin end
    thickness, thin_end = _thickness(values, diameter, drop, "thin end")
    low, high = model_band(values, [material], None, (wavelength,))
    coating_spec = values.get("ar_coating")
    coating = resolve_ar_coating(coating_spec, low, high, context, "ar_coating") if coating_spec is not None else None
    if coating is not None:
        low, high = model_band(values, [material], coating, (wavelength,))
    index = _index(material, wavelength)
    with localcontext(WORK_CONTEXT):
        sin_t = index * sin_a  # type: ignore[operator]
        if sin_t >= ONE:
            raise infeasible(f"the light is totally reflected at the exit face at {text(wavelength)} nm: use a smaller wedge angle",
                             "wedge_angle_deg")
        cos_t = (ONE - sin_t * sin_t).sqrt()
        deviation = degrees(asin(sin_t)) - alpha
        # body centroid on the axis, above the entrance face
        z_c = (thickness * thickness + tan_a * tan_a * radius * radius / FOUR) / (TWO * thickness)
        # the ray along +Z refracted at the exit face (outward normal m = (sin a, 0, cos a)): t = n d + (cos_t - n cos_a) m
        out = _unit((ZERO + (cos_t - index * cos_a) * sin_a, ZERO, index + (cos_t - index * cos_a) * cos_a))  # type: ignore[operator]
    entrance = (ZERO, ZERO, -z_c)
    exit_ = (ZERO, ZERO, thickness - z_c)
    normal = (sin_a, ZERO, cos_a)
    u_exit = (cos_a, ZERO, -sin_a)
    coated = coating.coating_id if coating is not None else None

    def posed(sid: str, origin, surface_normal, u_axis, status: FactStatus) -> dict[str, Any]:
        surface: dict[str, Any] = {"surface_id": sid, "origin_mm": _triple(origin, LENGTH_QUANTUM_MM), "normal": _triple(surface_normal),
                                   "u_axis": _triple(u_axis), "clear_radius_mm": text(clear_radius), "medium_front": "AIR",
                                   "medium_back": material.material_id, "traced": "TRANSMITTED", "status": status.value}
        if coated is not None:
            surface["coating_id"] = coated
        return surface

    surfaces = [posed("S1", entrance, (ZERO, ZERO, -ONE), (ONE, ZERO, ZERO), FactStatus.DESIGN_BASIS),
                posed("S2", exit_, normal, u_exit, FactStatus.DERIVED)]
    body = {"medium": material.material_id, "side_radius_mm": text(radius),
            "faces": [{"point_mm": _triple(exit_, LENGTH_QUANTUM_MM), "outward_normal": _triple(normal)},
                      {"point_mm": _triple(entrance, LENGTH_QUANTUM_MM), "outward_normal": ["0", "0", "-1"]}]}
    approximations = {SimulationApproximation.NOMINAL_PRESCRIPTION, SimulationApproximation.IDEAL_SURFACE_FIGURE,
                      SimulationApproximation.EDGE_AS_OPAQUE_RIM, SimulationApproximation.GHOST_REFLECTIONS_NOT_TRACED}
    if material.from_catalog:
        approximations.add(SimulationApproximation.CATALOG_MATERIAL_DISPERSION)
    approximations |= coating_approximations([coating] if coating else [])
    physics = {
        "model_kind": "SURFACE_SET", "model_version": "1", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value, "surfaces": surfaces,
        "body": body, "valid_band_nm": band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS),
        "approximations": sorted(a.value for a in approximations), "status": FactStatus.DERIVED.value,
    }
    ports = [{"port_id": "IN", "role": "INPUT", "point_mm": _triple(entrance, LENGTH_QUANTUM_MM), "direction": ["0", "0", "1"],
              "status": FactStatus.DESIGN_BASIS.value},
             {"port_id": "OUT", "role": "OUTPUT", "point_mm": _triple(exit_, LENGTH_QUANTUM_MM), "direction": _triple(out),
              "status": FactStatus.DERIVED.value}]
    frame = OpticalFrameV1.model_validate({"origin": FrameOrigin.BODY_CENTER.value, "nominal_propagation": "PORTS",
                                           "reference_axis": FrameReferenceAxis.PORTS.value, "symmetry": OpticalSymmetry.NONE.value,
                                           "ports": ports})
    with localcontext(WORK_CONTEXT):
        z_low, z_high = floor_to(-z_c, LENGTH_QUANTUM_MM), ceil_to(thickness + drop - z_c, LENGTH_QUANTUM_MM)
    model_id = values.get("model_id") or default_model_id("wedge", material.material_id, "a", alpha, "d", diameter)
    name = values.get("display_name") or f"Wedge prism {material.material_id} {text(alpha)} deg, D {text(diameter)} mm"
    model = optical_model(model_id, name, physics, frame=frame, geometry=cylinder(diameter, z_low, z_high), parameter="wedge_angle_deg")
    design = {
        "wedge_angle_deg": text(alpha), "design_wavelength_nm": text(wavelength), "refractive_index": text(index, RATIO_QUANTUM),  # type: ignore[arg-type]
        "deviation_deg": text(deviation, ANGLE_QUANTUM_DEG), "thin_prism_deviation_deg": text((index - ONE) * alpha, ANGLE_QUANTUM_DEG),  # type: ignore[operator]
        "center_thickness_mm": text(thickness), "thin_end_thickness_mm": length(thin_end), "thick_end_thickness_mm": length(thickness + drop),
        "body_center_above_entrance_mm": length(z_c), "out_direction": _triple(out),
        "equations": "deviation = asin(n sin alpha) - alpha toward the thick end (thin prism: (n - 1) alpha); centroid height "
                     "(t^2 + tan^2(alpha) R^2 / 4) / (2 t)",
    }
    return finish("prism.wedge", model, materials=[material.material], coatings=[coating] if coating else [], role="prism",
                  values=values, design=design)


__all__ = ["build_grin_rod", "AXICON_ORIENTATIONS", "axicon_deflection_deg", "build_axicon", "build_wedge"]
