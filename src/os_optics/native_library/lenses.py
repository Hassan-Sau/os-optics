"""Native lenses: the ideal thin lens, the parametric singlet and the cemented achromatic doublet.

``lens.ideal_thin`` -- an ``IDEAL_THIN_LENS`` model: a perfect paraxial lens of the given focal length at one plane.

``lens.singlet`` -- a ``SURFACE_SEQUENCE`` lens of one material designed for a focal length at a design wavelength.
With ``n`` the index at the design wavelength, ``K = n - 1``, the shape (Coddington) factor ``X = (c1 + c2) / (c1 - c2)``
(``c = 1 / R``) and ``a = c1 - c2``, the thick-lens power (lensmaker's equation)

    phi = 1 / f = K (c1 - c2) + K^2 t c1 c2 / n,      c1 = a (X + 1) / 2,  c2 = a (X - 1) / 2

is a quadratic in ``a``: ``A a^2 + K a - phi = 0`` with ``A = K^2 t (X^2 - 1) / (4 n)``, solved by the root that tends to
the thin-lens ``a = phi / K`` as ``t -> 0``: ``a = 2 phi / (K + sqrt(K^2 + 4 A phi))``. Shapes: ``X = 0`` symmetric
(equi-convex / equi-concave), ``X = +1`` plano on the exit side (curved face toward the incoming light), ``X = -1`` plano
on the entry side, ``BEST_FORM``: ``X = 2 (n^2 - 1) / (n + 2)``, the third-order minimum of spherical aberration for an
object at infinity (thin-lens theory; Kingslake & Johnson, *Lens Design Fundamentals*, 2nd ed., sec. 4.3; Hecht,
*Optics*, 5th ed., sec. 6.3). ``rear_conic = STIGMATIC_COLLIMATED`` (plano entry side only) makes the exit surface the
hyperboloid ``k = -n^2``, which focuses a collimated on-axis beam inside the glass to a perfect geometric point (the
classical Cartesian refracting conic, eccentricity ``n``; Born & Wolf, *Principles of Optics*, sec. 4.2.3).

``lens.achromat`` -- a cemented doublet (crown then flint, ``SURFACE_SEQUENCE`` of three surfaces). Thin-lens design with
Abbe numbers ``V_i = (n_i(l_c) - 1) / (n_i(l_s) - n_i(l_l))`` at the three design wavelengths (short, centre, long):

    phi_1 = phi V_1 / (V_1 - V_2),   phi_2 = -phi V_2 / (V_1 - V_2)       (achromatic condition phi_1/V_1 + phi_2/V_2 = 0)

the crown shaped by its factor ``X_1``, the flint sharing the cemented radius. The lens is then thickened (rule below)
and its two element powers re-solved exactly on the thick prescription (Newton iteration in decimal arithmetic) so that
the paraxial EFL at the centre wavelength is ``f`` and the back focal lengths (last vertex to focus) at the short and
long wavelengths are equal: a collimated beam on the axis focuses at one point for both wavelengths. This is the axial
achromat condition of a focusing doublet (zero longitudinal chromatic aberration between the two wavelengths for an
object at infinity). For a thick lens it is not the same as equal EFLs: the principal plane moves with wavelength, so
equal EFLs leave the F and C foci 10-130 um apart for an f = 100 mm, 25.4 mm doublet, while equal BFLs leave a small EFL
difference instead (a chromatic difference of image size, reported as ``efl_difference_long_minus_short_mm``). Both
conditions reduce to the thin-lens equations above as the thicknesses tend to zero (Kingslake & Johnson, sec. 5.1 and
ch. 6; Smith, *Modern Optical Engineering*, 4th ed., sec. 13.2).

Thickening rule (both refracting classes): an element whose centre thickness is not given gets the smallest multiple
of 1 um at which both its centre thickness is at least ``min_center_thickness_mm`` (default ``max(1 mm, 0.08 D)``)
and its edge thickness at the outer diameter is at least ``min_edge_thickness_mm`` (default ``max(0.5 mm, 0.05 D)``);
the radii are re-solved for every trial thickness. A given centre thickness is kept: its edge must then be positive,
and at least ``min_edge_thickness_mm`` only when that is given too. Every surface must be defined over the whole
outline (``|R|`` larger than the outer radius).

Radii are written to 1e-6 mm; the model's ``paraxial`` block is recomputed from the written prescription.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any

from .._vendor.shared_contracts.design_catalog import FactStatus, ModelFidelity, SimulationApproximation
from .common import (
    DEFAULT_CLEAR_FRACTION,
    THIN_ELEMENT_THICKNESS_MM,
    BuildContext,
    NativeBuild,
    band,
    cylinder,
    default_model_id,
    finish,
    optical_model,
    resolve_ar_coating,
    transmissive_frame,
)
from .errors import infeasible, invalid
from .materials import ResolvedMaterial, band_intersection
from .numbers import (
    ONE,
    RATIO_QUANTUM,
    THICKNESS_QUANTUM_MM,
    TWO,
    WORK_CONTEXT,
    ZERO,
    ceil_to,
    length,
    quantize,
    sag,
    text,
)
from .paraxial import ParaxialSurface, sequence_first_order, sequence_matrix

EDGE_RULE_FRACTION = Decimal("0.05")
EDGE_RULE_MINIMUM_MM = Decimal("0.5")
CENTER_RULE_FRACTION = Decimal("0.08")
CENTER_RULE_MINIMUM_MM = Decimal(1)
#: Fraunhofer F, d and C lines (nm): the default short / centre / long design wavelengths of an achromat.
FRAUNHOFER_FDC_NM = (Decimal("486.1327"), Decimal("587.5618"), Decimal("656.2725"))
MAX_THICKENING_STEPS = 200
MAX_NEWTON_STEPS = 60
#: Step halvings of one damped Newton step of the achromat solve.
MAX_STEP_HALVINGS = 40
#: Convergence of the exact achromat solve: residual powers below this fraction of the lens power.
NEWTON_RELATIVE_TOLERANCE = Decimal("1e-30")
#: Relative step of the finite-difference Jacobian of the achromat solve.
JACOBIAN_RELATIVE_STEP = Decimal("1e-15")
THICKNESS_CONVERGENCE_MM = Decimal("1e-12")
SHAPE_FORMS = ("SYMMETRIC", "PLANO_EXIT", "PLANO_ENTRY", "BEST_FORM")
REAR_CONICS = ("NONE", "STIGMATIC_COLLIMATED")
FOUR = Decimal(4)


def edge_rule(diameter: Decimal) -> Decimal:
    return max(EDGE_RULE_MINIMUM_MM, EDGE_RULE_FRACTION * diameter)


def edge_ok(edge: Decimal, minimum: Decimal | None) -> bool:
    """An edge thickness meets ``minimum`` (``None``: a given centre thickness without a stated minimum -- the edge only
    has to be positive)."""

    return edge > 0 if minimum is None else edge >= minimum


def center_rule(diameter: Decimal) -> Decimal:
    return max(CENTER_RULE_MINIMUM_MM, CENTER_RULE_FRACTION * diameter)


def radius_text(curvature: Decimal) -> str | None:
    """The written radius of a curvature (None for a plane), to 1e-6 mm."""

    if curvature == 0:
        return None
    with localcontext(WORK_CONTEXT):
        return length(ONE / curvature)


def written_curvature(curvature: Decimal) -> Decimal:
    radius = radius_text(curvature)
    if radius is None:
        return ZERO
    with localcontext(WORK_CONTEXT):
        return ONE / Decimal(radius)


def _edge_sag(curvature: Decimal, conic: Decimal, edge_radius: Decimal, parameter: str) -> Decimal:
    """Sag at the outline; the surface must be defined (and not vertical) over the whole outline."""

    with localcontext(WORK_CONTEXT):
        value = sag(curvature, conic, edge_radius)
        if value is None or (curvature != 0 and ONE - (ONE + conic) * curvature * curvature * edge_radius * edge_radius <= 0):
            raise infeasible(f"a surface of radius {radius_text(curvature)} mm is not defined over the outer radius {text(edge_radius)} mm: "
                             "use a longer focal length, a smaller diameter or another shape", parameter)
        return value


def coating_approximations(coatings) -> set[SimulationApproximation]:
    """Approximations a set of coatings implies (data-driven, as the public catalog states them)."""

    out: set[SimulationApproximation] = set()
    for coating in coatings:
        out.add(SimulationApproximation.COATING_PHASE_NOT_MODELLED)
        out.add(SimulationApproximation.TYPICAL_SPECTRAL_CURVE if coating.is_typical else SimulationApproximation.COATING_AT_SPECIFIED_BOUND)
        if not coating.polarization_resolved:
            out.add(SimulationApproximation.POLARIZATION_INDEPENDENT_COATING)
        if any(Decimal(b.reflectance) + Decimal(b.transmittance) == ONE for b in coating.bands or ()):
            out.add(SimulationApproximation.LOSSLESS_COATING)
    return out


def refractive_approximations(materials: list[ResolvedMaterial], coatings) -> list[str]:
    approximations = {SimulationApproximation.NOMINAL_PRESCRIPTION, SimulationApproximation.IDEAL_SURFACE_FIGURE,
                      SimulationApproximation.EDGE_AS_OPAQUE_RIM, SimulationApproximation.GHOST_REFLECTIONS_NOT_TRACED}
    if any(m.from_catalog for m in materials):
        approximations.add(SimulationApproximation.CATALOG_MATERIAL_DISPERSION)
    approximations |= coating_approximations(coatings)
    return sorted(a.value for a in approximations)


def model_band(values: Mapping[str, Any], materials: list[ResolvedMaterial], coating, wavelengths: tuple[Decimal, ...],
               context_parameter: str = "valid_band_nm") -> tuple[Decimal, Decimal]:
    """The model's valid band: ``valid_band_nm`` when given (inside every material's band), else their common band; each
    design wavelength must lie in it."""

    bands = [m.band for m in materials]
    if coating is not None:
        bands.append(coating.span_nm)
    low, high = band_intersection(bands, context_parameter)
    given = values.get("valid_band_nm")
    if given is not None:
        if not (low <= given[0] and given[1] <= high):
            raise invalid(f"valid_band_nm must lie inside the materials' / coating's common band [{text(low)}, {text(high)}] nm",
                          "valid_band_nm")
        low, high = given
    for wavelength in wavelengths:
        if not low <= wavelength <= high:
            raise invalid(f"the design wavelength {text(wavelength)} nm lies outside the model band [{text(low)}, {text(high)}] nm",
                          "design_wavelength_nm")
    return low, high


def _outline(values: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    """(edge radius, clear radius) from ``diameter_mm`` and ``clear_aperture_diameter_mm``."""

    diameter = values["diameter_mm"]
    clear = values.get("clear_aperture_diameter_mm")
    clear = DEFAULT_CLEAR_FRACTION * diameter if clear is None else clear
    if clear > diameter:
        raise invalid("the clear aperture cannot exceed the outer diameter", "clear_aperture_diameter_mm")
    return diameter / TWO, clear / TWO


# ------------------------------------------------------------------------------------------------ ideal thin lens


def build_ideal_thin(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    focal = values["focal_length_mm"]
    clear = values["clear_aperture_diameter_mm"]
    diameter = values.get("diameter_mm") or clear
    if clear > diameter:
        raise invalid("the clear aperture cannot exceed the outer diameter", "clear_aperture_diameter_mm")
    low, high = values["valid_band_nm"]
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("thin-lens", "f", focal)
    physics = {
        "model_kind": "IDEAL_THIN_LENS", "model_version": "1", "focal_length_mm": text(focal),
        "clear_radius_mm": text(clear / TWO), "edge_radius_mm": text(diameter / TWO), "transmittance": text(values["transmittance"]),
        "plane_z_mm": "0", "valid_band_nm": band(low, high, FactStatus.DESIGN_BASIS), "status": FactStatus.DESIGN_BASIS.value,
    }
    model = optical_model(model_id, values.get("display_name") or f"Ideal thin lens f = {text(focal)} mm", physics,
                          geometry=cylinder(diameter, ZERO, thickness))
    design = {"focal_length_mm": text(focal), "equations": "ray deflection of a perfect paraxial lens: u' = u - y / f",
              "drawn_thickness_mm": text(thickness)}
    return finish("lens.ideal_thin", model, role="lens", values=values, design=design)


# ---------------------------------------------------------------------------------------------------- singlet


def shape_factor(values: Mapping[str, Any], index: Decimal) -> Decimal:
    form, factor = values.get("form"), values.get("shape_factor")
    if form is not None and factor is not None:
        raise invalid("give either form or shape_factor, not both", "shape_factor")
    if factor is not None:
        return factor
    form = form or "SYMMETRIC"
    with localcontext(WORK_CONTEXT):
        return {"SYMMETRIC": ZERO, "PLANO_EXIT": ONE, "PLANO_ENTRY": -ONE,
                "BEST_FORM": TWO * (index * index - ONE) / (index + TWO)}[form]


def singlet_curvatures(power: Decimal, index: Decimal, thickness: Decimal, factor: Decimal) -> tuple[Decimal, Decimal]:
    """(c1, c2) of the thick singlet of ``power`` (module docstring); ``ValueError`` when none exists."""

    with localcontext(WORK_CONTEXT):
        k = index - ONE
        a_coefficient = k * k * thickness * (factor * factor - ONE) / (FOUR * index)
        discriminant = k * k + FOUR * a_coefficient * power
        if discriminant < 0:
            raise ValueError("no thick lens of this focal length, shape and thickness exists")
        a = TWO * power / (k + discriminant.sqrt())
        return a * (factor + ONE) / TWO, a * (factor - ONE) / TWO


def build_singlet(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    focal, wavelength = values["focal_length_mm"], values["design_wavelength_nm"]
    material = context.resolver.resolve(values["material"], "material")
    edge_radius, clear_radius = _outline(values)
    diameter = values["diameter_mm"]
    coating_spec = values.get("ar_coating")
    try:
        index = material.index(wavelength)
    except ValueError as exc:
        raise invalid(str(exc), "design_wavelength_nm") from None
    if index <= ONE:
        raise infeasible(f"a lens in air needs an index above 1 (n = {text(index, RATIO_QUANTUM)} at {text(wavelength)} nm)", "material")
    factor = shape_factor(values, index)
    rear_conic = values.get("rear_conic") or "NONE"
    conic2 = ZERO
    if rear_conic == "STIGMATIC_COLLIMATED":
        if factor != -ONE:
            raise invalid("the stigmatic rear conic needs a plano entry face (form PLANO_ENTRY)", "rear_conic")
        if values.get("cylinder_axis") is not None:
            raise invalid("a cylindrical surface has no conic constant", "rear_conic")
        conic2 = quantize(-(index * index), RATIO_QUANTUM)
    cylinder_axis = values.get("cylinder_axis")
    power = ONE / focal
    te_given = values.get("min_edge_thickness_mm")
    te_min = te_given or edge_rule(diameter)
    tc_min = values.get("min_center_thickness_mm") or center_rule(diameter)

    def solve(thickness: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        try:
            c1, c2 = singlet_curvatures(power, index, thickness, factor)
        except ValueError as exc:
            raise infeasible(f"{exc}: change the shape or the thickness", "focal_length_mm") from None
        edge = thickness + _edge_sag(c2, conic2, edge_radius, "focal_length_mm") - _edge_sag(c1, ZERO, edge_radius, "focal_length_mm")
        return c1, c2, edge

    given = values.get("center_thickness_mm")
    if given is not None:
        thickness = given
        c1, c2, edge = solve(thickness)
        if not edge_ok(edge, te_given):
            bound = f"below min_edge_thickness_mm {text(te_given)} mm" if te_given is not None else "not positive"
            raise infeasible(f"the edge thickness would be {length(edge)} mm, {bound}: increase center_thickness_mm "
                             "(or omit it to apply the thickening rule)", "center_thickness_mm")
    else:
        thickness = tc_min
        for _ in range(MAX_THICKENING_STEPS):
            _c1, _c2, edge = solve(thickness)
            target = max(tc_min, thickness + te_min - edge)
            if abs(target - thickness) <= THICKNESS_CONVERGENCE_MM:
                break
            thickness = target
        thickness = ceil_to(thickness, THICKNESS_QUANTUM_MM)
        for _ in range(MAX_THICKENING_STEPS):
            c1, c2, edge = solve(thickness)
            if edge >= te_min:
                break
            thickness += THICKNESS_QUANTUM_MM
        else:
            raise infeasible("no centre thickness gives the required edge thickness", "min_edge_thickness_mm")
    low, high = model_band(values, [material], None, (wavelength,))
    coating = resolve_ar_coating(coating_spec, low, high, context, "ar_coating") if coating_spec is not None else None
    if coating is not None:
        low, high = model_band(values, [material], coating, (wavelength,))
    surfaces = []
    for sid, z, curvature, conic in (("S1", ZERO, c1, ZERO), ("S2", thickness, c2, conic2)):
        surface: dict[str, Any] = {"surface_id": sid, "vertex_z_mm": text(z), "clear_radius_mm": text(clear_radius),
                                   "traced": "TRANSMITTED", "status": FactStatus.DERIVED.value}
        radius = radius_text(curvature)
        if radius is not None:
            surface["radius_mm"] = radius
            if conic != 0:
                surface["conic"] = text(conic)
            if cylinder_axis is not None:
                surface["cylinder_axis"] = cylinder_axis
        if coating is not None:
            surface["coating_id"] = coating.coating_id
        surfaces.append(surface)
    w1, w2 = written_curvature(c1), written_curvature(c2)
    first = sequence_first_order([ParaxialSurface(ZERO, w1, ONE, index), ParaxialSurface(thickness, w2, index, ONE)])
    axis = "ROTATIONAL" if cylinder_axis is None else ("Y" if cylinder_axis == "X" else "X")
    paraxial = paraxial_block(first, wavelength, axis)
    physics = {
        "model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value,
        "surfaces": surfaces, "media": [material.material_id], "edge_radius_mm": text(edge_radius), "paraxial": paraxial,
        "valid_band_nm": band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS),
        "approximations": refractive_approximations([material], [coating] if coating else []), "status": FactStatus.DERIVED.value,
    }
    edge_s1, edge_s2 = _edge_sag(w1, ZERO, edge_radius, "focal_length_mm"), _edge_sag(w2, conic2, edge_radius, "focal_length_mm")
    z_min, z_max = min(ZERO, edge_s1), max(thickness, thickness + edge_s2)
    model_id = values.get("model_id") or default_model_id("singlet", material.material_id, "f", focal)
    name = values.get("display_name") or f"Singlet {material.material_id} f = {text(focal)} mm"
    model = optical_model(model_id, name, physics, frame=transmissive_frame(cylinder_axis is None),
                          geometry=cylinder(diameter, z_min, z_max))
    design = {
        "refractive_index": text(index, RATIO_QUANTUM), "design_wavelength_nm": text(wavelength), "shape_factor": text(factor, RATIO_QUANTUM),
        "center_thickness_mm": text(thickness), "edge_thickness_mm": length(thickness + edge_s2 - edge_s1),
        "radii_mm": [s.get("radius_mm", "inf") for s in surfaces], "paraxial": paraxial,
        "equations": "thick-lens power 1/f = (n-1)(c1-c2) + (n-1)^2 t c1 c2 / n with c1 = a(X+1)/2, c2 = a(X-1)/2",
    }
    if conic2 != 0:
        design["rear_conic"] = text(conic2)
    return finish("lens.singlet", model, materials=[material.material], coatings=[coating] if coating else [], role="lens",
                  values=values, design=design)


def paraxial_block(first, wavelength: Decimal, axis: str = "ROTATIONAL") -> dict[str, str]:
    return {
        "wavelength_nm": text(wavelength), "effective_focal_length_mm": length(first.efl), "back_focal_length_mm": length(first.bfl),
        "front_focal_length_mm": length(first.ffl), "front_principal_plane_z_mm": length(first.front_principal_z),
        "rear_principal_plane_z_mm": length(first.rear_principal_z), "axis": axis, "status": FactStatus.DERIVED.value,
    }


# --------------------------------------------------------------------------------------------------- achromat


def _doublet_surfaces(a1: Decimal, a2: Decimal, factor: Decimal, t1: Decimal, t2: Decimal, n1: Decimal, n2: Decimal):
    c1, c2 = a1 * (factor + ONE) / TWO, a1 * (factor - ONE) / TWO
    c3 = c2 - a2
    return [ParaxialSurface(ZERO, c1, ONE, n1), ParaxialSurface(t1, c2, n1, n2), ParaxialSurface(t1 + t2, c3, n2, ONE)], (c1, c2, c3)


def _doublet_power(a1, a2, factor, t1, t2, n1, n2) -> Decimal:
    surfaces, _ = _doublet_surfaces(a1, a2, factor, t1, t2, n1, n2)
    return -sequence_matrix(surfaces)[2]


def _doublet_bfl(a1, a2, factor, t1, t2, n1, n2) -> Decimal:
    """Back focal length (last vertex to the rear focal point) of the doublet; ``ValueError`` for an afocal trial."""

    surfaces, _ = _doublet_surfaces(a1, a2, factor, t1, t2, n1, n2)
    a, _b, c, _d = sequence_matrix(surfaces)
    if c == 0:
        raise ValueError("the exact achromatic solve met an afocal doublet")
    return -a / c


def solve_achromat(power: Decimal, factor: Decimal, t1: Decimal, t2: Decimal, crown: tuple[Decimal, Decimal, Decimal],
                   flint: tuple[Decimal, Decimal, Decimal], start: tuple[Decimal, Decimal]) -> tuple[Decimal, Decimal]:
    """(a1, a2) with EFL(l_c) = 1 / power and BFL(l_s) = BFL(l_l) -- the short- and long-wavelength foci of a collimated
    beam coincide -- for the thick doublet (Newton, finite-difference Jacobian); ``ValueError`` when it does not
    converge."""

    (n1s, n1c, n1l), (n2s, n2c, n2l) = crown, flint
    with localcontext(WORK_CONTEXT):
        def residual(a1: Decimal, a2: Decimal) -> tuple[Decimal, Decimal]:
            return (_doublet_power(a1, a2, factor, t1, t2, n1c, n2c) - power,
                    _doublet_bfl(a1, a2, factor, t1, t2, n1s, n2s) - _doublet_bfl(a1, a2, factor, t1, t2, n1l, n2l))

        def size(g: tuple[Decimal, Decimal]) -> Decimal:  # both residuals made dimensionless
            return abs(g[0] / power) + abs(g[1] * power)

        a1, a2 = start
        tolerance = abs(power) * NEWTON_RELATIVE_TOLERANCE
        focus_tolerance = NEWTON_RELATIVE_TOLERANCE / abs(power)
        g1, g2 = residual(a1, a2)
        for _ in range(MAX_NEWTON_STEPS):
            if abs(g1) <= tolerance and abs(g2) <= focus_tolerance:
                return a1, a2
            h1, h2 = abs(a1) * JACOBIAN_RELATIVE_STEP, abs(a2) * JACOBIAN_RELATIVE_STEP
            f11, f21 = residual(a1 + h1, a2)
            f12, f22 = residual(a1, a2 + h2)
            j11, j21 = (f11 - g1) / h1, (f21 - g2) / h1
            j12, j22 = (f12 - g1) / h2, (f22 - g2) / h2
            det = j11 * j22 - j12 * j21
            if det == 0:
                break
            d1, d2 = (j22 * g1 - j12 * g2) / det, (j11 * g2 - j21 * g1) / det
            # damped step: halve it until the residual shrinks (a full Newton step can overshoot far from the solution)
            for _halving in range(MAX_STEP_HALVINGS):
                try:
                    trial = residual(a1 - d1, a2 - d2)
                except ValueError:
                    trial = None
                if trial is not None and size(trial) < size((g1, g2)):
                    break
                d1, d2 = d1 / 2, d2 / 2
            else:
                break
            a1, a2, (g1, g2) = a1 - d1, a2 - d2, trial
    raise ValueError("no thick doublet of these materials and crown shape brings the short- and long-wavelength foci together at "
                     "this focal length")


def build_achromat(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    focal = values["focal_length_mm"]
    crown = context.resolver.resolve(values["crown_material"], "crown_material")
    flint = context.resolver.resolve(values["flint_material"], "flint_material")
    if crown.material_id == flint.material_id:
        raise invalid("an achromat needs two different materials", "flint_material")
    wavelengths = values.get("wavelengths_nm") or FRAUNHOFER_FDC_NM
    short, center, long = wavelengths
    edge_radius, clear_radius = _outline(values)
    diameter = values["diameter_mm"]
    indices = []
    for material, parameter in ((crown, "crown_material"), (flint, "flint_material")):
        try:
            indices.append(tuple(material.index(w) for w in (short, center, long)))
        except ValueError as exc:
            raise invalid(f"{exc}; choose wavelengths_nm inside both materials' bands", "wavelengths_nm") from None
    n1, n2 = indices
    with localcontext(WORK_CONTEXT):
        if n1[0] == n1[2] or n2[0] == n2[2]:
            raise infeasible("a material without dispersion between the short and long wavelengths cannot form an achromat", "flint_material")
        v1 = (n1[1] - ONE) / (n1[0] - n1[2])
        v2 = (n2[1] - ONE) / (n2[0] - n2[2])
        if v1 == v2:
            raise infeasible("the two materials have the same Abbe number at these wavelengths", "flint_material")
        power = ONE / focal
        phi1, phi2 = power * v1 / (v1 - v2), -power * v2 / (v1 - v2)
        thin = (phi1 / (n1[1] - ONE), phi2 / (n2[1] - ONE))
    factor = values.get("crown_shape_factor")
    factor = ZERO if factor is None else factor
    te_given = values.get("min_edge_thickness_mm")
    te_min = te_given or edge_rule(diameter)
    tc_min = values.get("min_center_thickness_mm") or center_rule(diameter)
    t1_given, t2_given = values.get("crown_center_thickness_mm"), values.get("flint_center_thickness_mm")
    # an element with a given centre thickness keeps it: its edge only has to be positive, unless a minimum is given too
    edge_minimums = (te_min if t1_given is None else te_given, te_min if t2_given is None else te_given)

    def solve(t1: Decimal, t2: Decimal, start) -> tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal, Decimal], Decimal, Decimal]:
        try:
            a = solve_achromat(power, factor, t1, t2, n1, n2, start)
        except ValueError as exc:
            raise infeasible(f"{exc}: try another crown_shape_factor (0 or -0.5 suit most doublets), a longer focal length, a smaller "
                             "diameter or other materials", "focal_length_mm") from None
        _surfaces, (c1, c2, c3) = _doublet_surfaces(a[0], a[1], factor, t1, t2, n1[1], n2[1])
        s1, s2, s3 = (_edge_sag(c, ZERO, edge_radius, "focal_length_mm") for c in (c1, c2, c3))
        return a, (c1, c2, c3), t1 + s2 - s1, t2 + s3 - s2

    t1 = t1_given if t1_given is not None else tc_min
    t2 = t2_given if t2_given is not None else tc_min
    a = thin
    if t1_given is None or t2_given is None:
        for _ in range(MAX_THICKENING_STEPS):
            a, _c, e1, e2 = solve(t1, t2, a)
            n_t1 = t1 if t1_given is not None else max(tc_min, t1 + te_min - e1)
            n_t2 = t2 if t2_given is not None else max(tc_min, t2 + te_min - e2)
            if abs(n_t1 - t1) <= THICKNESS_CONVERGENCE_MM and abs(n_t2 - t2) <= THICKNESS_CONVERGENCE_MM:
                break
            t1, t2 = n_t1, n_t2
        t1 = t1 if t1_given is not None else ceil_to(t1, THICKNESS_QUANTUM_MM)
        t2 = t2 if t2_given is not None else ceil_to(t2, THICKNESS_QUANTUM_MM)
    for _ in range(MAX_THICKENING_STEPS):
        a, curvatures, e1, e2 = solve(t1, t2, a)
        short_edges = [(not edge_ok(edge, minimum), given) for edge, minimum, given in zip((e1, e2), edge_minimums, (t1_given, t2_given))]
        if not any(short for short, _ in short_edges):
            break
        for (short, given), parameter in zip(short_edges, ("crown_center_thickness_mm", "flint_center_thickness_mm")):
            if short and given is not None:
                bound = "thinner than min_edge_thickness_mm" if te_given is not None else "of zero or negative thickness"
                raise infeasible(f"an element's edge would be {bound} at the given centre thickness", parameter)
        t1 = t1 + THICKNESS_QUANTUM_MM if short_edges[0][0] else t1
        t2 = t2 + THICKNESS_QUANTUM_MM if short_edges[1][0] else t2
    else:
        raise infeasible("no centre thicknesses give the required edge thickness", "min_edge_thickness_mm")
    coating_spec = values.get("ar_coating")
    low, high = model_band(values, [crown, flint], None, wavelengths, "valid_band_nm")
    coating = resolve_ar_coating(coating_spec, low, high, context, "ar_coating") if coating_spec is not None else None
    if coating is not None:
        low, high = model_band(values, [crown, flint], coating, wavelengths, "valid_band_nm")
    written = [written_curvature(c) for c in curvatures]
    surfaces = []
    for k, (z, curvature) in enumerate(((ZERO, curvatures[0]), (t1, curvatures[1]), (t1 + t2, curvatures[2]))):
        surface: dict[str, Any] = {"surface_id": f"S{k + 1}", "vertex_z_mm": text(z), "clear_radius_mm": text(clear_radius),
                                   "traced": "TRANSMITTED", "status": FactStatus.DERIVED.value}
        radius = radius_text(curvature)
        if radius is not None:
            surface["radius_mm"] = radius
        if coating is not None and k != 1:
            surface["coating_id"] = coating.coating_id
        surfaces.append(surface)

    def written_first_order(index_crown: Decimal, index_flint: Decimal):
        return sequence_first_order([ParaxialSurface(ZERO, written[0], ONE, index_crown), ParaxialSurface(t1, written[1], index_crown, index_flint),
                                     ParaxialSurface(t1 + t2, written[2], index_flint, ONE)])

    first_c = written_first_order(n1[1], n2[1])
    first_s, first_l = written_first_order(n1[0], n2[0]), written_first_order(n1[2], n2[2])
    approximations = refractive_approximations([crown, flint], [coating] if coating else [])
    approximations = sorted({*approximations, SimulationApproximation.CEMENT_LAYER_NOT_MODELLED.value})
    physics = {
        "model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value,
        "surfaces": surfaces, "media": [crown.material_id, flint.material_id], "edge_radius_mm": text(edge_radius),
        "paraxial": paraxial_block(first_c, center),
        "valid_band_nm": band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS),
        "approximations": approximations, "status": FactStatus.DERIVED.value,
    }
    w_edges = [_edge_sag(c, ZERO, edge_radius, "focal_length_mm") for c in written]
    z_min = min(ZERO, w_edges[0])
    z_max = max(t1 + t2, t1 + t2 + w_edges[2])
    model_id = values.get("model_id") or default_model_id("achromat", crown.material_id, flint.material_id, "f", focal)
    name = values.get("display_name") or f"Achromatic doublet {crown.material_id}/{flint.material_id} f = {text(focal)} mm"
    model = optical_model(model_id, name, physics, frame=transmissive_frame(), geometry=cylinder(diameter, z_min, z_max))
    design = {
        "wavelengths_nm": [text(w) for w in wavelengths],
        "crown_indices": [text(n, RATIO_QUANTUM) for n in n1], "flint_indices": [text(n, RATIO_QUANTUM) for n in n2],
        "abbe_numbers": [text(v1, RATIO_QUANTUM), text(v2, RATIO_QUANTUM)],
        "thin_lens_element_focal_lengths_mm": [length(ONE / phi1), length(ONE / phi2)],
        "crown_shape_factor": text(factor), "center_thicknesses_mm": [text(t1), text(t2)],
        "edge_thicknesses_mm": [length(t1 + w_edges[1] - w_edges[0]), length(t2 + w_edges[2] - w_edges[1])],
        "radii_mm": [s.get("radius_mm", "inf") for s in surfaces], "paraxial": physics["paraxial"],
        "efl_short_mm": length(first_s.efl), "efl_long_mm": length(first_l.efl),
        "bfl_short_mm": length(first_s.bfl), "bfl_long_mm": length(first_l.bfl),
        # where a collimated beam focuses, long minus short wavelength (zero by design, up to the written radii)
        "axial_focus_shift_long_minus_short_mm": length(first_l.bfl - first_s.bfl),
        # what the axial condition leaves for a thick lens: a chromatic difference of focal length (image size)
        "efl_difference_long_minus_short_mm": length(first_l.efl - first_s.efl),
        "equations": "thin-lens achromat phi1 = phi V1/(V1-V2), phi2 = -phi V2/(V1-V2); then exact thick-lens re-solve of "
                     "EFL(l_c) = f and BFL(l_s) = BFL(l_l) (coincident foci of a collimated beam)",
    }
    return finish("lens.achromat", model, materials=[crown.material, flint.material], coatings=[coating] if coating else [],
                  role="lens", values=values, design=design)


__all__ = ["FRAUNHOFER_FDC_NM", "REAR_CONICS", "SHAPE_FORMS", "build_achromat", "build_ideal_thin", "build_singlet", "center_rule",
           "edge_rule", "paraxial_block", "singlet_curvatures", "solve_achromat"]
