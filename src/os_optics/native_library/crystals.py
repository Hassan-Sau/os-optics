"""Native crystal polarization optics: elements made of a uniaxial crystal (calcite, alpha-BBO, YVO4, ...) whose optic axis the
element states, traced by the Engine's uniaxial crystal model (ordinary and extraordinary beams, walk-off).

``splitter.beam_displacer`` -- a rectangular crystal plate (``SURFACE_SEQUENCE``: entrance face ``S1`` at ``z = 0``, exit face
``S2`` at ``z = L``, both normal to local Z) whose optic axis lies in the local X-Z plane at ``theta`` from +Z toward -X:
``(-sin theta, 0, cos theta)``. A beam along +Z splits into the ordinary beam (polarized along local Y), which goes straight
through, and the extraordinary beam (polarized along local X), whose ray walks off its wave normal by ``rho`` with
``tan(theta + rho) = (n_o / n_e)^2 tan(theta)`` and leaves parallel to the input, displaced by ``d = L tan(rho)`` along local
+X (a negative crystal, ``n_e < n_o``; a positive one displaces it along -X). The largest walk-off is at
``tan(theta) = n_e / n_o``. (Born & Wolf, *Principles of Optics*, 7th ed., sec. 15.3; Saleh & Teich, *Fundamentals of
Photonics*, 3rd ed., sec. 6.3.)

``polarizer.glan`` -- an air-spaced Glan polarizer (``COMPOUND`` of two right-angle crystal prisms, ``SURFACE_SET`` each, square
aperture ``a``): prism ``A`` has its entrance face at ``z = 0``, prism ``B`` its exit face at ``z = L``; the gap between their
hypotenuse faces is a plane through the local Y axis, its normal at the gap angle ``A_g`` to +Z (toward -X), ``air_gap_mm``
wide. A beam along +Z meets the gap at ``A_g``: the ordinary wave is totally reflected (``A_g`` beyond its critical angle
``asin(1 / n_o)``) and leaves prism A through its side face at local +X; the extraordinary wave (``A_g`` inside its critical
angle ``asin(1 / n_e)``) crosses the gap and leaves along +Z. ``GLAN_TAYLOR``: both optic axes along local X, in the plane of
incidence at the gap: the transmitted light is polarized along local X (p at the gap, which keeps its reflection loss small);
``GLAN_FOUCAULT``: both optic axes along local Y, perpendicular to the plane of incidence: the transmitted light is polarized
along local Y (s at the gap). The gap angle defaults to the middle of the window ``[asin(1 / n_o), asin(1 / n_e)]`` at the design
wavelength; a negative crystal is needed. The prism length is ``L = a tan(A_g) + air_gap / cos(A_g)``. (Born & Wolf, sec. 15.4;
Bennett, *Polarizers*, in Handbook of Optics vol. I, 2nd ed., ch. 3.)

``polarizer.glan_thompson`` -- the Glan geometry with the gap filled by a cement layer (``COMPOUND`` joint with
``cement_material_id``, ``cement_thickness_mm`` thick): the ordinary wave is totally reflected at the crystal-cement face
(``A_g`` beyond ``asin(n_c / n_o)``), the extraordinary wave crosses the layer (``A_g`` inside ``asin(n_c / n_e)``, or at any
angle when ``n_c >= n_e``). Both optic axes along local Y, perpendicular to the plane of reflection: the transmitted light is
polarized along local Y. A negative crystal and a cement with ``n_c < n_o`` are needed. (Bennett, *Polarizers*, Handbook of
Optics vol. I, 2nd ed., ch. 3.)

``splitter.wollaston`` and ``splitter.rochon`` -- two right-angle prisms of one uniaxial crystal joined along a plane through
the local Y axis whose normal is at the wedge angle ``theta`` to +Z (toward -X), as a ``COMPOUND`` with a joint (an optical
contact, or a cement layer). Square aperture ``a``; prism A has its entrance face at ``z = 0``, prism B its exit face at
``z = a tan(theta) + 2 e`` (``e`` the edge thickness of each prism at its thin end; plus ``t / cos(theta)`` across a cement layer
of thickness ``t``). Wollaston: prism A's optic axis along local X, prism B's along local Y: a beam along +Z leaves as two beams
of orthogonal linear polarization deviated to either side in the X-Z plane, their separation close to
``2 asin((n_o - n_e) tan(theta))``. Rochon: prism A's optic axis along local Z (the input travels as the ordinary wave),
prism B's along local Y: light polarized along local X leaves undeviated, light polarized along local Y is deviated by close
to ``asin((n_o - n_e) tan(theta))``. The design report gives the exact exit directions of the axial beam at the design
wavelength from Snell's law (the joint and the exit face are each crossed with the index of the wave on either side, every
wave there travelling perpendicular to its crystal's optic axis or along it). (Born & Wolf, sec. 15.4; Bennett.)

Every value is exact decimal arithmetic (``trig.py`` for the angles).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any

from .._vendor.shared_contracts.design_catalog import FactStatus, ModelFidelity, SimulationApproximation
from .common import BuildContext, NativeBuild, band, box, default_model_id, finish, optical_model, resolve_ar_coating, transmissive_frame
from .errors import infeasible, invalid
from .materials import band_intersection
from .lenses import coating_approximations
from .numbers import HALF, LENGTH_QUANTUM_MM, ONE, RATIO_QUANTUM, THICKNESS_QUANTUM_MM, TWO, WORK_CONTEXT, ZERO, ceil_to, length, quantize, text
from .trig import asin, atan, degrees, sin_cos_deg, tan_deg

GLAN_VARIANTS = ("GLAN_TAYLOR", "GLAN_FOUCAULT")
#: The optic axes of a split prism's members (prism A, prism B; unit frame).
SPLIT_PRISM_AXES = {"WOLLASTON": (("1", "0", "0"), ("0", "1", "0")), "ROCHON": (("0", "0", "1"), ("0", "1", "0"))}
#: Cement layer of a cemented prism when none is given (mm), and the edge thickness of a split prism's thin end as a
#: fraction of its aperture when none is given.
DEFAULT_CEMENT_MM = Decimal("0.01")
DEFAULT_EDGE_FRACTION = Decimal("0.1")
#: Resolution of angles written in design reports (degrees).
ANGLE_QUANTUM_DEG = Decimal("0.000000001")
#: Air gap of a Glan polarizer when none is given (mm).
DEFAULT_AIR_GAP_MM = Decimal("0.02")
#: Optic-axis angle of a beam displacer when none is given (deg from local +Z), its outer width and height (mm), and the clear
#: aperture fraction of a crystal face when none is given.
DEFAULT_AXIS_ANGLE_DEG = Decimal(45)
DEFAULT_DISPLACER_SIDE_MM = Decimal(10)
DEFAULT_CLEAR_FRACTION = Decimal("0.9")
#: Resolution of a default gap angle (deg).
GAP_ANGLE_QUANTUM_DEG = Decimal("0.001")
#: Longest unit model id that leaves room for the member suffixes (``-A``, ``-B``).
MAX_UNIT_ID = 60


def _indices(material, wavelength: Decimal, parameter: str = "design_wavelength_nm") -> tuple[Decimal, Decimal]:
    try:
        return material.index(wavelength), material.extraordinary_index(wavelength)
    except ValueError as exc:
        raise invalid(str(exc), parameter) from None


def _band(values: Mapping[str, Any], material, coating, wavelength: Decimal) -> tuple[Decimal, Decimal]:
    from .lenses import model_band

    return model_band(values, [material], coating, (wavelength,))


def _approximations(material, coating, untraced_faces: bool) -> list[str]:
    approximations = {SimulationApproximation.NOMINAL_PRESCRIPTION, SimulationApproximation.IDEAL_SURFACE_FIGURE,
                      SimulationApproximation.EDGE_AS_OPAQUE_RIM, SimulationApproximation.GHOST_REFLECTIONS_NOT_TRACED}
    if untraced_faces:
        approximations.add(SimulationApproximation.UNTRACED_FACES_BLOCK)
    approximations |= coating_approximations([coating] if coating else [])
    return sorted(a.value for a in approximations)


def _coating(values: Mapping[str, Any], material, wavelength: Decimal, context: BuildContext) -> tuple[Any, Decimal, Decimal]:
    low, high = _band(values, material, None, wavelength)
    spec = values.get("ar_coating")
    if spec is None:
        return None, low, high
    coating = resolve_ar_coating(spec, low, high, context, "ar_coating")
    low, high = _band(values, material, coating, wavelength)
    return coating, low, high


def _triple(vector, quantum: Decimal = RATIO_QUANTUM) -> list[str]:
    return [text(quantize(v, quantum) + 0) for v in vector]


# ---------------------------------------------------------------------------------------------------- beam displacer


def walk_off_deg(n_o: Decimal, n_e: Decimal, theta_deg: Decimal) -> Decimal:
    """The walk-off angle ``rho`` (deg) of the extraordinary ray from its wave normal at ``theta_deg`` to the optic axis:
    ``theta + rho = atan((n_o / n_e)^2 tan(theta))`` (positive for a negative crystal: away from the axis)."""

    with localcontext(WORK_CONTEXT):
        return degrees(atan((n_o / n_e) ** 2 * tan_deg(theta_deg))) - theta_deg


def build_beam_displacer(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    material = context.resolver.resolve_crystal(values["material"], "material")
    wavelength = values["design_wavelength_nm"]
    theta = values.get("axis_angle_deg") or DEFAULT_AXIS_ANGLE_DEG
    n_o, n_e = _indices(material, wavelength)
    if n_o == n_e:
        raise infeasible(f"the crystal has no birefringence at {text(wavelength)} nm (n_o = n_e): it displaces nothing", "material")
    rho = walk_off_deg(n_o, n_e, theta)
    with localcontext(WORK_CONTEXT):
        tan_rho = tan_deg(rho)
    given_length, given_shift = values.get("length_mm"), values.get("displacement_mm")
    if (given_length is None) == (given_shift is None):
        raise invalid("state exactly one of length_mm (the crystal length) and displacement_mm (the beam separation)", "length_mm")
    if given_length is not None:
        crystal_length = given_length
    else:
        with localcontext(WORK_CONTEXT):
            crystal_length = ceil_to(given_shift / abs(tan_rho), THICKNESS_QUANTUM_MM)
    with localcontext(WORK_CONTEXT):
        shift = crystal_length * tan_rho
        sin_t, cos_t = sin_cos_deg(theta)
        n_theta = ONE / (cos_t * cos_t / (n_o * n_o) + sin_t * sin_t / (n_e * n_e)).sqrt()
    width, height = values.get("width_mm") or DEFAULT_DISPLACER_SIDE_MM, values.get("height_mm") or DEFAULT_DISPLACER_SIDE_MM
    fraction = values.get("clear_aperture_fraction") or DEFAULT_CLEAR_FRACTION
    coating, low, high = _coating(values, material, wavelength, context)
    half_x, half_y = width / TWO, height / TWO
    face = {"clear_half_widths_mm": [text(fraction * half_x), text(fraction * half_y)], "traced": "TRANSMITTED",
            "status": FactStatus.DESIGN_BASIS.value}
    if coating is not None:
        face["coating_id"] = coating.coating_id
    physics = {
        "model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value,
        "surfaces": [{"surface_id": "S1", "vertex_z_mm": "0", **face}, {"surface_id": "S2", "vertex_z_mm": text(crystal_length), **face}],
        "media": [material.material_id], "edge_half_widths_mm": [text(half_x), text(half_y)],
        "valid_band_nm": band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS),
        "approximations": _approximations(material, coating, False), "status": FactStatus.DERIVED.value,
    }
    axis = (-sin_t, ZERO, cos_t)
    model_id = values.get("model_id") or default_model_id("displacer", material.material_id, "L", crystal_length)
    name = values.get("display_name") or f"Beam displacer {material.material_id} {text(crystal_length)} mm"
    model = optical_model(model_id, name, physics, frame=transmissive_frame(rotational=False), geometry=box(half_x, half_y, ZERO, crystal_length),
                          parameter="length_mm", optic_axes=[{"material_id": material.material_id, "direction": _triple(axis)}])
    design = {
        "design_wavelength_nm": text(wavelength), "ordinary_index": text(n_o, RATIO_QUANTUM), "extraordinary_index": text(n_e, RATIO_QUANTUM),
        "axis_angle_deg": text(theta), "optic_axis": _triple(axis), "extraordinary_phase_index": text(n_theta, RATIO_QUANTUM),
        "walk_off_deg": text(rho, ANGLE_QUANTUM_DEG), "length_mm": text(crystal_length), "displacement_mm": length(shift),
        "ordinary_polarization": "local Y (straight through)", "extraordinary_polarization": "local X (displaced along local X)",
        "equations": "tan(theta + rho) = (n_o / n_e)^2 tan(theta); d = L tan(rho) along local +X (negative crystal); "
                     "1 / n(theta)^2 = cos^2(theta) / n_o^2 + sin^2(theta) / n_e^2; largest walk-off at tan(theta) = n_e / n_o",
    }
    return finish("splitter.beam_displacer", model, materials=[material.material], coatings=[coating] if coating else [], role="splitter",
                  values=values, design=design)


# ------------------------------------------------------------------------------------------------------------ Glan


def _posed(sid: str, origin, normal, u_axis, half_u: Decimal, half_v: Decimal, crystal: str, coating) -> dict[str, Any]:
    surface: dict[str, Any] = {"surface_id": sid, "origin_mm": _triple(origin, LENGTH_QUANTUM_MM), "normal": _triple(normal),
                               "u_axis": _triple(u_axis), "clear_half_widths_mm": [text(half_u, LENGTH_QUANTUM_MM), text(half_v, LENGTH_QUANTUM_MM)],
                               "medium_front": "AIR", "medium_back": crystal, "traced": "TRANSMITTED", "status": FactStatus.DESIGN_BASIS.value}
    if coating is not None:
        surface["coating_id"] = coating.coating_id
    return surface


def _face(point, normal) -> dict[str, Any]:
    return {"point_mm": _triple(point, LENGTH_QUANTUM_MM), "outward_normal": _triple(normal)}


def build_glan(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    return _glan(values, context, None)


def build_glan_thompson(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    return _glan(values, context, context.resolver.resolve(values["cement"], "cement"))


def _glan(values: Mapping[str, Any], context: BuildContext, cement) -> NativeBuild:
    """A Glan polarizer: air-spaced (``cement`` ``None``: ``polarizer.glan``) or cemented (``polarizer.glan_thompson``)."""

    material = context.resolver.resolve_crystal(values["material"], "material")
    wavelength = values["design_wavelength_nm"]
    variant = "GLAN_THOMPSON" if cement is not None else values.get("variant") or "GLAN_TAYLOR"
    a = values["aperture_mm"]
    if cement is not None:
        gap = values.get("cement_thickness_mm") or DEFAULT_CEMENT_MM
    else:
        gap = values.get("air_gap_mm") or DEFAULT_AIR_GAP_MM
    fraction = values.get("clear_aperture_fraction") or DEFAULT_CLEAR_FRACTION
    n_o, n_e = _indices(material, wavelength)
    if not n_e < n_o:
        raise infeasible(f"a Glan polarizer reflects the ordinary wave at the gap: it needs a negative crystal (n_e < n_o); at "
                         f"{text(wavelength)} nm n_o = {text(n_o, RATIO_QUANTUM)}, n_e = {text(n_e, RATIO_QUANTUM)}", "material")
    n_gap = ONE
    if cement is not None:
        try:
            n_gap = cement.index(wavelength)
        except ValueError as exc:
            raise invalid(str(exc), "cement") from None
        if not n_gap < n_o:
            raise infeasible(f"the cement's index {text(n_gap, RATIO_QUANTUM)} at {text(wavelength)} nm is not below the crystal's ordinary "
                             f"index {text(n_o, RATIO_QUANTUM)}: the ordinary wave is not totally reflected at the layer", "cement")
    with localcontext(WORK_CONTEXT):
        critical_o = degrees(asin(n_gap / n_o))
        critical_e = degrees(asin(n_gap / n_e)) if n_gap < n_e else None
    angle = values.get("gap_angle_deg")
    if angle is None:
        if critical_e is None:
            raise invalid(f"the cement's index {text(n_gap, RATIO_QUANTUM)} is at least n_e = {text(n_e, RATIO_QUANTUM)}: the "
                          f"extraordinary wave crosses the layer at every angle; state gap_angle_deg (above "
                          f"{text(critical_o, GAP_ANGLE_QUANTUM_DEG)} deg)", "gap_angle_deg")
        with localcontext(WORK_CONTEXT):
            angle = quantize((critical_o + critical_e) / TWO, GAP_ANGLE_QUANTUM_DEG)
    elif critical_e is None and not critical_o < angle:
        raise infeasible(f"at {text(wavelength)} nm the gap angle must lie above the ordinary critical angle "
                         f"{text(critical_o, GAP_ANGLE_QUANTUM_DEG)} deg", "gap_angle_deg")
    elif critical_e is not None and not critical_o < angle < critical_e:
        raise infeasible(f"at {text(wavelength)} nm the gap angle must lie between the critical angles {text(critical_o, GAP_ANGLE_QUANTUM_DEG)} "
                         f"(ordinary) and {text(critical_e, GAP_ANGLE_QUANTUM_DEG)} deg (extraordinary)", "gap_angle_deg")
    coating, low, high = _coating(values, material, wavelength, context)
    if cement is not None:
        low, high = band_intersection([(low, high), cement.band], "cement")
    with localcontext(WORK_CONTEXT):
        s, c = sin_cos_deg(angle)
        t = s / c
        h = a * HALF                       # half aperture
        rise = a * t                       # hypotenuse rise across the aperture
        offset = gap / c                   # the gap along Z
        total = rise + offset              # prism length L
        m = (-s, ZERO, c)                  # gap normal (out of prism A)
        u = (c, ZERO, s)                   # in the gap plane, in the X-Z plane
        clear, clear_slant = fraction * h, fraction * h / c
    crystal = material.material_id
    axis = ("1", "0", "0") if variant == "GLAN_TAYLOR" else ("0", "1", "0")
    prefix = "glan" if cement is None else "glanth"
    unit_id = (values.get("model_id") or default_model_id(prefix, crystal, "a", a))[:MAX_UNIT_ID]
    # prism A (entrance): its frame is the unit frame; the hypotenuse runs from the entrance edge x = -a/2 to the far corner
    centre_a = (ZERO, ZERO, rise * HALF)
    side_a = (h, ZERO, rise * HALF)
    surfaces_a = [_posed("S1", (ZERO, ZERO, ZERO), (ZERO, ZERO, -ONE), (ONE, ZERO, ZERO), clear, clear, crystal, coating),
                  _posed("S2", centre_a, m, u, clear_slant, clear, crystal, None),
                  _posed("S3", side_a, (ONE, ZERO, ZERO), (ZERO, ZERO, ONE), fraction * rise * HALF, clear, crystal, None)]
    body_a = {"medium": crystal, "faces": [_face((ZERO, ZERO, ZERO), (ZERO, ZERO, -ONE)), _face(side_a, (ONE, ZERO, ZERO)),
                                           _face((ZERO, h, ZERO), (ZERO, ONE, ZERO)), _face((ZERO, -h, ZERO), (ZERO, -ONE, ZERO)),
                                           _face(centre_a, m)]}
    # prism B (exit): its frame's origin is the centre of its hypotenuse face, (0, 0, a tan(A) / 2 + gap / cos(A)) in the unit
    half_rise = rise * HALF
    surfaces_b = [_posed("S1", (ZERO, ZERO, ZERO), tuple(-v for v in m), u, clear_slant, clear, crystal, None),
                  _posed("S2", (ZERO, ZERO, half_rise), (ZERO, ZERO, ONE), (ONE, ZERO, ZERO), clear, clear, crystal, coating)]
    body_b = {"medium": crystal, "faces": [_face((ZERO, ZERO, ZERO), tuple(-v for v in m)), _face((ZERO, ZERO, half_rise), (ZERO, ZERO, ONE)),
                                           _face((-h, ZERO, ZERO), (-ONE, ZERO, ZERO)), _face((ZERO, h, ZERO), (ZERO, ONE, ZERO)),
                                           _face((ZERO, -h, ZERO), (ZERO, -ONE, ZERO))]}
    band_doc = band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS)
    members, models = [], []
    for member_id, surfaces, body, position in (("A", surfaces_a, body_a, (ZERO, ZERO, ZERO)),
                                                ("B", surfaces_b, body_b, (ZERO, ZERO, half_rise + offset))):
        physics = {"model_kind": "SURFACE_SET", "model_version": "1", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value,
                   "surfaces": surfaces, "body": body, "valid_band_nm": band_doc,
                   "approximations": _approximations(material, coating, True), "status": FactStatus.DERIVED.value}
        model_id = f"{unit_id}-{member_id}"
        models.append(optical_model(model_id, f"Glan prism {member_id} {crystal}", physics, parameter="aperture_mm",
                                    optic_axes=[{"material_id": crystal, "direction": list(axis)}]))
        members.append({"member_id": member_id, "model_id": model_id,
                        "pose": {"position_mm": [float(quantize(v, LENGTH_QUANTUM_MM)) for v in position], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}})
    ports = [{"port_id": "IN", "role": "INPUT", "point_mm": ["0", "0", "0"], "direction": ["0", "0", "1"], "status": FactStatus.DESIGN_BASIS.value},
             {"port_id": "OUT", "role": "OUTPUT", "point_mm": ["0", "0", length(total)], "direction": ["0", "0", "1"],
              "status": FactStatus.DERIVED.value}]
    physics = {"model_kind": "COMPOUND", "model_version": "1", "members": members, "envelope": box(h, h, ZERO, ceil_to(total, LENGTH_QUANTUM_MM)),
               "ports": ports}
    if cement is not None:
        physics["joints"] = [{"member_a": "A", "surface_a": "S2", "member_b": "B", "surface_b": "S1", "cement_material_id": cement.material_id}]
    title = {"GLAN_TAYLOR": "Glan-Taylor", "GLAN_FOUCAULT": "Glan-Foucault", "GLAN_THOMPSON": "Glan-Thompson"}[variant]
    model = optical_model(unit_id, values.get("display_name") or f"{title} polarizer {crystal} {text(a)} mm", physics)
    with localcontext(WORK_CONTEXT):
        reflected = (TWO * s * c, ZERO, -(c * c - s * s))  # the ordinary ray reflected at the gap (inside prism A)
    design = {
        "variant": variant, "design_wavelength_nm": text(wavelength), "ordinary_index": text(n_o, RATIO_QUANTUM),
        "extraordinary_index": text(n_e, RATIO_QUANTUM), "gap_angle_deg": text(angle),
        "critical_angle_ordinary_deg": text(critical_o, ANGLE_QUANTUM_DEG),
        "critical_angle_extraordinary_deg": None if critical_e is None else text(critical_e, ANGLE_QUANTUM_DEG),
        "aperture_mm": text(a), "length_mm": length(total), "optic_axis": list(axis),
        "transmitted_polarization": "local X (p at the gap)" if variant == "GLAN_TAYLOR" else "local Y (s at the gap)",
        "rejected_ordinary_direction_in_prism": _triple(reflected),
        "equations": "ordinary wave totally reflected at the gap: asin(n_gap / n_o) < A_g < asin(n_gap / n_e) (n_gap = 1 for air, the "
                     "cement's index for a cemented prism; no upper limit when n_gap >= n_e); L = a tan(A_g) + gap / cos(A_g); the "
                     "rejected wave leaves prism A through its side face at local +X",
    }
    if cement is None:
        design["air_gap_mm"] = text(gap)
    else:
        design.update(cement_index=text(n_gap, RATIO_QUANTUM), cement_thickness_mm=text(gap))
    class_id = "polarizer.glan" if cement is None else "polarizer.glan_thompson"
    used = [material.material] if cement is None else [material.material, cement.material]
    return finish(class_id, model, members=tuple(models), materials=used, coatings=[coating] if coating else [],
                  role="polarizer", values=values, design=design)


# ------------------------------------------------------------------------------------------------ Wollaston, Rochon


def _snell_deg(n_in: Decimal, n_out: Decimal, aoi_deg: Decimal, parameter: str) -> Decimal:
    """The refraction angle (deg, signed like ``aoi_deg``) from ``n_in`` into ``n_out``; total internal reflection is
    infeasible."""

    with localcontext(WORK_CONTEXT):
        sine = n_in / n_out * sin_cos_deg(aoi_deg)[0]
        if abs(sine) >= ONE:
            raise infeasible(f"the beam is totally reflected (n {text(n_in, RATIO_QUANTUM)} -> {text(n_out, RATIO_QUANTUM)} at "
                             f"{text(aoi_deg, ANGLE_QUANTUM_DEG)} deg): reduce the wedge angle", parameter)
        return degrees(asin(sine)) if sine >= 0 else -degrees(asin(-sine))


def _exit_angle_deg(n_a: Decimal, n_b: Decimal, theta: Decimal) -> Decimal:
    """The angle (deg from +Z, positive toward local +X) of the axial beam behind a split prism whose wave has index ``n_a`` in
    prism A and ``n_b`` in prism B: refraction at the joint (normal at ``theta`` to +Z toward -X, incidence angle ``theta``),
    then at the exit face (normal +Z). A cement layer between parallel faces changes nothing in it (n sin is kept)."""

    inside = _snell_deg(n_a, n_b, theta, "wedge_angle_deg")  # from the joint normal
    with localcontext(WORK_CONTEXT):
        tilt = inside - theta                                   # from +Z, positive toward +X
    return _snell_deg(n_b, ONE, tilt, "wedge_angle_deg")


def build_wollaston(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    return _split_prism(values, context, "WOLLASTON")


def build_rochon(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    return _split_prism(values, context, "ROCHON")


def _split_prism(values: Mapping[str, Any], context: BuildContext, kind: str) -> NativeBuild:
    """``splitter.wollaston`` (``kind`` ``WOLLASTON``) and ``splitter.rochon`` (``ROCHON``): module docstring."""

    material = context.resolver.resolve_crystal(values["material"], "material")
    cement = context.resolver.resolve(values["cement"], "cement") if values.get("cement") is not None else None
    wavelength = values["design_wavelength_nm"]
    theta = values["wedge_angle_deg"]
    a = values["aperture_mm"]
    fraction = values.get("clear_aperture_fraction") or DEFAULT_CLEAR_FRACTION
    edge = values.get("edge_thickness_mm") or quantize(a * DEFAULT_EDGE_FRACTION, LENGTH_QUANTUM_MM)
    layer = (values.get("cement_thickness_mm") or DEFAULT_CEMENT_MM) if cement is not None else ZERO
    n_o, n_e = _indices(material, wavelength)
    if n_o == n_e:
        raise infeasible(f"the crystal has no birefringence at {text(wavelength)} nm (n_o = n_e): it splits nothing", "material")
    coating, low, high = _coating(values, material, wavelength, context)
    if cement is not None:
        low, high = band_intersection([(low, high), cement.band], "cement")
    # the axial beam's waves (polarization, index in prism A, index in prism B): Wollaston X-polarized e in A -> o in B,
    # Y-polarized o -> e; Rochon: o in A (its axis along the beam) -> o (X) or e (Y) in B
    pairs = {"WOLLASTON": (("X", n_e, n_o), ("Y", n_o, n_e)), "ROCHON": (("X", n_o, n_o), ("Y", n_o, n_e))}[kind]
    exits = {pol: _exit_angle_deg(n_a, n_b, theta) for pol, n_a, n_b in pairs}
    with localcontext(WORK_CONTEXT):
        s, c = sin_cos_deg(theta)
        t = s / c
        h = a * HALF
        rise = a * t
        centre_z = edge + h * t                       # the joint plane's centre on prism A's side (unit frame)
        shift = layer / c                             # the cement layer along Z
        total = rise + TWO * edge + shift
        exit_z = edge + h * t                         # prism B's exit face in its own frame
        m = (-s, ZERO, c)                             # joint normal, out of prism A
        u = (c, ZERO, s)
        clear, clear_slant = fraction * h, fraction * h / c
        separation = abs(exits["X"] - exits["Y"])
    crystal = material.material_id
    axis_a, axis_b = SPLIT_PRISM_AXES[kind]
    unit_id = (values.get("model_id") or default_model_id(kind.lower()[:4], crystal, "a", a))[:MAX_UNIT_ID]
    centre_a = (ZERO, ZERO, centre_z)
    surfaces_a = [_posed("S1", (ZERO, ZERO, ZERO), (ZERO, ZERO, -ONE), (ONE, ZERO, ZERO), clear, clear, crystal, coating),
                  _posed("S2", centre_a, m, u, clear_slant, clear, crystal, None)]
    body_a = {"medium": crystal, "faces": [_face((ZERO, ZERO, ZERO), (ZERO, ZERO, -ONE)), _face((h, ZERO, ZERO), (ONE, ZERO, ZERO)),
                                           _face((-h, ZERO, ZERO), (-ONE, ZERO, ZERO)), _face((ZERO, h, ZERO), (ZERO, ONE, ZERO)),
                                           _face((ZERO, -h, ZERO), (ZERO, -ONE, ZERO)), _face(centre_a, m)]}
    surfaces_b = [_posed("S1", (ZERO, ZERO, ZERO), tuple(-v for v in m), u, clear_slant, clear, crystal, None),
                  _posed("S2", (ZERO, ZERO, exit_z), (ZERO, ZERO, ONE), (ONE, ZERO, ZERO), clear, clear, crystal, coating)]
    body_b = {"medium": crystal, "faces": [_face((ZERO, ZERO, ZERO), tuple(-v for v in m)), _face((ZERO, ZERO, exit_z), (ZERO, ZERO, ONE)),
                                           _face((h, ZERO, ZERO), (ONE, ZERO, ZERO)), _face((-h, ZERO, ZERO), (-ONE, ZERO, ZERO)),
                                           _face((ZERO, h, ZERO), (ZERO, ONE, ZERO)), _face((ZERO, -h, ZERO), (ZERO, -ONE, ZERO))]}
    band_doc = band(low, high, FactStatus.DERIVED if values.get("valid_band_nm") is None else FactStatus.DESIGN_BASIS)
    title = "Wollaston" if kind == "WOLLASTON" else "Rochon"
    members, models = [], []
    for member_id, surfaces, body, position, axis in (("A", surfaces_a, body_a, (ZERO, ZERO, ZERO), axis_a),
                                                      ("B", surfaces_b, body_b, (ZERO, ZERO, centre_z + shift), axis_b)):
        physics = {"model_kind": "SURFACE_SET", "model_version": "1", "fidelity": ModelFidelity.SURFACE_PRESCRIPTION.value,
                   "surfaces": surfaces, "body": body, "valid_band_nm": band_doc,
                   "approximations": _approximations(material, coating, True), "status": FactStatus.DERIVED.value}
        model_id = f"{unit_id}-{member_id}"
        models.append(optical_model(model_id, f"{title} prism {member_id} {crystal}", physics, parameter="wedge_angle_deg",
                                    optic_axes=[{"material_id": crystal, "direction": list(axis)}]))
        members.append({"member_id": member_id, "model_id": model_id,
                        "pose": {"position_mm": [float(quantize(v, LENGTH_QUANTUM_MM)) for v in position], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}})
    joint: dict[str, Any] = {"member_a": "A", "surface_a": "S2", "member_b": "B", "surface_b": "S1"}
    if cement is not None:
        joint["cement_material_id"] = cement.material_id
    ports = [{"port_id": "IN", "role": "INPUT", "point_mm": ["0", "0", "0"], "direction": ["0", "0", "1"], "status": FactStatus.DESIGN_BASIS.value},
             {"port_id": "OUT", "role": "OUTPUT", "point_mm": ["0", "0", length(total)], "direction": ["0", "0", "1"],
              "status": FactStatus.DERIVED.value}]
    physics = {"model_kind": "COMPOUND", "model_version": "1", "members": members, "joints": [joint],
               "envelope": box(h, h, ZERO, ceil_to(total, LENGTH_QUANTUM_MM)), "ports": ports}
    model = optical_model(unit_id, values.get("display_name") or f"{title} prism {crystal} {text(a)} mm", physics)
    design = {
        "design_wavelength_nm": text(wavelength), "ordinary_index": text(n_o, RATIO_QUANTUM), "extraordinary_index": text(n_e, RATIO_QUANTUM),
        "wedge_angle_deg": text(theta), "aperture_mm": text(a), "edge_thickness_mm": text(edge), "length_mm": length(total),
        "optic_axis_prism_a": list(axis_a), "optic_axis_prism_b": list(axis_b),
        "exit_angle_x_polarized_deg": text(exits["X"], ANGLE_QUANTUM_DEG), "exit_angle_y_polarized_deg": text(exits["Y"], ANGLE_QUANTUM_DEG),
        "separation_deg": text(separation, ANGLE_QUANTUM_DEG),
        "equations": "exit angle phi from local +Z (positive toward +X): n_A sin(theta) = n_B sin(theta - phi_B) at the joint and "
                     "n_B sin(phi_B) = sin(phi) at the exit face, with each polarization's index in prism A and in prism B; the "
                     "separation is close to " + ("2 asin((n_o - n_e) tan(theta))" if kind == "WOLLASTON" else "asin((n_o - n_e) tan(theta))"),
    }
    if cement is not None:
        design["cement_thickness_mm"] = text(layer)
    used = [material.material] if cement is None else [material.material, cement.material]
    class_id = "splitter.wollaston" if kind == "WOLLASTON" else "splitter.rochon"
    return finish(class_id, model, members=tuple(models), materials=used, coatings=[coating] if coating else [], role="splitter",
                  values=values, design=design)


__all__ = ["DEFAULT_AIR_GAP_MM", "GLAN_VARIANTS", "SPLIT_PRISM_AXES", "build_beam_displacer", "build_glan", "build_glan_thompson",
           "build_rochon", "build_wollaston", "walk_off_deg"]
