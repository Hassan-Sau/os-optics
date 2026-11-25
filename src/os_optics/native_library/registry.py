"""The registry of native component classes: what each class is, what it takes and what it produces.

Every entry is a ``NativeClass``: its id (``<category>.<name>``), category, title and description, the ``optical_role``
its component takes, the model kind it produces, its parameters (``ParameterSpec``), the physics it implements, the
approximations of the produced model and textbook references. ``build`` parses parameters and calls the class's
builder; nothing else creates native models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import partial
from typing import Any

from .common import BuildContext, NativeBuild
from .compound import build_relay, build_telescope
from .crystals import GLAN_VARIANTS, build_beam_displacer, build_glan, build_glan_thompson, build_rochon, build_wollaston
from .errors import NativeComponentError, NativeErrorCode, NativeProblem
from .lenses import FRAUNHOFER_FDC_NM, REAR_CONICS, SHAPE_FORMS, build_achromat, build_ideal_thin, build_singlet
from .numbers import BUILD_CONTEXT, text
from .params import ParameterSpec as P
from .params import ParameterType as T
from .params import parse_parameters
from .plates import MIRROR_KINDS, build_curved_mirror, build_flat_mirror, build_layered, build_plate_splitter, build_window
from .polarization import RETARDANCE_DISPERSIONS, build_linear_polarizer, build_retarder
from .refractors import AXICON_ORIENTATIONS, build_axicon, build_grin_rod, build_wedge
from .responses import (
    AOD_ORDERS,
    build_acousto_optic,
    FARADAY_INTERPOLATIONS,
    FOCAL_INTERPOLATIONS,
    GRATING_MODES,
    ND_LAWS,
    TRANSMITTANCE_INTERPOLATIONS,
    build_faraday_rotator,
    build_grating,
    build_tabulated_thin,
    build_variable_nd,
)
from .retroreflectors import build_hollow_corner, build_hollow_roof
from .stops import build_annular, build_central_obstruction, build_circular, build_slit

Builder = Callable[[Mapping[str, Any], BuildContext], NativeBuild]
POSITIVE = Decimal(0)
UNIT = Decimal(1)
RIGHT_ANGLE_DEG = Decimal(90)
#: The largest wedge angle of a wedge prism (beyond it the class is a general prism).
HALF_RIGHT_ANGLE_DEG = Decimal(45)
FULL_TURN_DEG = Decimal(360)
#: Default band of an ideal thin lens (it has no dispersion; the band only states where it is used).
IDEAL_LENS_BAND_NM = ("200", "20000")
LENS_CLASSES = ("lens.ideal_thin", "lens.singlet", "lens.achromat")

REF_HECHT = "E. Hecht, Optics, 5th ed. (Pearson, 2017)"
REF_BORN_WOLF = "M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999)"
REF_KINGSLAKE = "R. Kingslake and R. B. Johnson, Lens Design Fundamentals, 2nd ed. (Academic Press, 2010)"
REF_SMITH = "W. J. Smith, Modern Optical Engineering, 4th ed. (McGraw-Hill, 2008)"
REF_GOODMAN = "J. W. Goodman, Introduction to Fourier Optics, 3rd ed. (Roberts & Co., 2005)"
REF_MCLEOD = "J. H. McLeod, The axicon: a new type of optical element, J. Opt. Soc. Am. 44, 592-597 (1954)"
REF_DURNIN = "J. Durnin, J. J. Miceli and J. H. Eberly, Diffraction-free beams, Phys. Rev. Lett. 58, 1499-1501 (1987)"
REF_ECKHARDT = "H. D. Eckhardt, Simple model of corner reflector phenomena, Appl. Opt. 10, 1559-1566 (1971)"
REF_SALEH_TEICH = "B. E. A. Saleh and M. C. Teich, Fundamentals of Photonics, 3rd ed. (Wiley, 2019)"
REF_LOEWEN_POPOV = "E. G. Loewen and E. Popov, Diffraction Gratings and Applications (Marcel Dekker, 1997)"
REF_MARCHAND = "E. W. Marchand, Gradient Index Optics (Academic Press, 1978)"
REF_BENNETT = "J. M. Bennett, Polarizers, in Handbook of Optics, vol. I, 2nd ed., ch. 3 (McGraw-Hill, 1995)"
REF_ISO_ND = "ISO 5-2:2009, Photography and graphic technology - Density measurements, Part 2: Geometric conditions for transmittance density"

#: How a native component is available (``describe()["availability"]``): it simulates on this device, and a component using it
#: is the customer's own optic (a ``custom_part``). A native class is never an orderable product.
AVAILABILITY = ("LOCAL_SIMULATION", "CUSTOMER_SUPPLIED")
LENS_FORMS = ("PLANO_CONVEX_LENS", "BICONVEX_LENS", "PLANO_CONCAVE_LENS", "BICONCAVE_LENS", "POSITIVE_MENISCUS_LENS",
              "NEGATIVE_MENISCUS_LENS", "BEST_FORM_LENS", "ASPHERIC_LENS", "CYLINDRICAL_LENS", "ACHROMATIC_DOUBLET")
LENS_MATCH = ("focal_length_mm", "design_wavelength_nm", "outer_diameter_mm", "clear_aperture_diameter_mm")


def _equivalent(category: str, forms: tuple[str, ...], match: tuple[str, ...]) -> dict[str, Any]:
    form = forms[0] if len(forms) == 1 else "FORM"
    return {"category": category, "forms": list(forms), "match": list(match),
            "query": f"os-optics catalog list --category {category} --form {form} --wavelength-nm WAVELENGTH"}


#: Per class, the public catalog products that play the same part (``describe()["catalog_equivalents"]``): the category and forms
#: to list (``catalog list --category C --form F --wavelength-nm W``; one form per call) and the summary-row values to compare.
#: Empty: the public catalog has no product of this kind. This is a search aid for comparable catalog products, not a claim that
#: one is equivalent to the parameters chosen here.
CATALOG_EQUIVALENTS: dict[str, tuple[dict[str, Any], ...]] = {
    "lens.ideal_thin": (_equivalent("LENS", LENS_FORMS, ("focal_length_mm", "outer_diameter_mm", "clear_aperture_diameter_mm")),),
    "lens.singlet": (_equivalent("LENS", LENS_FORMS[:-1], LENS_MATCH),),
    "lens.achromat": (_equivalent("LENS", ("ACHROMATIC_DOUBLET",), LENS_MATCH),),
    "plate.window": (_equivalent("WINDOW", ("PLANE_WINDOW",), ("thickness_mm", "outer_diameter_mm", "outer_dimensions_mm")),),
    "mirror.flat": (_equivalent("MIRROR", ("FLAT_MIRROR",), ("outer_diameter_mm", "reflectance")),),
    "mirror.curved": (_equivalent("MIRROR", ("CONCAVE_MIRROR",), ("focal_length_mm", "outer_diameter_mm", "reflectance")),),
    "splitter.plate": (_equivalent("BEAMSPLITTER", ("PLATE_BEAMSPLITTER", "POLARIZING_PLATE_BEAMSPLITTER", "BEAM_SAMPLER"),
                                   ("reflectance", "transmittance", "outer_diameter_mm")),),
    "polarizer.linear": (_equivalent("POLARIZER", ("LINEAR_POLARIZER",), ("transmittance", "extinction_ratio", "outer_diameter_mm")),),
    "waveplate.retarder": (_equivalent("WAVEPLATE", ("HALF_WAVE_PLATE", "QUARTER_WAVE_PLATE"),
                                       ("retardance_waves", "design_wavelength_nm", "outer_diameter_mm")),),
    "lens.axicon": (_equivalent("LENS", ("AXICON",), ("cone_angle_deg", "outer_diameter_mm", "clear_aperture_diameter_mm")),),
    "prism.wedge": (_equivalent("PRISM", ("WEDGE_PRISM",), ("wedge_angle_deg", "deviation_angle_deg", "outer_diameter_mm")),),
    "filter.variable_nd": (_equivalent("FILTER", ("NEUTRAL_DENSITY_FILTER",), ("optical_density", "outer_diameter_mm",
                                                                             "clear_aperture_diameter_mm")),),
    # a right-angle prism used through its hypotenuse returns a beam by two total internal reflections, reversed in one axis
    "mirror.hollow_roof": (_equivalent("PRISM", ("RIGHT_ANGLE_PRISM",), ("outer_dimensions_mm", "clear_aperture_dimensions_mm")),),
}


@dataclass(frozen=True)
class NativeClass:
    class_id: str
    category: str
    title: str
    description: str
    optical_role: str
    model_kind: str
    parameters: tuple[P, ...]
    physics: str
    approximations: tuple[str, ...]
    references: tuple[str, ...]
    keywords: tuple[str, ...]
    builder: Builder

    def build_with(self, parameters: Mapping[str, Any] | None, context: BuildContext) -> NativeBuild:
        """Parse ``parameters`` (every problem reported together) and build (in ``BUILD_CONTEXT``: the result never depends
        on the caller's decimal context)."""

        with localcontext(BUILD_CONTEXT):
            return self.builder(parse_parameters(self.parameters, parameters), context)

    def describe(self) -> dict[str, Any]:
        with localcontext(BUILD_CONTEXT):
            return self._describe()

    def _describe(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id, "category": self.category, "title": self.title, "description": self.description,
            "optical_role": self.optical_role, "model_kind": self.model_kind, "keywords": list(self.keywords),
            "parameters": [p.describe() for p in self.parameters], "physics": self.physics,
            "approximations": list(self.approximations), "references": list(self.references),
            "availability": list(AVAILABILITY),
            "catalog_equivalents": [dict(e, forms=list(e["forms"]), match=list(e["match"])) for e in CATALOG_EQUIVALENTS.get(self.class_id, ())],
        }


# ------------------------------------------------------------------------------------------------- parameters

COMMON = (
    P("model_id", T.MODEL_ID, "id of the produced optical model in the Project", default_rule="derived from the class and its key values"),
    P("display_name", T.TEXT, "name shown for the model", default_rule="derived from the class and its key values"),
    P("instance_id", T.INSTANCE_ID, "instance id of the component in component_template", default_rule="the model id"),
)


def _d(name: str, description: str, unit: str | None = "mm", **kw) -> P:
    return P(name, T.DECIMAL, description, unit=unit, **kw)


DIAMETER = _d("diameter_mm", "outer diameter of the element", required=True, exclusive_minimum=POSITIVE)
CLEAR = _d("clear_aperture_diameter_mm", "clear aperture diameter (between it and the outer diameter light is blocked)",
           default_rule="0.9 x diameter_mm", exclusive_minimum=POSITIVE)
BAND_REQUIRED = P("valid_band_nm", T.BAND, "vacuum wavelengths the declared values hold for: [min_nm, max_nm]", unit="nm", required=True,
                  exclusive_minimum=POSITIVE)
BAND_OPTIONAL = P("valid_band_nm", T.BAND, "vacuum wavelengths the model is valid for: [min_nm, max_nm]", unit="nm",
                  default_rule="the common valid band of the materials (and coating)", exclusive_minimum=POSITIVE)
AOI = P("valid_aoi_deg", T.PAIR, "angle-of-incidence range the declared values hold for: [min_deg, max_deg]", unit="deg",
        default_rule="[0, 90]: the same values at every angle", minimum=POSITIVE, maximum=RIGHT_ANGLE_DEG)
THIN_THICKNESS = _d("thickness_mm", "drawn thickness of the thin element (its outline; the element acts at its plane z = 0)",
                    default="1", exclusive_minimum=POSITIVE)
AR_COATING = P("ar_coating", T.COATING, 'anti-reflection coating of the outer faces: a public catalog coating id, {"reflectance": R, '
               '"valid_aoi_deg": [a, b]} (declared, lossless) or {"angular_spectrum": {"wavelength_nm": [...], "aoi_deg": [...], '
               '"reflectance_s" / "reflectance_p" / "transmittance_s" / "transmittance_p": [[per angle] per wavelength]}} (declared, '
               'over wavelength and angle of incidence, read in the incident medium)', default_rule="uncoated (bare Fresnel interfaces)")
CRYSTAL_HELP = ('a uniaxial crystal: {"ordinary": INDEX, "extraordinary": INDEX, "valid_band_nm": [a, b]} with INDEX '
                '{"constant_index": n} or {"sellmeier": {"b": [..3], "c_um2": [..3]}} (optional: material_id, display_name, '
                'absorption_coefficient_per_mm, basis), or a ProjectModelMaterialV1 document with birefringence')
MATERIAL_HELP = ('a public catalog material id (e.g. "N-BK7", copied from the catalog), {"constant_index": n, "valid_band_nm": [a, b]}, '
                 '{"sellmeier": {"b": [..3], "c_um2": [..3]}, "valid_band_nm": [a, b]} (optional: material_id, display_name, '
                 'absorption_coefficient_per_mm, basis) or a full ProjectModelMaterialV1 document')
EDGE_MIN = _d("min_edge_thickness_mm", "smallest edge thickness the thickening rule allows", default_rule="max(0.5 mm, 0.05 x diameter_mm)",
              exclusive_minimum=POSITIVE)
CENTER_MIN = _d("min_center_thickness_mm", "smallest centre thickness the thickening rule allows",
                default_rule="max(1 mm, 0.08 x diameter_mm)", exclusive_minimum=POSITIVE)
FRACTION = _d("clear_aperture_fraction", "clear aperture as a fraction of the outline (each dimension)", unit=None, default="0.9",
              exclusive_minimum=POSITIVE, maximum=UNIT)
ROUND_OR_RECT = (
    _d("diameter_mm", "outer diameter of a round plate (or give width_mm and height_mm)", exclusive_minimum=POSITIVE),
    _d("width_mm", "outer width (local X) of a rectangular plate", exclusive_minimum=POSITIVE),
    _d("height_mm", "outer height (local Y) of a rectangular plate", exclusive_minimum=POSITIVE),
    FRACTION,
)
REFLECTANCE = (
    _d("reflectance", "power reflectance R (unpolarized)", unit=None, minimum=POSITIVE, maximum=UNIT),
    _d("transmittance", "power transmittance T (R + T <= 1)", unit=None, minimum=POSITIVE, maximum=UNIT,
       default_rule="0 for a metallic mirror, 1 - R for a dielectric mirror or a splitter"),
    _d("reflectance_s", "s-polarized power reflectance (with reflectance_p, instead of reflectance)", unit=None, minimum=POSITIVE, maximum=UNIT),
    _d("reflectance_p", "p-polarized power reflectance", unit=None, minimum=POSITIVE, maximum=UNIT),
    _d("transmittance_s", "s-polarized power transmittance", unit=None, minimum=POSITIVE, maximum=UNIT, default_rule="as transmittance"),
    _d("transmittance_p", "p-polarized power transmittance", unit=None, minimum=POSITIVE, maximum=UNIT, default_rule="as transmittance"),
)
PLATE_DIAMETER = _d("plate_diameter_mm", "outer diameter of the opaque plate (light beyond it passes by)",
                    default_rule="max(25.4 mm, twice the opening's reach), rounded up to 0.1 mm", exclusive_minimum=POSITIVE)
CENTER = P("opening_center_mm", T.PAIR, "decentre of the opening (local X, Y)", unit="mm", default_rule="[0, 0]")
LENS_SPEC = ('{"class_id": "lens.ideal_thin" | "lens.singlet" | "lens.achromat", "parameters": {...}} (the class\'s own parameters)')


def _lens(name: str, description: str) -> P:
    return P(name, T.LENS, f"{description}: {LENS_SPEC}", required=True, lens_classes=LENS_CLASSES)


SEAM = _d("seam_mm", "gap between neighbouring mirror faces along their common edge (light reaching it passes between them)",
          default="0.01", exclusive_minimum=POSITIVE)
SUBSTRATE = _d("substrate_thickness_mm", "drawn substrate thickness of each mirror behind its face",
               default_rule="max(1 mm, a tenth of the face's longer side), rounded up to 0.1 mm", exclusive_minimum=POSITIVE)
CENTER_THICKNESS = _d("center_thickness_mm", "thickness on the axis (the axicon's apex / the wedge's centre)",
                      default_rule="the thin edge at least max(0.5 mm, 0.05 x diameter_mm), rounded up to 1 um", exclusive_minimum=POSITIVE)
COMPOUND_WAVELENGTH = _d("design_wavelength_nm", "wavelength at which the lenses are spaced (paraxial focal lengths)", unit="nm",
                         default_rule="the lenses' own design wavelength (needed only for refracting lenses)", exclusive_minimum=POSITIVE)


# ------------------------------------------------------------------------------------------------- the classes


def _class(class_id: str, category: str, title: str, description: str, role: str, kind: str, parameters: tuple[P, ...], physics: str,
           approximations: tuple[str, ...], references: tuple[str, ...], keywords: tuple[str, ...], builder: Builder) -> NativeClass:
    return NativeClass(class_id, category, title, description, role, kind, (*parameters, *COMMON), physics, approximations, references,
                       keywords, builder)


STOP_APPROXIMATIONS = ("geometric stop: diffraction at the opening is not modelled", "infinitely thin opaque plate (drawn with a thickness)")
THIN_JONES = ("ideal thin Jones element: plate thickness and refraction ignored (THIN_ELEMENT)",)
REFRACTIVE = ("nominal prescription, ideal surface figure", "light between the clear aperture and the edge is blocked (EDGE_AS_OPAQUE_RIM)",
              "residual (ghost) reflections of transmitting surfaces are not followed")


def _build_nested(registry: dict[str, NativeClass], spec: Mapping[str, Any], model_id: str, context: BuildContext, parameter: str) -> NativeBuild:
    parameters = dict(spec.get("parameters", {}))
    parameters.setdefault("model_id", model_id)
    try:
        return registry[spec["class_id"]].build_with(parameters, context)
    except NativeComponentError as exc:
        raise NativeComponentError([NativeProblem(p.code, p.message, f"{parameter}.{p.parameter}" if p.parameter else parameter)
                                    for p in exc.problems]) from None


def _registry() -> dict[str, NativeClass]:
    classes: dict[str, NativeClass] = {}
    nested = partial(_build_nested, classes)
    entries = [
        _class("aperture.circular", "STOP", "Circular aperture (iris, stop, pinhole)",
               "An opaque plate with a round opening: an iris, an aperture or field stop, a pinhole.", "aperture", "APERTURE_STOP",
               (_d("opening_diameter_mm", "diameter of the opening", required=True, exclusive_minimum=POSITIVE), CENTER, PLATE_DIAMETER, THIN_THICKNESS),
               "Rays crossing z = 0 inside the opening pass unchanged; rays on the plate are stopped; rays beyond the plate pass by.",
               STOP_APPROXIMATIONS, (REF_HECHT,), ("iris", "pinhole", "aperture stop", "field stop", "diaphragm"), build_circular),
        _class("aperture.slit", "STOP", "Slit", "An opaque plate with a rectangular opening (an entrance slit, a line aperture).",
               "aperture", "APERTURE_STOP",
               (_d("slit_width_mm", "opening width along local X", required=True, exclusive_minimum=POSITIVE),
                _d("slit_height_mm", "opening height along local Y", required=True, exclusive_minimum=POSITIVE), CENTER, PLATE_DIAMETER, THIN_THICKNESS),
               "Rays crossing z = 0 inside the rectangle pass; rotate the component about local Z to turn the slit.",
               STOP_APPROXIMATIONS, (REF_HECHT,), ("slit", "rectangular aperture", "spectrometer slit"), build_slit),
        _class("aperture.annular", "STOP", "Annular aperture", "A round opening with an opaque central disc: an annular pupil, a centrally "
               "obstructed aperture.", "aperture", "APERTURE_STOP",
               (_d("opening_diameter_mm", "outer diameter of the open ring", required=True, exclusive_minimum=POSITIVE),
                _d("obstruction_diameter_mm", "diameter of the opaque central disc", required=True, exclusive_minimum=POSITIVE), PLATE_DIAMETER,
                THIN_THICKNESS),
               "Rays crossing z = 0 between the disc and the opening's edge pass; the transmitted ring continues.",
               STOP_APPROXIMATIONS, (REF_BORN_WOLF,), ("annulus", "central obstruction", "annular pupil", "ring aperture"), build_annular),
        _class("aperture.central_obstruction", "STOP", "Central obstruction (opaque disc)",
               "An opaque disc in a clear carrier: a beam block, a dark-field or Fourier-plane DC stop, a secondary-mirror obstruction.",
               "aperture", "APERTURE_STOP",
               (_d("obstruction_diameter_mm", "diameter of the opaque disc", required=True, exclusive_minimum=POSITIVE),
                _d("carrier_diameter_mm", "diameter of the clear carrier holding the disc (light inside it and beyond it passes)",
                   default_rule="max(25.4 mm, twice the disc), rounded up to 0.1 mm", exclusive_minimum=POSITIVE), THIN_THICKNESS),
               "Rays crossing z = 0 on the disc are stopped; all other rays pass; the carrier itself does not refract.",
               (*STOP_APPROXIMATIONS, "the carrier is not modelled (no refraction, no support vanes)"), (REF_GOODMAN,),
               ("beam block", "obscuration", "dark field stop", "DC block", "central stop"), build_central_obstruction),
        _class("lens.ideal_thin", "LENS", "Ideal thin lens", "A perfect paraxial lens of given focal length at one plane.", "lens", "IDEAL_THIN_LENS",
               (_d("focal_length_mm", "focal length (negative: diverging)", required=True, nonzero=True),
                _d("clear_aperture_diameter_mm", "clear aperture diameter", required=True, exclusive_minimum=POSITIVE),
                _d("diameter_mm", "outer diameter (light between it and the clear aperture is blocked)", default_rule="the clear aperture",
                   exclusive_minimum=POSITIVE),
                _d("transmittance", "power transmittance", unit=None, default="1", exclusive_minimum=POSITIVE, maximum=UNIT),
                P("valid_band_nm", T.BAND, "wavelengths the lens is used at", unit="nm", default=list(IDEAL_LENS_BAND_NM), exclusive_minimum=POSITIVE),
                THIN_THICKNESS),
               "Every ray is deflected as by a perfect lens (u' = u - y / f): no thickness, aberration, dispersion or optical-path phase.",
               ("ideal paraxial element: no aberrations, no dispersion", "phase and interference through it are not defined"),
               (REF_HECHT,), ("thin lens", "paraxial lens", "ideal lens"), build_ideal_thin),
        _class("lens.singlet", "LENS", "Singlet lens", "A spherical (optionally conic or cylindrical) singlet of one material designed for "
               "a focal length at a design wavelength.", "lens", "SURFACE_SEQUENCE",
               (_d("focal_length_mm", "paraxial focal length at the design wavelength (negative: diverging)", required=True, nonzero=True),
                _d("design_wavelength_nm", "wavelength the focal length holds at", unit="nm", required=True, exclusive_minimum=POSITIVE),
                P("material", T.MATERIAL, MATERIAL_HELP, required=True), DIAMETER, CLEAR,
                P("form", T.ENUM, "shape: SYMMETRIC (X = 0), PLANO_EXIT (X = +1, curved face toward the light), PLANO_ENTRY (X = -1), "
                  "BEST_FORM (minimum spherical aberration for an object at infinity)", choices=SHAPE_FORMS,
                  default_rule="SYMMETRIC (or shape_factor)"),
                _d("shape_factor", "Coddington shape factor X = (c1 + c2) / (c1 - c2), instead of form", unit=None),
                _d("center_thickness_mm", "centre thickness", default_rule="the thickening rule", exclusive_minimum=POSITIVE), CENTER_MIN, EDGE_MIN,
                P("rear_conic", T.ENUM, "STIGMATIC_COLLIMATED: exit surface k = -n^2, a perfect focus of a collimated on-axis beam "
                  "(needs form PLANO_ENTRY)", choices=REAR_CONICS, default="NONE"),
                P("cylinder_axis", T.ENUM, "make both curved surfaces cylinders straight along this local axis", choices=("X", "Y")),
                AR_COATING, BAND_OPTIONAL),
               "Thick-lens power 1/f = (n-1)(c1-c2) + (n-1)^2 t c1 c2 / n solved exactly for the curvatures at the given shape factor "
               "and thickness; thickening rule for the centre thickness; paraxial block recomputed from the written prescription.",
               REFRACTIVE, (REF_HECHT, REF_KINGSLAKE, REF_BORN_WOLF), ("lens", "plano-convex", "biconvex", "plano-concave", "meniscus",
                                                                       "best form", "asphere", "cylindrical lens"), build_singlet),
        _class("lens.achromat", "LENS", "Cemented achromatic doublet", "A crown / flint cemented doublet that brings a collimated beam of a "
               "short and a long wavelength to the same focus (equal back focal lengths).", "lens", "SURFACE_SEQUENCE",
               (_d("focal_length_mm", "paraxial focal length at the centre wavelength", required=True, nonzero=True),
                P("crown_material", T.MATERIAL, f"first (crown) element material: {MATERIAL_HELP}", required=True),
                P("flint_material", T.MATERIAL, "second (flint) element material, same forms", required=True),
                P("wavelengths_nm", T.TRIPLE, "short, centre and long design wavelengths", unit="nm",
                  default=[text(w) for w in FRAUNHOFER_FDC_NM], exclusive_minimum=POSITIVE),
                DIAMETER, CLEAR,
                _d("crown_shape_factor", "Coddington shape factor of the crown element", unit=None, default="0"),
                _d("crown_center_thickness_mm", "crown centre thickness", default_rule="the thickening rule", exclusive_minimum=POSITIVE),
                _d("flint_center_thickness_mm", "flint centre thickness", default_rule="the thickening rule", exclusive_minimum=POSITIVE),
                CENTER_MIN, EDGE_MIN, AR_COATING, BAND_OPTIONAL),
               "Thin-lens achromat phi1 = phi V1/(V1-V2), phi2 = -phi V2/(V1-V2), then the two element powers re-solved exactly on the "
               "thick prescription: EFL(centre) = f and BFL(short) = BFL(long) (coincident foci).",
               (*REFRACTIVE, "the cement layer is not modelled"), (REF_KINGSLAKE, REF_SMITH), ("achromat", "doublet", "cemented doublet",
                                                                                              "color corrected lens"), build_achromat),
        _class("plate.window", "PLATE", "Window / plane-parallel plate", "A plane-parallel plate of one material: a window, a substrate, a "
               "sample slab of declared index and absorption.", "window", "SURFACE_SEQUENCE",
               (P("material", T.MATERIAL, MATERIAL_HELP, required=True), _d("thickness_mm", "plate thickness", required=True,
                                                                             exclusive_minimum=POSITIVE), *ROUND_OR_RECT, AR_COATING, BAND_OPTIONAL),
               "Two plane refracting faces: lateral shift and focus shift of a converging beam, dispersion, bulk absorption, Fresnel "
               "loss of uncoated faces.", REFRACTIVE, (REF_HECHT,), ("window", "plate", "substrate", "sample", "slab"), build_window),
        _class("plate.layered", "PLATE", "Layered plate (cuvette, plate stack)", "Plane-parallel layers of different materials in contact: "
               "a cuvette (wall / liquid / wall), a cemented plate stack.", "window", "SURFACE_SEQUENCE",
               (P("layers", T.LAYERS, f'the layers along +Z: [{{"material": <material>, "thickness_mm": t}}, ...]; material: {MATERIAL_HELP}',
                  required=True), *ROUND_OR_RECT, AR_COATING, BAND_OPTIONAL),
               "Plane interfaces between consecutive layers, each refracting between its two media.", REFRACTIVE, (REF_HECHT,),
               ("cuvette", "sample cell", "flow cell", "layer stack"), build_layered),
        _class("mirror.flat", "MIRROR", "Flat mirror", "A plane mirror with a declared reflectance.", "mirror", "SURFACE_SEQUENCE",
               (DIAMETER, CLEAR, *REFLECTANCE, P("coating_kind", T.ENUM, "reflector type", choices=MIRROR_KINDS, default="METALLIC_REFLECTOR"),
                BAND_REQUIRED, AOI, _d("substrate_thickness_mm", "drawn substrate thickness", default_rule="max(1 mm, diameter / 4), rounded "
                                       "up to 0.1 mm", exclusive_minimum=POSITIVE)),
               "Specular reflection with the declared power reflectance (per s / p when given).",
               ("declared reflectance, constant over its band", "coating phase not modelled", "transmitted light not followed"), (REF_HECHT,),
               ("mirror", "fold mirror", "reflector"), build_flat_mirror),
        _class("mirror.curved", "MIRROR", "Curved mirror", "A concave or convex spherical (or conic) mirror of given focal length.", "mirror",
               "SURFACE_SEQUENCE",
               (_d("focal_length_mm", "focal length f = R / 2 (positive: concave)", required=True, nonzero=True),
                _d("conic", "conic constant of the surface (0 sphere, -1 paraboloid)", unit=None, default="0"),
                DIAMETER, CLEAR, *REFLECTANCE, P("coating_kind", T.ENUM, "reflector type", choices=MIRROR_KINDS, default="METALLIC_REFLECTOR"),
                BAND_REQUIRED, AOI, _d("substrate_thickness_mm", "drawn substrate thickness at the vertex (a convex mirror's must exceed "
                                       "the depth of its face at the outer diameter)", default_rule="max(1 mm, diameter / 4), rounded "
                                       "up to 0.1 mm; a convex mirror adds the depth of its face at the outer diameter", exclusive_minimum=POSITIVE)),
               "Reflection at a conic surface of vertex radius R = 2 f; a paraboloid images an on-axis point at infinity perfectly.",
               ("declared reflectance, constant over its band", "coating phase not modelled", "transmitted light not followed"),
               (REF_HECHT, REF_BORN_WOLF), ("concave mirror", "convex mirror", "parabolic mirror", "focusing mirror"), build_curved_mirror),
        _class("splitter.plate", "SPLITTER", "Plate beamsplitter (ideal)", "One coated plane that reflects R and transmits T of the power "
               "(optionally per s / p): a non-polarizing or polarizing plate splitter.", "splitter", "SURFACE_SEQUENCE",
               (DIAMETER, CLEAR, *REFLECTANCE, BAND_REQUIRED, AOI, THIN_THICKNESS),
               "Both the reflected and the transmitted branch are traced with the declared power split; the substrate is not modelled.",
               ("substrate not modelled (no plate refraction, no ghost)", "coating phase not modelled"), (REF_HECHT,),
               ("beamsplitter", "beam splitter", "PBS", "partial reflector", "pickoff"), build_plate_splitter),
        _class("polarizer.linear", "POLARIZATION", "Linear polarizer", "A thin linear polarizer of declared transmittance and extinction ratio "
               "(transmission axis local +X).", "polarizer", "POLARIZATION_ELEMENT",
               (_d("transmittance", "power transmittance along the transmission axis", unit=None, required=True, exclusive_minimum=POSITIVE,
                   maximum=UNIT),
                _d("extinction_ratio", "ratio of the transmittances along and across the axis", unit=None, required=True, minimum=UNIT),
                DIAMETER, CLEAR, BAND_REQUIRED, AOI, THIN_THICKNESS),
               "Jones matrix diag(sqrt(T), sqrt(T / ER)) in the (axis, perpendicular) basis.", THIN_JONES, (REF_HECHT, REF_BORN_WOLF),
               ("polarizer", "analyzer", "linear polarizer"), build_linear_polarizer),
        _class("waveplate.retarder", "POLARIZATION", "Waveplate (retarder)", "A thin linear retarder of declared retardance (fast axis local +X): "
               "half-wave 0.5, quarter-wave 0.25.", "waveplate", "POLARIZATION_ELEMENT",
               (_d("retardance_waves", "retardance at the design wavelength, in waves", unit="waves", required=True, exclusive_minimum=POSITIVE),
                _d("design_wavelength_nm", "design wavelength", unit="nm", required=True, exclusive_minimum=POSITIVE),
                P("retardance_dispersion", T.ENUM, "ZERO_ORDER: delta(l) = delta0 l0 / l; ACHROMATIC: constant over the band",
                  choices=tuple(RETARDANCE_DISPERSIONS), default="ZERO_ORDER"),
                _d("transmittance", "power transmittance", unit=None, default="1", exclusive_minimum=POSITIVE, maximum=UNIT),
                DIAMETER, CLEAR, BAND_REQUIRED, AOI, THIN_THICKNESS),
               "Jones matrix sqrt(T) diag(1, exp(i 2 pi delta)) in the (fast, slow) basis.", THIN_JONES, (REF_HECHT, REF_BORN_WOLF),
               ("waveplate", "half-wave plate", "quarter-wave plate", "retarder", "HWP", "QWP"), build_retarder),
        _class("rotator.faraday", "POLARIZATION", "Faraday rotator (stated response)", "A thin nonreciprocal polarization rotator: the rotation "
               "and transmittance per wavelength are what you state, with the angles of incidence, operating conditions and evidence they "
               "hold for when you have them. Local +Z is the field axis. "
               "Between a polarizer and an analyzer it builds an optical isolator.", "other_optical", "FARADAY_ROTATOR",
               (P("rows", T.JSON, '[{"wavelength_nm", "rotation_deg", "transmittance"}, ...] in increasing wavelength, or columns '
                  '{"wavelength_nm": [...], "rotation_deg": [...], "transmittance": [...]}; rotation_deg is right-handed about local +Z',
                  required=True),
                P("interpolation", T.ENUM, "between rows: linear in wavelength over the rows' span, or only at the rows' exact wavelengths",
                  choices=FARADAY_INTERPOLATIONS, default="LINEAR"),
                DIAMETER, CLEAR,
                P("valid_aoi_deg", T.PAIR, "chief-ray angles of incidence the response holds for: [min_deg, max_deg]", unit="deg",
                  minimum=POSITIVE, maximum=RIGHT_ANGLE_DEG, default_rule="not stated: no angle of incidence is checked"),
                P("operating_conditions", T.TEXT, "the conditions the table holds for (temperature, magnet or field), as its source states them",
                  default_rule="not stated"),
                P("evidence", T.JSON, '{"method", "tools": [{"name", "version"}], "source_data_sha256", "configuration_sha256"?, '
                  '"fit_model", "validation_summary", "date": "YYYY-MM-DD"}', default_rule="not stated: an assumed response"),
                _d("length_mm", "drawn length of the rotator body along local Z, centred on its plane z = 0 (it acts at that plane)", default="20",
                   exclusive_minimum=POSITIVE)),
               "Jones matrix sqrt(T) [[cos b, -sin b], [sin b, cos b]] in the ray's (e1, e2) with e1 x e2 = d, b = sign(d . z) theta(lambda): the "
               "field turns the same way in the laboratory for both directions of travel, so a round trip adds 2 theta. Light of a "
               "wavelength outside the stated rows stops at the element, outside the model.",
               ("ideal thin element: the body's length and refractive index do not act on the light",
                "no optical path: phase, wavefront and the linear optical network through it are refused",
                "rotation independent of the angle of incidence (inside valid_aoi_deg when it is stated)",
                "the response holds only under its stated conditions"),
               (REF_SALEH_TEICH, REF_BORN_WOLF), ("Faraday rotator", "optical isolator", "nonreciprocal rotator", "Faraday mirror", "magneto-optic"),
               build_faraday_rotator),
        _class("grating.plane", "GRATING", "Diffraction grating (plane)", "A plane ruled grating, reflecting or transmitting, on a rectangular plate. "
               "Its period and groove direction give every order's direction, at any wavelength and angle of incidence; a stated response gives "
               "the orders' powers. Without a response each order is traced with its power not known and bounded. In a reflection "
               "grating local +Z is the outward normal of the ruled face; in a transmission grating light travels along +Z.", "other_optical", "DIFFRACTION_GRATING",
               (P("mode", T.ENUM, "reflection or transmission", choices=GRATING_MODES, required=True),
                _d("line_density_per_mm", "line density; state this or period_nm", unit="1/mm", exclusive_minimum=POSITIVE),
                _d("period_nm", "grating period; state this or line_density_per_mm", unit="nm", exclusive_minimum=POSITIVE),
                _d("grating_vector_angle_deg", "direction K across the grooves, from local +X toward local +Y; the grooves run along local +Z x K",
                   unit="deg", default="0"),
                P("orders", T.JSON, "the orders to follow: a list of integers, strictly increasing; state this or all_propagating_orders"),
                P("all_propagating_orders", T.JSON, "[first, last]: every order of this range that propagates; state this or orders"),
                _d("width_mm", "outer width along local X", required=True, exclusive_minimum=POSITIVE),
                _d("height_mm", "outer height along local Y", required=True, exclusive_minimum=POSITIVE),
                _d("clear_width_mm", "clear aperture width along local X", default_rule="0.9 width_mm", exclusive_minimum=POSITIVE),
                _d("clear_height_mm", "clear aperture height along local Y", default_rule="0.9 height_mm", exclusive_minimum=POSITIVE),
                _d("thickness_mm", "drawn plate thickness behind the ruled plane z = 0", default="6", exclusive_minimum=POSITIVE),
                P("response", T.JSON, 'the order response (GratingOrderResponseV1): {"normalization", "polarization_basis", "interpolation", "rows": '
                  '[{"wavelength_nm", "aoi_deg", "order", "efficiency_te", "efficiency_tm"}, ...], "absorption"?, "max_conical_angle_deg"?, "data_kind", '
                  '"basis", "evidence"?}')),
               "An order m leaves with the tangential part n_exit t_exit = n_in t_in + m (lambda / period) K; beyond n_exit it does not propagate. Each "
               "polarization's order powers are the stated efficiencies; the part of the incident power no followed order carries and no stated "
               "absorption takes is the element's unmodelled response.",
               ("direction: an infinite periodic plane; diffraction by the finite aperture and resolving power are not modelled",
                "power: only what the stated response gives; a response without TE and TM values bounds it",
                "no phase or optical path: wavefront, pulse dispersion and the linear optical network through it are refused",
                "thin element: the plate's thickness and index do not act on the light"),
               (REF_LOEWEN_POPOV, REF_BORN_WOLF), ("grating", "diffraction grating", "reflection grating", "transmission grating", "ruled grating", "holographic grating",
                                                   "blazed grating", "grism", "Littrow", "spectrometer", "monochromator"),
               build_grating),
        _class("deflector.acousto_optic", "MODULATOR", "Acousto-optic deflector (RF tones)", "An acousto-optic deflector or modulator driven by a list of RF tones: each "
               "tone deflects its own first order by lambda f_RF / v; the zero order stays. Its efficiency per tone is a table you state, and its directions need none.",
               "other_optical", "DIFFRACTION_GRATING",
               (_d("acoustic_velocity_m_per_s", "acoustic velocity in the crystal, as its source states it", unit="m/s", required=True, exclusive_minimum=POSITIVE),
                P("rf_tones_mhz", T.JSON, "the RF tones in MHz, a list of positive numbers (at most 64, each a whole multiple of 1 mHz)", required=True),
                P("diffraction_order", T.ENUM, "+1: toward the acoustic axis, optical frequency up by f_RF; -1: away from it, down by f_RF", choices=AOD_ORDERS, default="+1"),
                _d("acoustic_axis_angle_deg", "direction of the acoustic wave, from local +X toward local +Y", unit="deg", default="0"),
                _d("width_mm", "outer width along local X", required=True, exclusive_minimum=POSITIVE),
                _d("height_mm", "outer height along local Y", required=True, exclusive_minimum=POSITIVE),
                _d("clear_width_mm", "clear aperture width along local X", default_rule="0.9 width_mm", exclusive_minimum=POSITIVE),
                _d("clear_height_mm", "clear aperture height along local Y", default_rule="0.9 height_mm", exclusive_minimum=POSITIVE),
                _d("thickness_mm", "drawn crystal length behind its plane z = 0 (it acts at that plane)", default="6", exclusive_minimum=POSITIVE),
                _d("design_wavelength_nm", "wavelength of the direction sines in the design report", unit="nm", exclusive_minimum=POSITIVE),
                P("efficiency", T.JSON, '{"rows": [{"wavelength_nm", "rf_mhz", "efficiency"}, ...] (every tone at every wavelength; rf_mhz 0 states the zero order), "valid_aoi_deg": '
                  '[min_deg, max_deg], "data_kind": "MEASURED" | "INDEPENDENTLY_COMPUTED" | "ASSUMED", "basis", "evidence"?, "max_conical_angle_deg"?}: the share of the '
                  "incident power in each order, for this set of tones", default_rule="not stated: each order's direction is traced with its power not known")),
               "One thin grating of period v / df, df the greatest common divisor of the tones, each tone its integer order m f_RF / df: sin(theta) = sin(theta_in) + m lambda f_RF / v. "
               "The design report maps each tone to its order and optical frequency shift m f_RF.",
               ("static thin element: the crystal's length, Bragg selectivity and birefringence do not act on the light; the efficiency table states what they do",
                "tone phases, timing, RF crosstalk and intermodulation are not modeled (design.quantities)",
                "the table's efficiency holds for the polarization it was characterized at, whichever polarization arrives",
                "no optical path: wavefront and the linear optical network through it are refused"),
               (REF_SALEH_TEICH, REF_GOODMAN), ("AOD", "AOM", "acousto-optic deflector", "acousto-optic modulator", "Bragg cell", "RF tones", "beam steering", "tweezer array"),
               build_acousto_optic),
        _class("compound.keplerian_telescope", "COMPOUND", "Keplerian telescope / beam expander (spatial filter)",
               "Two positive native lenses spaced afocally; optional pinhole at the internal focus (spatial filter).", "other_optical", "COMPOUND",
               (_lens("lens1", "first lens"), _lens("lens2", "second lens"), COMPOUND_WAVELENGTH,
                _d("focal_stop_diameter_mm", "diameter of a pinhole at the internal focus", exclusive_minimum=POSITIVE),
                _d("focal_stop_thickness_mm", "drawn thickness of the pinhole plate", default="0.1", exclusive_minimum=POSITIVE)),
               "z2 = z1 + last1 + BFL1 + FFL2 (rear focus of lens 1 = front focus of lens 2): afocal; beam magnification -f2/f1.",
               ("paraxial spacing at the design wavelength", "member approximations apply"), (REF_HECHT, REF_SMITH),
               ("telescope", "beam expander", "spatial filter", "Keplerian"), partial(build_telescope, nested=nested, keplerian=True)),
        _class("compound.galilean_telescope", "COMPOUND", "Galilean telescope / beam expander",
               "A positive and a negative native lens spaced afocally (no internal focus).", "other_optical", "COMPOUND",
               (_lens("lens1", "first lens"), _lens("lens2", "second lens"), COMPOUND_WAVELENGTH),
               "z2 = z1 + last1 + BFL1 + FFL2 with one negative focal length: afocal, upright; beam magnification -f2/f1.",
               ("paraxial spacing at the design wavelength", "member approximations apply"), (REF_HECHT, REF_SMITH),
               ("telescope", "beam expander", "beam reducer", "Galilean"), partial(build_telescope, nested=nested, keplerian=False)),
        _class("compound.relay_4f", "COMPOUND", "4f relay (Fourier filter)",
               "Two positive native lenses imaging the unit's object plane (its origin) to its image plane (port OUT), with an optional "
               "Fourier-plane stop.", "other_optical", "COMPOUND",
               (_lens("lens1", "first lens"), _lens("lens2", "second lens"), COMPOUND_WAVELENGTH,
                _d("fourier_stop_diameter_mm", "opening of a Fourier-plane stop (low-pass)", exclusive_minimum=POSITIVE),
                _d("fourier_obstruction_diameter_mm", "opaque central disc in the Fourier plane (DC block / high-pass)", exclusive_minimum=POSITIVE),
                _d("fourier_stop_thickness_mm", "drawn thickness of the Fourier-plane stop", default="0.1", exclusive_minimum=POSITIVE)),
               "Object at lens 1's front focal plane, afocal spacing, image at lens 2's rear focal plane: m = -f2/f1, telecentric.",
               ("paraxial spacing at the design wavelength", "geometric Fourier-plane filtering: diffraction not modelled",
                "member approximations apply"), (REF_GOODMAN, REF_HECHT), ("4f", "relay", "Fourier filter", "spatial filtering", "imaging relay"),
               partial(build_relay, nested=nested)),
        _class("lens.axicon", "LENS", "Axicon (cone lens)", "A plane face and a cone with its apex on the axis: turns a collimated beam "
               "into a converging cone (a Bessel-like line focus) or a ring.", "lens", "SURFACE_SEQUENCE",
               (_d("cone_angle_deg", "base angle alpha between the cone and the plane face", unit="deg", required=True,
                   exclusive_minimum=POSITIVE, exclusive_maximum=RIGHT_ANGLE_DEG),
                P("material", T.MATERIAL, MATERIAL_HELP, required=True), DIAMETER, CLEAR, CENTER_THICKNESS,
                P("orientation", T.ENUM, "PLANE_FIRST: the plane face meets the light first (cone at the exit, the usual use); "
                  "CONE_FIRST: the apex points at the incoming light", choices=AXICON_ORIENTATIONS, default="PLANE_FIRST"),
                _d("design_wavelength_nm", "wavelength of the design report (index, deflection, Bessel zone length)", unit="nm",
                   exclusive_minimum=POSITIVE), AR_COATING, BAND_OPTIONAL),
               "Exact refraction at the plane and the cone (linear sag term -/+ tan(alpha) rho): a ray parallel to the axis leaves at "
               "beta = asin(n sin alpha) - alpha (plane first); the rays of a beam of radius h cross the axis within h / tan(beta).",
               (*REFRACTIVE, "geometric rays: the Bessel-beam diffraction pattern is not modelled; a Gaussian beam behind the cone is "
                "outside the Gaussian model (rays, power and spot metrics are traced)"),
               (REF_MCLEOD, REF_DURNIN, REF_BORN_WOLF), ("axicon", "cone lens", "Bessel beam", "line focus", "ring generator"), build_axicon),
        _class("prism.wedge", "PRISM", "Wedge prism (round)", "A round prism with a small angle between its faces: deviates a beam by a "
               "fixed angle; pairs of them steer a beam (Risley prisms).", "prism", "SURFACE_SET",
               (_d("wedge_angle_deg", "angle between the entrance and the exit face", unit="deg", required=True, exclusive_minimum=POSITIVE,
                   exclusive_maximum=HALF_RIGHT_ANGLE_DEG),
                _d("design_wavelength_nm", "wavelength of the OUT port and the deviation in the design report", unit="nm", required=True,
                   exclusive_minimum=POSITIVE),
                P("material", T.MATERIAL, MATERIAL_HELP, required=True), DIAMETER, CLEAR, CENTER_THICKNESS, AR_COATING, BAND_OPTIONAL),
               "Exact refraction at a plane entrance face and a plane exit face tilted by alpha about local Y (thick end toward local "
               "-X): deviation delta = asin(n sin alpha) - alpha toward the thick end for light along the axis ((n - 1) alpha for a "
               "thin prism); dispersion traced per wavelength.", REFRACTIVE, (REF_HECHT, REF_SMITH),
               ("wedge prism", "wedge", "beam deviator", "Risley prism", "beam steering"), build_wedge),
        _class("mirror.hollow_roof", "MIRROR", "Hollow roof mirror", "Two flat mirrors at 90 deg (a hollow roof prism): returns a beam "
               "antiparallel in the plane across its ridge and reverses the image across the ridge.", "mirror", "COMPOUND",
               (_d("aperture_width_mm", "aperture along the ridge (local X)", required=True, exclusive_minimum=POSITIVE),
                _d("aperture_height_mm", "aperture across the ridge (local Y); each face is height / sqrt 2 wide", required=True,
                   exclusive_minimum=POSITIVE),
                *REFLECTANCE, P("coating_kind", T.ENUM, "reflector type", choices=MIRROR_KINDS, default="METALLIC_REFLECTOR"),
                BAND_REQUIRED, AOI, SEAM, SUBSTRATE),
               "Two plane reflections at faces inclined at +-45 deg to local Z, meeting along local X: the reflection matrix is "
               "diag(1, -1, -1); a ray at height y returns at -y.",
               ("declared reflectance, constant over its band", "coating phase not modelled", "transmitted light not followed",
                "ideal faces at exactly 90 deg, a seam of seam_mm along the ridge"), (REF_HECHT, REF_BORN_WOLF),
               ("roof mirror", "hollow roof prism", "lateral transfer retroreflector", "image reversal"), build_hollow_roof),
        _class("retroreflector.hollow_corner", "MIRROR", "Hollow corner-cube retroreflector", "Three mutually perpendicular flat mirrors "
               "(a cube corner): returns every beam that meets all three antiparallel to itself, whatever its direction.", "mirror",
               "COMPOUND",
               (_d("mirror_edge_mm", "edge of each square mirror", required=True, exclusive_minimum=POSITIVE),
                *REFLECTANCE, P("coating_kind", T.ENUM, "reflector type", choices=MIRROR_KINDS, default="METALLIC_REFLECTOR"),
                BAND_REQUIRED, AOI, SEAM, SUBSTRATE),
               "Three plane reflections at mutually orthogonal faces: reflection matrix -I (antiparallel return), the exit point "
               "reflected through the axis (local Z, the cube diagonal).",
               ("declared reflectance, constant over its band", "coating phase not modelled", "transmitted light not followed",
                "ideal faces at exactly 90 deg, a seam of seam_mm along each edge"), (REF_HECHT, REF_ECKHARDT),
               ("corner cube", "retroreflector", "hollow retroreflector", "laser tracker target"), build_hollow_corner),
        _class("splitter.beam_displacer", "SPLITTER", "Beam displacer (uniaxial crystal)", "A rectangular plate of a uniaxial crystal "
               "(calcite, alpha-BBO, YVO4) cut with its optic axis at an angle to the faces: splits a beam into two parallel beams of "
               "orthogonal linear polarization, separated by the walk-off of the extraordinary ray.", "splitter", "SURFACE_SEQUENCE",
               (P("material", T.MATERIAL, CRYSTAL_HELP, required=True),
                _d("design_wavelength_nm", "wavelength of the walk-off and displacement in the design report (and of length from "
                   "displacement_mm)", unit="nm", required=True, exclusive_minimum=POSITIVE),
                _d("length_mm", "crystal length along local Z (or give displacement_mm)", exclusive_minimum=POSITIVE),
                _d("displacement_mm", "separation of the two beams at the design wavelength (the length follows, rounded up to 1 um)",
                   exclusive_minimum=POSITIVE),
                _d("axis_angle_deg", "angle of the optic axis from local +Z toward local -X", unit="deg", default="45",
                   exclusive_minimum=POSITIVE, exclusive_maximum=RIGHT_ANGLE_DEG),
                _d("width_mm", "outer width (local X)", default="10", exclusive_minimum=POSITIVE),
                _d("height_mm", "outer height (local Y)", default="10", exclusive_minimum=POSITIVE),
                FRACTION, AR_COATING, BAND_OPTIONAL),
               "Uniaxial crystal (Engine model): the ordinary beam (polarized along local Y) goes straight; the extraordinary beam "
               "(polarized along local X) walks off by rho, tan(theta + rho) = (n_o / n_e)^2 tan(theta), and leaves parallel, displaced by "
               "L tan(rho) along local +X for a negative crystal (-X for a positive one).",
               (*REFRACTIVE, "the crystal faces' power split per wave is the isotropic Fresnel result at each wave's index (exact at normal "
                "incidence; a run states the approximation beyond a few degrees)"),
               (REF_BORN_WOLF, REF_SALEH_TEICH), ("beam displacer", "walk-off", "calcite", "birefringent", "polarizing beam displacer",
                                                  "double refraction", "crystal"), build_beam_displacer),
        _class("polarizer.glan", "POLARIZATION", "Glan polarizer (air-spaced crystal prisms)", "Two right-angle prisms of a negative "
               "uniaxial crystal (calcite, alpha-BBO) with an air gap: the ordinary wave is totally reflected at the gap and leaves "
               "through the side face, the extraordinary wave passes. GLAN_TAYLOR transmits light polarized along local X, "
               "GLAN_FOUCAULT along local Y.", "polarizer", "COMPOUND",
               (P("material", T.MATERIAL, CRYSTAL_HELP, required=True),
                _d("design_wavelength_nm", "wavelength the gap angle window is checked (and its default chosen) at", unit="nm", required=True,
                   exclusive_minimum=POSITIVE),
                _d("aperture_mm", "side of the square aperture (local X and Y)", required=True, exclusive_minimum=POSITIVE),
                P("variant", T.ENUM, "GLAN_TAYLOR: optic axes along local X (in the plane of incidence at the gap); GLAN_FOUCAULT: "
                  "along local Y", choices=GLAN_VARIANTS, default="GLAN_TAYLOR"),
                _d("gap_angle_deg", "angle of incidence of the axial beam on the gap", unit="deg",
                   default_rule="the middle of [asin(1 / n_o), asin(1 / n_e)] at the design wavelength, to 0.001 deg", exclusive_minimum=POSITIVE,
                   exclusive_maximum=RIGHT_ANGLE_DEG),
                _d("air_gap_mm", "width of the air gap across its faces", default="0.02", exclusive_minimum=POSITIVE),
                FRACTION, AR_COATING, BAND_OPTIONAL),
               "Uniaxial crystal (Engine model): at the gap the ordinary wave meets its critical angle asin(1 / n_o) and is totally "
               "reflected, the extraordinary wave (critical angle asin(1 / n_e)) refracts into the gap and into the second prism and "
               "leaves along local +Z. Prism length a tan(A_g) + gap / cos(A_g).",
               (*REFRACTIVE, "faces with no traced surface (the prisms' top, bottom and back) absorb the light that reaches them",
                "GLAN_TAYLOR: the gap's power split of the extraordinary wave is the isotropic Fresnel result at its index (a run states "
                "the approximation); GLAN_FOUCAULT: exact (the axes are perpendicular to the plane of incidence)"),
               (REF_BENNETT, REF_BORN_WOLF), ("Glan-Taylor", "Glan-laser", "Glan-Foucault", "Glan polarizer", "calcite polarizer",
                                              "crystal polarizer", "high extinction polarizer"), build_glan),
        _class("lens.grin_rod", "LENS", "GRIN rod lens (radial gradient index)", "A cylindrical rod of a radial gradient-index material "
               "with plane end faces: it focuses by its index profile n(r)^2 = n0^2 (1 - (g r)^2); a quarter pitch focuses a collimated "
               "beam on its exit face.", "lens", "SURFACE_SEQUENCE",
               (P("material", T.MATERIAL, "a gradient-index material: a ProjectModelMaterialV1 document with gradient (its dispersion "
                  "is the index on the axis n0)", required=True),
                _d("design_wavelength_nm", "wavelength of the pitch, the focal length and the matrix in the design report", unit="nm",
                   required=True, exclusive_minimum=POSITIVE),
                DIAMETER, CLEAR,
                _d("length_mm", "rod length (or give pitch)", exclusive_minimum=POSITIVE),
                _d("pitch", "pitch P = g L / (2 pi) at the design wavelength (the length follows, rounded up to 1 um)", unit=None,
                   exclusive_minimum=POSITIVE),
                AR_COATING, BAND_OPTIONAL),
               "Radial gradient-index medium (Engine model): every ray, meridional or skew, follows x(z) = x0 cos(W z) + (x0' / W) "
               "sin(W z) with W = n0 g / beta between the end faces (closed form); a paraxial ray's period is 2 pi / g; paraxial "
               "matrix ((cos gL, sin gL / (n0 g)), (-n0 g sin gL, cos gL)).",
               (*REFRACTIVE, "the index profile is n(r)^2 = n0^2 (1 - (g r)^2) out to the rod's side (light reaching the side is not "
                "followed)", "end faces perpendicular to the axis"),
               (REF_MARCHAND, REF_BORN_WOLF), ("GRIN", "GRIN lens", "gradient index", "rod lens", "fiber collimator"),
               build_grin_rod),
        _class("polarizer.glan_thompson", "POLARIZATION", "Glan-Thompson polarizer (cemented crystal prisms)", "Two right-angle prisms "
               "of a negative uniaxial crystal (calcite, alpha-BBO) cemented along their long faces: the ordinary wave is totally "
               "reflected at the cement layer and leaves through the side face, the extraordinary wave (polarized along local Y) "
               "passes.", "polarizer", "COMPOUND",
               (P("material", T.MATERIAL, CRYSTAL_HELP, required=True),
                P("cement", T.MATERIAL, "the cement layer's isotropic material (its index below the crystal's ordinary index), in "
                  "the forms of an isotropic material: " + MATERIAL_HELP, required=True),
                _d("design_wavelength_nm", "wavelength the gap angle window is checked (and its default chosen) at", unit="nm", required=True,
                   exclusive_minimum=POSITIVE),
                _d("aperture_mm", "side of the square aperture (local X and Y)", required=True, exclusive_minimum=POSITIVE),
                _d("gap_angle_deg", "angle of incidence of the axial beam on the cement layer", unit="deg",
                   default_rule="the middle of [asin(n_c / n_o), asin(n_c / n_e)] at the design wavelength, to 0.001 deg (required when "
                   "n_c >= n_e)", exclusive_minimum=POSITIVE, exclusive_maximum=RIGHT_ANGLE_DEG),
                _d("cement_thickness_mm", "thickness of the cement layer", default="0.01", exclusive_minimum=POSITIVE, maximum=UNIT),
                FRACTION, AR_COATING, BAND_OPTIONAL),
               "Uniaxial crystal (Engine model) and a cemented joint: at the crystal-cement face the ordinary wave meets its critical "
               "angle asin(n_c / n_o) and is totally reflected, the extraordinary wave crosses the layer into the second prism and leaves "
               "along local +Z. Both optic axes along local Y (perpendicular to the plane of reflection). Prism length "
               "a tan(A_g) + t / cos(A_g).",
               (*REFRACTIVE, "faces with no traced surface (the prisms' top, bottom and back) absorb the light that reaches them",
                "the cement layer is a uniform isotropic layer between parallel faces (its thin-film interference is not modelled)",
                "the faces' power split is exact (the optic axes are perpendicular to the plane of incidence)"),
               (REF_BENNETT, REF_BORN_WOLF), ("Glan-Thompson", "cemented polarizer", "calcite polarizer", "crystal polarizer",
                                              "high extinction polarizer"), build_glan_thompson),
        _class("splitter.wollaston", "SPLITTER", "Wollaston prism (crystal polarizing beam splitter)", "Two right-angle prisms of a "
               "uniaxial crystal (calcite, quartz, alpha-BBO, YVO4) joined along the hypotenuse with their optic axes crossed (prism "
               "A along local X, prism B along local Y): splits a beam into two diverging beams of orthogonal linear polarization, "
               "deviated to either side in the local X-Z plane.", "splitter", "COMPOUND",
               (P("material", T.MATERIAL, CRYSTAL_HELP, required=True),
                _d("design_wavelength_nm", "wavelength of the exit angles in the design report", unit="nm", required=True,
                   exclusive_minimum=POSITIVE),
                _d("aperture_mm", "side of the square aperture (local X and Y)", required=True, exclusive_minimum=POSITIVE),
                _d("wedge_angle_deg", "angle between the joint and the entrance face", unit="deg", required=True,
                   exclusive_minimum=POSITIVE, exclusive_maximum=RIGHT_ANGLE_DEG),
                _d("edge_thickness_mm", "thickness of each prism at its thin end", default_rule="a tenth of aperture_mm",
                   exclusive_minimum=POSITIVE),
                P("cement", T.MATERIAL, "optional cement layer's isotropic material (absent: an optical contact): " + MATERIAL_HELP),
                _d("cement_thickness_mm", "thickness of the cement layer (with cement)", default="0.01", exclusive_minimum=POSITIVE,
                   maximum=UNIT),
                FRACTION, AR_COATING, BAND_OPTIONAL),
               "Uniaxial crystal (Engine model) and a joint between two crystals cut differently: the light polarized along local X "
               "travels as the extraordinary wave in prism A and the ordinary wave in prism B, the light polarized along local Y the "
               "other way round, so the two refract at the joint with indices exchanged and leave on either side of local +Z; "
               "separation close to 2 asin((n_o - n_e) tan(theta)).",
               (*REFRACTIVE, "faces with no traced surface (the prisms' sides) absorb the light that reaches them",
                "the joint's power split of the wave polarized along local X is the isotropic Fresnel result at each wave's index (prism "
                "A's optic axis lies in the plane of incidence: a run states the approximation beyond a few degrees)",
                "a cement layer is a uniform isotropic layer between parallel faces (its thin-film interference is not modelled)"),
               (REF_BENNETT, REF_BORN_WOLF), ("Wollaston", "Wollaston prism", "polarizing beam splitter", "crystal splitter", "calcite",
                                              "birefringent prism"), build_wollaston),
        _class("splitter.rochon", "SPLITTER", "Rochon prism (crystal polarizing beam splitter)", "Two right-angle prisms of a uniaxial "
               "crystal (quartz, calcite, MgF2, alpha-BBO) joined along the hypotenuse, prism A's optic axis along the beam (local Z), "
               "prism B's along local Y: light polarized along local X leaves undeviated, light polarized along local Y is deviated "
               "in the local X-Z plane.", "splitter", "COMPOUND",
               (P("material", T.MATERIAL, CRYSTAL_HELP, required=True),
                _d("design_wavelength_nm", "wavelength of the exit angles in the design report", unit="nm", required=True,
                   exclusive_minimum=POSITIVE),
                _d("aperture_mm", "side of the square aperture (local X and Y)", required=True, exclusive_minimum=POSITIVE),
                _d("wedge_angle_deg", "angle between the joint and the entrance face", unit="deg", required=True,
                   exclusive_minimum=POSITIVE, exclusive_maximum=RIGHT_ANGLE_DEG),
                _d("edge_thickness_mm", "thickness of each prism at its thin end", default_rule="a tenth of aperture_mm",
                   exclusive_minimum=POSITIVE),
                P("cement", T.MATERIAL, "optional cement layer's isotropic material (absent: an optical contact): " + MATERIAL_HELP),
                _d("cement_thickness_mm", "thickness of the cement layer (with cement)", default="0.01", exclusive_minimum=POSITIVE,
                   maximum=UNIT),
                FRACTION, AR_COATING, BAND_OPTIONAL),
               "Uniaxial crystal (Engine model) and a joint between two crystals cut differently: along the optic axis of prism A both "
               "polarizations travel as the ordinary wave; in prism B light polarized along local X stays ordinary (undeviated) and "
               "light polarized along local Y is extraordinary, deviated by close to asin((n_o - n_e) tan(theta)).",
               (*REFRACTIVE, "faces with no traced surface (the prisms' sides) absorb the light that reaches them",
                "the joint's power split is the isotropic Fresnel result at each wave's index (prism A's optic axis lies in the plane "
                "of incidence: a run states the approximation beyond a few degrees)",
                "a cement layer is a uniform isotropic layer between parallel faces (its thin-film interference is not modelled)"),
               (REF_BENNETT, REF_BORN_WOLF), ("Rochon", "Rochon prism", "polarizing beam splitter", "crystal splitter", "quartz",
                                              "birefringent prism"), build_rochon),
        _class("filter.variable_nd", "FILTER", "Variable neutral-density filter", "A filter wheel or a linear gradient filter whose "
               "optical density depends on where the beam crosses it; the wheel's rotation or the filter's shift is its position "
               "parameter.", "filter", "NEUTRAL_DENSITY",
               (P("law", T.ENUM, "LINEAR_ROTARY: OD linear in the azimuth about the axis (a wheel); LINEAR_GRADIENT: OD linear along "
                  "local X", choices=ND_LAWS, required=True),
                _d("od_start", "optical density at the start of the coated arc / interval", unit=None, required=True, minimum=POSITIVE),
                _d("od_end", "optical density at its end", unit=None, required=True, minimum=POSITIVE),
                _d("arc_start_deg", "LINEAR_ROTARY: start of the coated arc (azimuth from local +X toward +Y)", unit="deg",
                   default_rule="0 for a wheel", minimum=POSITIVE, maximum=FULL_TURN_DEG),
                _d("arc_end_deg", "LINEAR_ROTARY: end of the coated arc", unit="deg", exclusive_minimum=POSITIVE, maximum=FULL_TURN_DEG),
                _d("linearity_percent", "LINEAR_ROTARY: stated deviation from the linear law (recorded with the model)", unit="percent",
                   minimum=POSITIVE),
                _d("x_start_mm", "LINEAR_GRADIENT: start of the coated interval along local X"),
                _d("x_end_mm", "LINEAR_GRADIENT: end of the coated interval"),
                _d("coated_inner_radius_mm", "a coating on an annulus: the uncoated hub's radius (a round filter)", exclusive_minimum=POSITIVE),
                _d("uncoated_transmittance", "power transmittance where the plate is uncoated (outside the arc / interval, in the hub)",
                   unit=None, minimum=POSITIVE, maximum=UNIT, default_rule="not stated: light there is outside the model"),
                _d("reflectance", "power reflectance of the filter (traced as a reflected beam)", unit=None, minimum=POSITIVE, maximum=UNIT,
                   default_rule="0: the light not transmitted is absorbed"),
                *ROUND_OR_RECT, BAND_REQUIRED,
                _d("position", "the wheel's rotation (deg, nd_rotation_deg) or the gradient's shift along local X (mm, nd_offset_mm), "
                   "written into the component template", unit="deg | mm", default_rule="0"),
                _d("thickness_mm", "drawn thickness of the filter (it acts at its plane z = 0)", default="1", exclusive_minimum=POSITIVE)),
               "Thin attenuator: each ray is scaled by sqrt(10^-OD) with the density of the law at its own crossing point; a beam "
               "whose density varies by more than 0.2 OD across its 1/e^2 footprint is outside the model.",
               ("ideal thin element: the substrate is not modelled", "density independent of wavelength over the band",
                "no phase: the linear optical network through it is refused"), (REF_ISO_ND, REF_HECHT),
               ("variable ND", "ND wheel", "continuously variable attenuator", "gradient ND", "neutral density filter"), build_variable_nd),
        _class("lens.tabulated_thin", "LENS", "Tabulated thin lens (external response)", "A thin focusing element whose focal length "
               "and delivered power fraction are a table over wavelength from an external simulation or measurement (a metalens, a "
               "diffractive lens), with its validated domain, metrics and evidence.", "lens", "TABULATED_THIN_LENS",
               (P("rows", T.JSON, '[{"wavelength_nm", "focal_length_mm", "transmittance"}, ...] in increasing wavelength, or columns '
                  '{"wavelength_nm": [...], "focal_length_mm": [...], "transmittance": [...]}', required=True),
                P("focal_interpolation", T.ENUM, "between rows: 1/f linear in wavelength, or only at the rows (within 0.5 percent)",
                  choices=FOCAL_INTERPOLATIONS, default="LINEAR_IN_OPTICAL_POWER"),
                P("transmittance_interpolation", T.ENUM, "between rows: linear, or only at the rows", choices=TRANSMITTANCE_INTERPOLATIONS,
                  default="LINEAR"),
                _d("clear_aperture_diameter_mm", "clear aperture diameter", required=True, exclusive_minimum=POSITIVE),
                _d("diameter_mm", "outer diameter (light between it and the clear aperture is blocked)", default_rule="the clear aperture",
                   exclusive_minimum=POSITIVE),
                _d("max_field_angle_deg", "largest chief-ray angle to the axis the table was validated for", unit="deg",
                   exclusive_minimum=POSITIVE, exclusive_maximum=RIGHT_ANGLE_DEG, default_rule="not stated: no field angle is checked"),
                _d("max_beam_radius_mm", "largest Gaussian beam radius at the element the table was validated for", exclusive_minimum=POSITIVE),
                P("validated_polarizations", T.JSON, 'input polarizations validated: ["LINEAR_X"], ["LINEAR_Y"], both, or ["ANY"]',
                  default_rule="not stated: every polarization"),
                P("supported_metrics", T.JSON, '[{"metric": <target metric>, "abs_error"?: e, "rel_error"?: r, "basis": "..."}, ...]: '
                  "the result metrics the response was validated for", default_rule="not stated: no metric is singled out"),
                P("evidence", T.JSON, '{"method", "tools": [{"name", "version"}], "source_data_sha256", "configuration_sha256"?, '
                  '"fit_model", "validation_summary", "date": "YYYY-MM-DD"}', default_rule="not stated: an assumed response"),
                THIN_THICKNESS),
               "Each traced line is deflected as by a perfect thin lens of its own focal length f(lambda) (u' = u - y / f) and carries "
               "T(lambda) of its power; the rest is the element's unmodelled response. Outside the table's wavelengths, or a stated field "
               "angle, beam size, polarization or metric limit, a run is outside the model's scope.",
               ("ideal thin element: no aberrations and no phase (interference and the linear optical network through it are refused)",
                "the response holds only inside its stated domain"), (REF_GOODMAN, REF_HECHT),
               ("metalens", "metasurface lens", "diffractive lens", "tabulated lens", "external response"), build_tabulated_thin),
    ]
    for entry in entries:
        classes[entry.class_id] = entry
    return classes


#: Every native component class, by id.
NATIVE_CLASSES = _registry()


def native_class(class_id: str) -> NativeClass:
    entry = NATIVE_CLASSES.get(class_id) if isinstance(class_id, str) else None
    if entry is None:
        raise NativeComponentError([NativeProblem(NativeErrorCode.CLASS_UNKNOWN,
                                                  f"no native component class {class_id!r}; known: {sorted(NATIVE_CLASSES)}", "class_id")])
    return entry


__all__ = ["AVAILABILITY", "CATALOG_EQUIVALENTS", "LENS_CLASSES", "NATIVE_CLASSES", "NativeClass", "native_class"]
