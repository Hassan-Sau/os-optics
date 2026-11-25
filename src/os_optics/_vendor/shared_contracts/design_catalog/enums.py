"""Closed vocabularies of the design-stage public optical catalog.

Every enum here is closed. `OpticalCategory` names the elements that directly take part in the optical system;
mounting hardware (mounts, posts, stages, bases, breadboards, fasteners) is configured by OpenSci for each submitted
design (``MechanicalAccessoryPolicy``).
"""

from __future__ import annotations

from enum import StrEnum


class OpticalCategory(StrEnum):
    """Design-stage optical element category.

    ``SOURCE`` and ``DETECTOR`` are never published products: a light source is a Project ``custom_part`` with an explicit emission model and an
    observation plane is virtual design intent.
    """

    LENS = "LENS"
    MIRROR = "MIRROR"
    BEAMSPLITTER = "BEAMSPLITTER"
    FILTER = "FILTER"
    WINDOW = "WINDOW"
    WAVEPLATE = "WAVEPLATE"
    POLARIZER = "POLARIZER"
    PRISM = "PRISM"
    SOURCE = "SOURCE"
    DETECTOR = "DETECTOR"


class OpticalForm(StrEnum):
    """The optical form within a category (``FORMS_BY_CATEGORY`` says which category may carry which form)."""

    # LENS
    PLANO_CONVEX_LENS = "PLANO_CONVEX_LENS"
    BICONVEX_LENS = "BICONVEX_LENS"
    PLANO_CONCAVE_LENS = "PLANO_CONCAVE_LENS"
    BICONCAVE_LENS = "BICONCAVE_LENS"
    POSITIVE_MENISCUS_LENS = "POSITIVE_MENISCUS_LENS"
    NEGATIVE_MENISCUS_LENS = "NEGATIVE_MENISCUS_LENS"
    BEST_FORM_LENS = "BEST_FORM_LENS"
    ACHROMATIC_DOUBLET = "ACHROMATIC_DOUBLET"
    ASPHERIC_LENS = "ASPHERIC_LENS"
    CYLINDRICAL_LENS = "CYLINDRICAL_LENS"
    AXICON = "AXICON"
    # MIRROR
    FLAT_MIRROR = "FLAT_MIRROR"
    CONCAVE_MIRROR = "CONCAVE_MIRROR"
    OFF_AXIS_PARABOLIC_MIRROR = "OFF_AXIS_PARABOLIC_MIRROR"
    # BEAMSPLITTER
    PLATE_BEAMSPLITTER = "PLATE_BEAMSPLITTER"
    BEAM_SAMPLER = "BEAM_SAMPLER"
    CUBE_BEAMSPLITTER = "CUBE_BEAMSPLITTER"
    POLARIZING_CUBE_BEAMSPLITTER = "POLARIZING_CUBE_BEAMSPLITTER"
    POLARIZING_PLATE_BEAMSPLITTER = "POLARIZING_PLATE_BEAMSPLITTER"
    # FILTER
    DICHROIC_MIRROR = "DICHROIC_MIRROR"
    BANDPASS_FILTER = "BANDPASS_FILTER"
    LONGPASS_FILTER = "LONGPASS_FILTER"
    SHORTPASS_FILTER = "SHORTPASS_FILTER"
    NOTCH_FILTER = "NOTCH_FILTER"
    NEUTRAL_DENSITY_FILTER = "NEUTRAL_DENSITY_FILTER"
    # WINDOW
    PLANE_WINDOW = "PLANE_WINDOW"
    # WAVEPLATE
    HALF_WAVE_PLATE = "HALF_WAVE_PLATE"
    QUARTER_WAVE_PLATE = "QUARTER_WAVE_PLATE"
    # POLARIZER
    LINEAR_POLARIZER = "LINEAR_POLARIZER"
    # PRISM
    RIGHT_ANGLE_PRISM = "RIGHT_ANGLE_PRISM"
    PENTA_PRISM = "PENTA_PRISM"
    DOVE_PRISM = "DOVE_PRISM"
    EQUILATERAL_PRISM = "EQUILATERAL_PRISM"
    WEDGE_PRISM = "WEDGE_PRISM"
    ROOF_PRISM = "ROOF_PRISM"
    CORNER_CUBE_RETROREFLECTOR = "CORNER_CUBE_RETROREFLECTOR"
    # POLARIZING PRISMS (uniaxial crystal prisms joined face to face or across an air gap; catalog schema 3.5)
    GLAN_TAYLOR_PRISM = "GLAN_TAYLOR_PRISM"
    GLAN_LASER_PRISM = "GLAN_LASER_PRISM"
    GLAN_THOMPSON_PRISM = "GLAN_THOMPSON_PRISM"
    WOLLASTON_PRISM = "WOLLASTON_PRISM"
    ROCHON_PRISM = "ROCHON_PRISM"


class FactStatus(StrEnum):
    """How a published value is established.

    ``SPECIFIED``: a nominal specification value of the product.
    ``DERIVED``: computed by OpenSci from the product's engineering data (for example a focal length at a
    wavelength other than the nominal one, or a paraxial property of a specified prescription).
    ``DESIGN_BASIS``: the value designed and simulated with for a specification stated as a bound or a percentage (for
    example a specified upper bound used as the value, or a clear aperture given as a percentage of the diameter).
    ``NOMINAL``: a nominal body drawn from the element's outer size (for example a cylinder of its outer diameter and
    thickness as design geometry).
    ``TYPICAL``: a manufacturer's typical measured value or curve (for example a typical transmission spectrum); the
    part-to-part variation around it is not modelled.
    ``DECLARED``: a value the author of a Project states for an element the Project declares (for example the power
    rating of the customer's own optic); the public catalog never publishes it.
    """

    SPECIFIED = "SPECIFIED"
    DERIVED = "DERIVED"
    DESIGN_BASIS = "DESIGN_BASIS"
    NOMINAL = "NOMINAL"
    TYPICAL = "TYPICAL"
    DECLARED = "DECLARED"


class CommercialReadiness(StrEnum):
    """Whether OpenSci supplies the product.

    ``ORDERABLE``: OpenSci supplies it: a design that uses it can be submitted for commercial resolution and ordered, and
    a part list can name it. Whether OpenSci also mounts it is the product's ``opensci_mounting``.
    ``SIMULATION_ONLY``: available for design and simulation; OpenSci does not supply it. A design that uses it can still
    be resolved and ordered for the products OpenSci supplies: the customer supplies this one.
    """

    SIMULATION_ONLY = "SIMULATION_ONLY"
    ORDERABLE = "ORDERABLE"


class OpenSciMounting(StrEnum):
    """Whether OpenSci supplies mounting hardware for the product (catalog schema 3.1).

    ``AVAILABLE``: bought with OpenSci mechanics (a complete system, mounts per optic, or a part list with
    ``mechanical_supply`` ``DERIVED``), the product comes with an OpenSci mount, post and post holder.
    ``NOT_AVAILABLE``: OpenSci has no mounting hardware for it. An ORDERABLE product is still supplied: bought as optics
    only or on a part list without mechanics as usual; in a design bought with OpenSci mechanics it is supplied
    unmounted -- the customer mounts it at its design position, and the OpenSci assembly service is not offered for that
    design; a part list with derived mechanics cannot derive its mount (``MOUNTING_NOT_AVAILABLE``). A SIMULATION_ONLY
    product is always ``NOT_AVAILABLE``.
    """

    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class DeliveredForm(StrEnum):
    """The form in which OpenSci delivers the product (catalog schema 3.1).

    ``OPTIC``: the optical element itself; its outline is the product's design geometry.
    ``HOUSED``: the optical element in its manufacturer's housing. Its optics and simulation model are those of the
    element, but the delivered part's outline is the housing, which differs from the design geometry: allow for the
    housing's size and mount the product by its housing. OpenSci does not mount a housed product
    (``opensci_mounting`` ``NOT_AVAILABLE``).
    """

    OPTIC = "OPTIC"
    HOUSED = "HOUSED"


class CoatingKind(StrEnum):
    """What kind of coating a published coating model (or a product's summary) is."""

    ANTI_REFLECTION = "ANTI_REFLECTION"
    UNCOATED = "UNCOATED"
    DIELECTRIC_HIGH_REFLECTOR = "DIELECTRIC_HIGH_REFLECTOR"
    METALLIC_REFLECTOR = "METALLIC_REFLECTOR"
    BEAMSPLITTER = "BEAMSPLITTER"
    DICHROIC = "DICHROIC"
    BANDPASS = "BANDPASS"
    EDGEPASS = "EDGEPASS"
    NOTCH = "NOTCH"
    NEUTRAL_DENSITY = "NEUTRAL_DENSITY"


class CoatingAoiReference(StrEnum):
    """Which angle a coating's ``valid_aoi_deg`` states.

    ``AIR_EQUIVALENT``: the chief ray's angle of incidence expressed as its equivalent angle in air (``n sin(theta)``
    is invariant through a thin film), so a coating on the back face of a plate is judged by the same number as on its
    front.
    ``INCIDENT_MEDIUM``: the angle of incidence in the medium the light arrives from (for example 45 deg inside the
    glass for the coated diagonal of a cube splitter).
    """

    AIR_EQUIVALENT = "AIR_EQUIVALENT"
    INCIDENT_MEDIUM = "INCIDENT_MEDIUM"


class OpticalSymmetry(StrEnum):
    """Symmetry of the element about its local +Z optical axis.

    ``ROTATIONAL_ABOUT_LOCAL_Z``: rotating the element about local Z (Rz) has no optical effect.
    ``NONE``: turning the element about local Z changes what it does: it has an in-plane axis (local +X, e.g. a waveplate
    fast axis or a polarizer transmission axis), a cylindrical surface, a rectangular outline or aperture, a decentred
    aperture, or posed surfaces (a cube, a prism, a wedge, an off-axis mirror section).
    """

    ROTATIONAL_ABOUT_LOCAL_Z = "ROTATIONAL_ABOUT_LOCAL_Z"
    NONE = "NONE"


class FrameOrigin(StrEnum):
    """Where the element's local frame origin sits. Local +Z is always the optical/reference axis.

    ``FIRST_SURFACE_VERTEX``: the vertex of a transmissive element's first surface (light arrives from local -Z).
    ``REFLECTING_FACE_CENTER``: the centre of a reflector's (or plate splitter's) coated face, on its face normal.
    ``BODY_CENTER``: the centroid (centre of volume) of a solid element's body (a cube, a prism); for a round body (a
    round wedge prism), the point of its axis (local Z) at the centroid's height.
    ``APERTURE_CENTER``: the centre of a decentred surface's clear aperture, on the surface (an off-axis mirror section).
    """

    FIRST_SURFACE_VERTEX = "FIRST_SURFACE_VERTEX"
    REFLECTING_FACE_CENTER = "REFLECTING_FACE_CENTER"
    BODY_CENTER = "BODY_CENTER"
    APERTURE_CENTER = "APERTURE_CENTER"


class FrameReferenceAxis(StrEnum):
    """What local +Z means when it is not a transmissive element's optical axis.

    ``FACE_NORMAL``: local +Z is the outward normal of the coated face and points toward the side light must arrive
    from. Light arrives travelling toward local -Z (at some angle of incidence) and leaves reflected about local +Z
    (and, for a splitter, also transmitted through the plate); the substrate lies behind the face, at local z < 0.
    ``PORTS``: the element is placed by its ports (``OpticalFrameV1.ports``): the nominal input beam travels along the
    ``IN`` port's direction through its point, and each output port says where and in which direction light leaves
    (a cube splitter, a prism, an off-axis mirror section).
    """

    FACE_NORMAL = "FACE_NORMAL"
    PORTS = "PORTS"


class ModelFidelity(StrEnum):
    """How closely a simulation model represents the physical product.

    ``SURFACE_PRESCRIPTION``: the element's own surfaces (radii, vertex positions, clear apertures) and materials (a
    published dispersion model) are traced exactly: refraction, dispersion, Fresnel loss of uncoated surfaces,
    aberrations and the Gaussian beam through each surface come from the Engine's general physics.
    ``SPECIFICATION_ENVELOPE``: the element's response is its specification envelope (for example a coating's
    specified reflectance bound, constant within a specified band); geometry may still be physical.
    ``IDEAL_THIN_ELEMENT``: an idealized element of zero thickness at one plane (for example a Jones-matrix waveplate or
    polarizer, or a coated plane with no substrate modelled).
    """

    SURFACE_PRESCRIPTION = "SURFACE_PRESCRIPTION"
    SPECIFICATION_ENVELOPE = "SPECIFICATION_ENVELOPE"
    IDEAL_THIN_ELEMENT = "IDEAL_THIN_ELEMENT"


class SimulationApproximation(StrEnum):
    """Named simplifications a simulation model makes relative to the physical element.

    ``NOMINAL_PRESCRIPTION``: nominal radii, thicknesses and materials; manufacturing tolerances are not modelled.
    ``IDEAL_SURFACE_FIGURE``: surfaces have exactly their published profile (plane, sphere, conic, even asphere,
    cylinder); figure error, scatter and roughness are not modelled.
    ``CATALOG_MATERIAL_DISPERSION``: the material's refractive index is the dispersion formula of its glass catalogue
    (for a product specified as "or equivalent", that of the named reference glass).
    ``COATING_AT_SPECIFIED_BOUND``: a coating's power reflectance / transmittance is its specified bound or nominal value
    (for example "Ravg < 0.5 %" modelled as R = 0.5 %, T = 99.5 %; "R:T = 50:50 +- 8 %" as R = T = 50 %), constant
    within each specified band.
    ``LOSSLESS_COATING``: where a specification bounds only one of R and T in a band, the other is taken as 1 minus it
    (no absorption or scatter in the coating).
    ``CONSTANT_OPTICAL_DENSITY``: a neutral-density filter's optical density is its specified value at every wavelength of
    its working band (its spectral variation is not modelled).
    ``POLARIZATION_INDEPENDENT_COATING``: the s and p power coefficients of a coating are equal (its specification is an
    unpolarized or average value).
    ``COATING_PHASE_NOT_MODELLED``: a coating's own reflection / transmission phase and s/p retardance are not modelled
    (the bare-interface or ideal-reflector phase convention is used).
    ``GHOST_REFLECTIONS_NOT_TRACED``: the residual reflection of a transmitting surface is not followed.
    ``TRANSMITTED_LIGHT_NOT_TRACED``: power a reflector does not reflect is not followed.
    ``REFLECTED_LIGHT_NOT_TRACED``: power a filter does not transmit is not followed.
    ``SUBSTRATE_NOT_MODELLED``: the element is one coated plane; its substrate's thickness and refraction are ignored.
    ``THIN_ELEMENT``: a Jones-matrix element of zero thickness at one plane (plate thickness and its refraction ignored).
    ``RETARDANCE_INVERSE_WAVELENGTH``: a zero-order retarder's retardance scales as ``delta0 * lambda0 / lambda``
    (the birefringence dispersion of the plate is not modelled).
    ``CONSTANT_RETARDANCE``: an achromatic retarder's retardance is its design value across its valid band.
    ``EDGE_AS_OPAQUE_RIM``: the ring between the clear aperture and the outer edge blocks light; light beyond the edge
    passes by.
    ``TYPICAL_SPECTRAL_CURVE``: a coating's reflectance / transmittance is the manufacturer's typical measured curve
    (``FactStatus.TYPICAL``), linearly interpolated between its points; part-to-part variation is not modelled.
    ``CEMENT_LAYER_NOT_MODELLED``: the cement between the two halves of a cemented element (a cube splitter) is not
    modelled: both halves are one medium with a coated internal surface between them.
    ``UNTRACED_FACES_BLOCK``: light that reaches a face of the element's body that is not one of its traced surfaces
    leaves the model there (a run reports it as outside the model's scope).
    ``PRESCRIPTION_FROM_MANUFACTURER_LENS_FILE``: the surface prescription (radii, conic constants, polynomial
    coefficients, thicknesses, materials) is taken from the manufacturer's lens data file of the product.
    ``CRYSTAL_INTERFACE_PER_WAVE``: light crossing a face of a uniaxial crystal splits into the ordinary and extraordinary
    waves, each with the isotropic Fresnel coefficients at its own index (exact at normal incidence and when every optic
    axis is perpendicular to the plane of incidence; at other faces a run warns BIREFRINGENT_INTERFACE_APPROXIMATE once
    the angle of incidence passes its set limit).
    ``CEMENT_LAYER_INTERFERENCE_NOT_MODELLED``: a cement layer between two joined bodies is a slab of its isotropic
    material: light crosses, or is totally reflected at, each of its two faces; interference within the layer is not
    modelled.
    """

    NOMINAL_PRESCRIPTION = "NOMINAL_PRESCRIPTION"
    IDEAL_SURFACE_FIGURE = "IDEAL_SURFACE_FIGURE"
    CATALOG_MATERIAL_DISPERSION = "CATALOG_MATERIAL_DISPERSION"
    COATING_AT_SPECIFIED_BOUND = "COATING_AT_SPECIFIED_BOUND"
    LOSSLESS_COATING = "LOSSLESS_COATING"
    CONSTANT_OPTICAL_DENSITY = "CONSTANT_OPTICAL_DENSITY"
    POLARIZATION_INDEPENDENT_COATING = "POLARIZATION_INDEPENDENT_COATING"
    COATING_PHASE_NOT_MODELLED = "COATING_PHASE_NOT_MODELLED"
    GHOST_REFLECTIONS_NOT_TRACED = "GHOST_REFLECTIONS_NOT_TRACED"
    TRANSMITTED_LIGHT_NOT_TRACED = "TRANSMITTED_LIGHT_NOT_TRACED"
    REFLECTED_LIGHT_NOT_TRACED = "REFLECTED_LIGHT_NOT_TRACED"
    SUBSTRATE_NOT_MODELLED = "SUBSTRATE_NOT_MODELLED"
    THIN_ELEMENT = "THIN_ELEMENT"
    RETARDANCE_INVERSE_WAVELENGTH = "RETARDANCE_INVERSE_WAVELENGTH"
    CONSTANT_RETARDANCE = "CONSTANT_RETARDANCE"
    EDGE_AS_OPAQUE_RIM = "EDGE_AS_OPAQUE_RIM"
    TYPICAL_SPECTRAL_CURVE = "TYPICAL_SPECTRAL_CURVE"
    CEMENT_LAYER_NOT_MODELLED = "CEMENT_LAYER_NOT_MODELLED"
    UNTRACED_FACES_BLOCK = "UNTRACED_FACES_BLOCK"
    PRESCRIPTION_FROM_MANUFACTURER_LENS_FILE = "PRESCRIPTION_FROM_MANUFACTURER_LENS_FILE"
    CRYSTAL_INTERFACE_PER_WAVE = "CRYSTAL_INTERFACE_PER_WAVE"
    CEMENT_LAYER_INTERFERENCE_NOT_MODELLED = "CEMENT_LAYER_INTERFERENCE_NOT_MODELLED"


class MechanicalAccessoryPolicy(StrEnum):
    """How mounting hardware (mounts, posts, stages, bases, breadboards) relates to catalog products.

    ``CONFIGURED_PER_DESIGN``: a design places optical products only; OpenSci configures the mounting hardware for
    each submitted design and lists it, with prices, in that design's Customer Purchase BOM.
    """

    CONFIGURED_PER_DESIGN = "CONFIGURED_PER_DESIGN"
