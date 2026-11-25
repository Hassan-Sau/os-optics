"""Stable machine-readable error / issue codes."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    # input / contract
    INVALID_INPUT = "INVALID_INPUT"
    ZERO_LENGTH_VECTOR = "ZERO_LENGTH_VECTOR"
    NON_UNIT_VECTOR = "NON_UNIT_VECTOR"
    NON_UNIT_QUATERNION = "NON_UNIT_QUATERNION"
    DEGENERATE_SURFACE = "DEGENERATE_SURFACE"
    DUPLICATE_ID = "DUPLICATE_ID"
    UNKNOWN_REFERENCE = "UNKNOWN_REFERENCE"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    # ray geometry
    GRAZING_INCIDENCE = "GRAZING_INCIDENCE"
    PARALLEL_RAY = "PARALLEL_RAY"
    TIR_BOUNDARY = "TIR_BOUNDARY"
    MAX_INTERACTIONS = "MAX_INTERACTIONS"
    MAX_PATHS = "MAX_PATHS"
    MAX_BEAM_PAIRS = "MAX_BEAM_PAIRS"
    RETURN_ANALYSIS_BUDGET = "RETURN_ANALYSIS_BUDGET"
    MEDIUM_MISMATCH = "MEDIUM_MISMATCH"
    BLOCKED_BY_BACK_SIDE = "BLOCKED_BY_BACK_SIDE"
    ZERO_POWER = "ZERO_POWER"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    # aperture / clipping
    CLIPPED = "CLIPPED"
    BEAM_CENTER_OUTSIDE_APERTURE = "BEAM_CENTER_OUTSIDE_APERTURE"
    INCOMPLETE_ENVELOPE = "INCOMPLETE_ENVELOPE"
    MONITOR_MISSED = "MONITOR_MISSED"
    # materials / coatings
    MATERIAL_OUT_OF_RANGE = "MATERIAL_OUT_OF_RANGE"
    MATERIAL_INVALID = "MATERIAL_INVALID"
    COATING_OUT_OF_RANGE = "COATING_OUT_OF_RANGE"
    UNKNOWN_MATERIAL = "UNKNOWN_MATERIAL"
    # a medium's measured spectral bulk loss does not cover the traced wavelength: its absorption there is not established
    # (the light is traced without it; the transmitted power is an upper bound)
    BULK_LOSS_NOT_ESTABLISHED = "BULK_LOSS_NOT_ESTABLISHED"
    # gaussian
    GAUSSIAN_HIGH_NA = "GAUSSIAN_HIGH_NA"
    GAUSSIAN_NEAR_HIGH_NA = "GAUSSIAN_NEAR_HIGH_NA"
    GAUSSIAN_SURFACE_TOO_STRONG = "GAUSSIAN_SURFACE_TOO_STRONG"
    GAUSSIAN_LARGE_AOI = "GAUSSIAN_LARGE_AOI"
    GAUSSIAN_SEVERE_TRUNCATION = "GAUSSIAN_SEVERE_TRUNCATION"
    GAUSSIAN_TRUNCATION = "GAUSSIAN_TRUNCATION"
    GAUSSIAN_TWISTED_ASTIGMATISM = "GAUSSIAN_TWISTED_ASTIGMATISM"
    GAUSSIAN_PLANE_AXIAL_SPREAD = "GAUSSIAN_PLANE_AXIAL_SPREAD"
    GAUSSIAN_INVALIDATED_UPSTREAM = "GAUSSIAN_INVALIDATED_UPSTREAM"
    GAUSSIAN_ABERRATION_DOMINATED = "GAUSSIAN_ABERRATION_DOMINATED"
    # polarization
    RETARDER_OBLIQUE_INCIDENCE = "RETARDER_OBLIQUE_INCIDENCE"
    POLARIZATION_BASIS_DRIFT = "POLARIZATION_BASIS_DRIFT"
    # uniaxial crystals: a use of a crystal outside the model (a curved face, a face that does not refract, light starting or
    # observed inside it, light in it meeting another element); a face met obliquely, where its power split is approximate;
    # a Gaussian beam whose anisotropic diffraction in a crystal exceeds what one index represents
    BIREFRINGENT_MEDIUM_UNSUPPORTED = "BIREFRINGENT_MEDIUM_UNSUPPORTED"
    BIREFRINGENT_INTERFACE_APPROXIMATE = "BIREFRINGENT_INTERFACE_APPROXIMATE"
    GAUSSIAN_ANISOTROPIC_MEDIUM = "GAUSSIAN_ANISOTROPIC_MEDIUM"
    NONLINEAR_CONVERSION_OUT_OF_SCOPE = "NONLINEAR_CONVERSION_OUT_OF_SCOPE"
    HARMONIC_BEAM_NOT_GAUSSIAN = "HARMONIC_BEAM_NOT_GAUSSIAN"
    # radial gradient-index media: a use outside the model (a face that is not a refracting plane perpendicular to the axis, not
    # exactly two faces, light starting or observed inside it, light in it meeting another surface)
    GRADIENT_INDEX_UNSUPPORTED = "GRADIENT_INDEX_UNSUPPORTED"
    # interference / coherence
    COHERENCE_UNDEFINED = "COHERENCE_UNDEFINED"
    NO_FRINGES = "NO_FRINGES"
    BEAMS_NOT_OVERLAPPING = "BEAMS_NOT_OVERLAPPING"
    INCOHERENT_SOURCES = "INCOHERENT_SOURCES"
    WAVELENGTH_MISMATCH = "WAVELENGTH_MISMATCH"
    ORTHOGONAL_POLARIZATION = "ORTHOGONAL_POLARIZATION"
    PHASE_UNDEFINED = "PHASE_UNDEFINED"
    # the two polarization components of a partially polarized source: mutually incoherent, no two-beam quantity between them
    PARTIAL_POLARIZATION_COMPONENTS = "PARTIAL_POLARIZATION_COMPONENTS"
    # a beam's group delay (or two beams' relative delay) has no value: a pass of undetermined group delay, no group index
    GROUP_DELAY_UNDEFINED = "GROUP_DELAY_UNDEFINED"
    # targets / scope
    UNSUPPORTED_METRIC = "UNSUPPORTED_METRIC"
    OUT_OF_MODEL_SCOPE = "OUT_OF_MODEL_SCOPE"
    TARGET_NOT_MET = "TARGET_NOT_MET"
    OBSERVATION_NOT_REACHED = "OBSERVATION_NOT_REACHED"
    POWER_BALANCE_RESIDUAL = "POWER_BALANCE_RESIDUAL"
    SUPERPOSITION_WHOLE_PLANE = "SUPERPOSITION_WHOLE_PLANE"
    # known zero / empty selection / central obstruction
    NO_TRACED_ARRIVAL = "NO_TRACED_ARRIVAL"
    TARGET_SELECTION_EMPTY = "TARGET_SELECTION_EMPTY"
    GAUSSIAN_CENTRAL_OBSTRUCTION = "GAUSSIAN_CENTRAL_OBSTRUCTION"
    ANNULAR_BEAM_NOT_FOLLOWED = "ANNULAR_BEAM_NOT_FOLLOWED"
    # axicon surfaces and tabulated (response-model) thin lenses
    GAUSSIAN_UNDEFINED_AFTER_AXICON = "GAUSSIAN_UNDEFINED_AFTER_AXICON"
    # two-variable sag shapes (Zernike, XY polynomial, biconic, toroid)
    SAG_SHAPE_COMBINATION_NOT_SUPPORTED = "SAG_SHAPE_COMBINATION_NOT_SUPPORTED"
    SAG_NORMALIZATION_MISMATCH = "SAG_NORMALIZATION_MISMATCH"
    SAG_TERM_NOT_SUPPORTED = "SAG_TERM_NOT_SUPPORTED"
    SAG_APERTURE_BEYOND_NORMALIZATION = "SAG_APERTURE_BEYOND_NORMALIZATION"
    SAG_OUTSIDE_DOMAIN = "SAG_OUTSIDE_DOMAIN"
    SAG_INTERSECTION_NOT_CONVERGED = "SAG_INTERSECTION_NOT_CONVERGED"
    MODEL_WAVELENGTH_OUTSIDE_BAND = "MODEL_WAVELENGTH_OUTSIDE_BAND"
    MODEL_FIELD_ANGLE_EXCEEDED = "MODEL_FIELD_ANGLE_EXCEEDED"
    MODEL_BEAM_SIZE_EXCEEDED = "MODEL_BEAM_SIZE_EXCEEDED"
    MODEL_POLARIZATION_NOT_VALIDATED = "MODEL_POLARIZATION_NOT_VALIDATED"
    METRIC_NOT_VALIDATED_FOR_ELEMENT = "METRIC_NOT_VALIDATED_FOR_ELEMENT"
    # diffraction gratings: an order's power that its response does not give, a followed order with no direction
    GRATING_RESPONSE_UNAVAILABLE = "GRATING_RESPONSE_UNAVAILABLE"
    GRATING_ORDER_NONPROPAGATING = "GRATING_ORDER_NONPROPAGATING"
    # power handling: per-surface exposure against the elements' power ratings
    POWER_RATING_EXCEEDED = "POWER_RATING_EXCEEDED"
    POWER_RATING_MISSING = "POWER_RATING_MISSING"
    POWER_RATING_CONDITIONS_MISMATCH = "POWER_RATING_CONDITIONS_MISMATCH"
    POWER_RATING_CONDITIONS_NOT_VERIFIED = "POWER_RATING_CONDITIONS_NOT_VERIFIED"
    INCOMPLETE_TRACE = "INCOMPLETE_TRACE"
    # mutually coherent beams on one surface whose combined exposure depends on their relative phases (a range)
    COHERENT_BEAMS_BOUNDED = "COHERENT_BEAMS_BOUNDED"
    # light the trace leaves out as negligible may reach a surface: its diffraction limit bounds the exposure (a range)
    FAINT_LIGHT_BOUNDED = "FAINT_LIGHT_BOUNDED"
    # light left out as negligible may go back and forth between two surfaces facing each other (a window's faces, a
    # lens's, a linear cavity) and leave them toward any surface: its bound counts every return; where nothing bounds
    # their returns it bounds nothing
    FAINT_LIGHT_RECIRCULATES = "FAINT_LIGHT_RECIRCULATES"
    # light the trace does not follow, followed by probe traces to a plane or surface: a known amount bounds the result
    UNTRACED_LIGHT_BOUNDED = "UNTRACED_LIGHT_BOUNDED"
    # light a surface or a medium scattered out of the traced light and the run does not follow (optics/rays/scatter.py)
    SCATTERED_LIGHT_NOT_FOLLOWED = "SCATTERED_LIGHT_NOT_FOLLOWED"
    # an exposure upper bound the run withholds on one surface: a return pair nothing bounds, or a coating use outside its
    # record; the surface's comparisons and the targets on it state their own outcome, this notice names the surface
    EXPOSURE_BOUND_WITHHELD = "EXPOSURE_BOUND_WITHHELD"
    # a plane through a physical surface's hit point records the beam arriving at that surface, before the interaction
    SURFACE_COINCIDENT_ARRIVAL = "SURFACE_COINCIDENT_ARRIVAL"
    # a beam power added over the beams of a plane where a coherent quantity exists for them: the sum of their powers
    POWER_INCOHERENT_SUM = "POWER_INCOHERENT_SUM"
    # a beam without a valid Gaussian state whose pupil rays a clear aperture cuts: its power from there on is the share of
    # its equal-power pupil samples that pass, within a stated sampling resolution; its Gaussian beam metrics are unavailable
    GEOMETRIC_POWER_BASIS = "GEOMETRIC_POWER_BASIS"
    # a surface takes a bounded share of a beam before a plane its chief ray crosses: the power crossing the plane lies in an
    # interval below the beam's power
    POWER_TAKEN_BEFORE_PLANE = "POWER_TAKEN_BEFORE_PLANE"
    # position-dependent attenuators (variable neutral-density filters)
    ND_DENSITY_VARIES_ACROSS_BEAM = "ND_DENSITY_VARIES_ACROSS_BEAM"
    ND_UNCOATED_REGION = "ND_UNCOATED_REGION"
    # thin complex-transmission elements (optics/complex_transmission.py): phase structure across a beam (no Gaussian beam
    # behind it), an amplitude the beam's power factor is not resolved for, light the phase gradient turns evanescent
    TRANSMISSION_MAP_PHASE_STRUCTURE = "TRANSMISSION_MAP_PHASE_STRUCTURE"
    TRANSMISSION_MAP_VARIES_ACROSS_BEAM = "TRANSMISSION_MAP_VARIES_ACROSS_BEAM"
    TRANSMISSION_MAP_EVANESCENT = "TRANSMISSION_MAP_EVANESCENT"
    # a triangle-facet surface outside the traced domain (an open sheet between two media, a crystal, too many patches)
    FACET_SURFACE_NOT_TRACED = "FACET_SURFACE_NOT_TRACED"
    # part of a beam falls beyond a facet patch's edge onto a neighbouring facet: that share is not followed (fate not known)
    FACET_EDGE_DIVIDES_BEAM = "FACET_EDGE_DIVIDES_BEAM"
    # a beam met flat facets approximating a curved surface (TESSELLATED_FROM_CAD): no Gaussian state behind them
    GAUSSIAN_UNDEFINED_AFTER_FACETS = "GAUSSIAN_UNDEFINED_AFTER_FACETS"
    # detection (contracts/detection.py): a detection quantity a target reads is not available -- a detector parameter is
    # not stated, a wavelength lies outside the quantum-efficiency table, the plane's power is not known, or a model
    # condition does not hold (validation/detection.py)
    DETECTION_QUANTITY_UNAVAILABLE = "DETECTION_QUANTITY_UNAVAILABLE"
    # evaluation scope (contracts/evaluation.py): a target of a narrowed evaluation that a capability outside the executed
    # scope would decide -- no traced beam matches it and the plane bounds did not run; the message names the capability
    CAPABILITY_NOT_COMPUTED = "CAPABILITY_NOT_COMPUTED"
    # notice: facets within the coordinates' rounding of one plane are traced as one flat face whose corners lie farther
    # from it than FITTED_FACE_NOTICE_FRACTION of the part's size (a bend that small in the part is traced flat)
    FACET_FACE_FITTED = "FACET_FACE_FITTED"
    # notice: a sag surface's u_axis lies out of the surface plane by more than unit_vector_tol; its projection onto the
    # plane orients the profile (scene/lower.py)
    U_AXIS_PROJECTED = "U_AXIS_PROJECTED"
    # a sampled source field (a sampled complex field, a speckle field) that its grid does not sample (a phase step of pi or
    # more between neighbouring lit grid points) or whose phase gradient is at or beyond the medium's wavenumber (evanescent)
    SOURCE_FIELD_UNDERSAMPLED = "SOURCE_FIELD_UNDERSAMPLED"
    SOURCE_FIELD_EVANESCENT = "SOURCE_FIELD_EVANESCENT"
    # layout
    INVALID_FOOTPRINT = "INVALID_FOOTPRINT"
    PHYSICAL_INTERSECTION = "PHYSICAL_INTERSECTION"
    KEEPOUT_VIOLATION = "KEEPOUT_VIOLATION"
    BEAM_ENVELOPE_BLOCKED = "BEAM_ENVELOPE_BLOCKED"
    PLATE_BOUNDARY_VIOLATION = "PLATE_BOUNDARY_VIOLATION"
    OPTICAL_HEIGHT_MISMATCH = "OPTICAL_HEIGHT_MISMATCH"
    MOVING_ENVELOPE_VIOLATION = "MOVING_ENVELOPE_VIOLATION"
    TOOL_ACCESS_BLOCKED = "TOOL_ACCESS_BLOCKED"
    CABLE_KEEPOUT_VIOLATION = "CABLE_KEEPOUT_VIOLATION"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    # control
    JACOBIAN_ILL_CONDITIONED = "JACOBIAN_ILL_CONDITIONED"
    UNOBSERVABLE_DOF = "UNOBSERVABLE_DOF"
    DOF_SATURATED = "DOF_SATURATED"
    BEAM_LOST = "BEAM_LOST"
    NOT_RECOVERABLE = "NOT_RECOVERABLE"
    CAPTURE_RANGE_INSUFFICIENT = "CAPTURE_RANGE_INSUFFICIENT"
    LM_NOT_CONVERGED = "LM_NOT_CONVERGED"
    SEED_REQUIRED = "SEED_REQUIRED"
    # a target on a retained DOF's travel, drift or repeatability where the DOF declares none of it: the adjustment is
    # not verified (validation/mechanical.py)
    MECHANICAL_LIMIT_NOT_DECLARED = "MECHANICAL_LIMIT_NOT_DECLARED"
    # optical solids (os_optics.optical_solids: the optical bodies themselves, before any ray is traced)
    SAG_DOMAIN_INSIDE_OUTLINE = "SAG_DOMAIN_INSIDE_OUTLINE"
    SURFACES_INTERSECT = "SURFACES_INTERSECT"
    KNIFE_EDGE = "KNIFE_EDGE"
    MEDIUM_BOUNDARY_INCONSISTENT = "MEDIUM_BOUNDARY_INCONSISTENT"
    OPTICAL_BODIES_OVERLAP = "OPTICAL_BODIES_OVERLAP"
    OPTICAL_BODIES_TOUCH = "OPTICAL_BODIES_TOUCH"
    MEMBER_OUTSIDE_ENVELOPE = "MEMBER_OUTSIDE_ENVELOPE"
