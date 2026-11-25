"""Lower a pure-optical ``ProjectDocumentV2`` + the public optical catalog to an Engine ``ValidationRequest``.

Everything in the request is derived from the Project (poses, the user's explicit source model, observations,
targets) and the published simulation model of each catalog product.

Supported inputs (anything else raises a ``LocalRunError``):

* **source**: a ``custom_part`` component with ``optical_role = source`` and an explicit
  ``GaussianBeamSourceV1`` (``source_model``). It becomes an Engine ``SourceSpec`` with
  ``source_id = instance_id``, emitting along the component's local +Z from its local origin. A single-line source
  gives ``wavelength_nm`` and ``power_W``; a multi-line source (``spectral_lines``) gives ``spectrum`` -- one
  ``SpectralLine`` per line, in the author's order, weighted by the line power -- and ``power_W`` = the sum of the
  line powers, so each line carries its own power. Its polarization becomes the Jones vector in the source's local
  (X, Y) (``source_jones``): linear, or elliptical / circular with the handedness of ``EllipticalPolarizationV1``. A
  partially polarized source (``PARTIAL``) gives the Jones vector of its polarized part and ``degree_of_polarization``
  (omitted at 1, so a PARTIAL source of degree 1 is the fully polarized source, bit for bit); an ``UNPOLARIZED`` one the
  Jones vector of local +X and degree 0 (``source_polarization``). The
  Engine traces every line as its own beam
  (``spectral_index`` = the line's position), and observation results, targets (``wavelength_nm``) and the power
  budget are per line. A ``FLUORESCENCE`` / ``RAMAN`` sample becomes an ``ExtendedEmitter`` whose line powers follow from
  the excitation a first trace delivers to its plane, with its ``SampleEmission`` record (``sample_emission.py``); an
  observation's ``detection`` becomes the Engine's ``Detection`` (a lock-in's modulated shares resolved there too).
* **catalog product**: a ``product`` component whose product is published in the catalog and may play the component's
  ``optical_role`` (``product.design_roles``, from ``DESIGN_ROLES_BY_CATEGORY``). Every product of every published
  category is lowered the same way: ``compile.compile_element`` turns its one published simulation model (or the one
  ``model_kind`` selects) into Engine surfaces in the element frame -- real refracting / reflecting / splitting
  surfaces with the catalog's materials and coatings, or a Jones-matrix polarization element (see ``compile.py``).
  The component's ``nominal_pose`` places that frame. The Engine materials the Project's products use
  (``catalog:<MATERIAL_ID>``) are added to ``ProjectSnapshot.materials``, sorted by id, with the air of the Project's
environment as the Engine material ``air`` (``compile.environment_air``; every native run refracts in and accumulates optical
path in physical air, vacuum at zero pressure; a ``VACUUM`` gap is the built-in ``vacuum``); nothing else in the Engine
  material library changes. Several sources of different wavelengths may share any element: dispersion, coating
  bands and polarization are the Engine's physics, and whether a wavelength or angle stays inside a published model's
  characterization is checked after the run (``validity.py``: ``OUT_OF_MODEL_SCOPE``, never a refusal).
* **declared optical model**: a ``custom_part`` component with an ``optical_model_id`` naming one of the Project's
  ``optical_models``. ``compile.compile_declared`` compiles it with the same rules as a catalog product's model
  (``compile.compile_model``) into Engine surfaces in the element frame; its media and coatings come only from the
  Project's ``model_materials`` / ``model_coatings`` (Engine materials ``project:<MATERIAL_ID>``, never resolved against
  the catalog). A ``COMPOUND`` unit is ONE Engine component holding every member's surfaces at the member poses. The
  component's ``optical_role`` must be one the model kind can play (``declared_model_roles``). A Project whose components
  are all sources, instruments and declared models needs no catalog: none is loaded.
* **terminal instrument**: a ``custom_part`` component with ``optical_role`` ``detector`` (the user's own
  camera, spectrometer, power meter or other detector) or ``dump`` (their beam dump). It has no public optical model:
  it becomes an Engine device without optical surfaces at its nominal pose, and what it does to the beam is stated by
  the absorbing observation plane(s) the Project attaches to it (``device_instance_id``, ``absorbing = true``: its
  sensitive or absorbing face, where the beam is evaluated and stops). Without one it is refused
  (``CUSTOM_PART_UNSUPPORTED``), like every other user-defined component without an ``optical_model_id``.
* observations and targets are copied field for field; ``device_instance_id`` becomes ``device_id``, and a
  target's ``source_id`` names a source component's ``instance_id``. A target metric must be one of
  ``target_metrics()`` (optical observation / two-beam metrics, the geometric pair metrics with their second operand's
  line ``wavelength_nm_b``, and ``power_rating_utilization`` of the element ``device_instance_id``) or an Engine
  out-of-scope metric (which then evaluates to ``OUT_OF_MODEL_SCOPE``); layout and control metrics need mechanical
  envelopes / actuators. A target that cannot be read as written (such a metric, an unknown one, a selector or a
  parameter its metric does not take, a ``source_id`` that names no source) is not refused: it lowers to a ``Target``
  that states only its id, its metric and ``not_assessable`` (the problem's code and message), selects nothing, and is
  reported ``UNKNOWN`` / ``INDETERMINATE`` with that code; ``LoweredProject.target_problems`` lists the problems and the
  check reports each as a warning. The rest of the request is that of the Project without the target.
* **pulsed source**: a source model's ``pulse`` (repetition rate, FWHM duration, and ``total_pulses`` when stated)
  becomes the ``SourceSpec.pulse`` of its Engine source; its powers stay average powers. It feeds the per-surface
  exposure (peak fluence) and the power-rating comparison only.
* **power ratings**: a catalog product's ``specifications.power_ratings`` and a declared model's ``power_ratings`` (a
  ``COMPOUND`` unit: each member's, limited to that member's surfaces) become the Engine component's ``power_ratings``
  (``origin`` names the product or model), compared with the run's per-surface exposure (``ValidationResult.exposures``).
  A ``polarization_condition`` that names a polarization in one of the plain forms of ``RATING_POLARIZATION_TEXT``
  (``s-polarized`` / ``p-polarized`` -- the catalog's form of a published S-Pol / P-Pol --, ``s`` / ``TE``, ``p`` /
  ``TM``, ``linear``, ``circular``, ``unpolarized``, ...; case, surrounding spaces and a final full stop ignored) also
  becomes the rating's machine-readable ``polarization``,
  compared with the polarization of the light at the surface (every beam within the tolerance of one common state); any
  other text stays a condition the comparison reports as not verified.
* **attenuator position**: a ``NEUTRAL_DENSITY`` element whose model names a ``position_parameter``
  (``nd_rotation_deg`` / ``nd_offset_mm``) takes that ``selected_parameters`` entry of its component (a ``DECIMAL``,
  unit ``deg`` / ``mm`` or none; absent: 0) as the wheel rotation / gradient shift. A position parameter on a component
  that does not take it, or in another unit, is refused (``SELECTED_PARAMETERS_UNSUPPORTED``): it is simulation input,
  misapplied.
* **inputs the simulation does not read**: every other ``selected_parameters`` entry (a source is described by its
  ``source_model``, a catalog product by its published model, a declared element by its model) and the Project's
  ``materials`` list (a declared model's media are ``model_materials``). They are passed over, never refused, and listed
  in ``LoweredProject.inputs_not_simulated`` (``inputs_not_simulated``): the check reports each as an
  ``INPUT_NOT_SIMULATED`` warning and the run's summary and manifest carry the list. None of them enters the Engine
  request, so the request and its hash are those of the Project without them.

The source's declared coherence (``linewidth_nm`` or ``coherence_length_mm``, at most one) is passed to the Engine as
given and holds for every spectral line (a linewidth gives each line the coherence length of its own wavelength); a
source that declares neither is taken as monochromatic: the two-beam quantities that depend on the degree of coherence
have values with a ``value_basis`` entry naming that stand-in (assumed ``FULL_COHERENCE``), at a degree of coherence of 1
where neither source of the pair declares its coherence and at the other source's coherence length where it does, and a
target on one of them is judged from 0 up to that value. Where the two paths' relative group delay is not determined, those
quantities are stated at no added delay with an ``ESTIMATE`` entry (assumed ``UNSTATED_GROUP_DELAY_ZERO``) and decide no
target; within coating group-delay bounds a stated coherence gives them with the interval the bounds leave, and where only
one source of the pair declares its coherence they are ``ESTIMATE`` values at the traced delay (assumed ``FULL_COHERENCE``
and ``UNSTATED_GROUP_DELAY_ZERO``) with ``coherence_factor_max`` 1 (the bounds allow any degree of coherence at a shorter
coherence length of the other source) and no ``coherence_factor_min``, a target judged from 0 up to the value at that
end, and so are they where the two paths' group-delay dispersions differ (``coherence_factor_max`` the largest any
shorter coherence length gives, assumed ``FULL_COHERENCE``); where a medium states no group index they are not computed
(``COHERENCE_UNDEFINED``; a target on one is ``UNKNOWN``, other metrics are unaffected). Engine source settings the contract does not carry are the Engine's documented defaults and are part of
the hashed Engine input: each source its own coherence group (beams of one source are mutually coherent, beams of two
sources are mutually incoherent), emission into air, and the pupil sampling of the default ``NumericalConfig``.

Retained DOFs are design intent; the Engine request evaluates the nominal pose. Layout and control metrics
apply to Engine requests with mechanical envelopes and actuators, which a pure-optical Project does not describe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.optical_design.compute_budget import check_compute_budget
from .._vendor.shared_contracts.optical_design.numerical_settings import check_numerical_settings
from .._vendor.shared_contracts.optical_design.design_content import CircularApertureV1, RectangularApertureV1, TargetV1
from .._vendor.shared_contracts.optical_design import (
    EllipticalPolarizationV1,
    ExtendedEmitterSourceV1,
    GaussianBeamSourceV1,
    LinearPolarizationV1,
    OpticalComponentInstanceV1,
    OpticalRoleV1,
    ProjectDocumentV2,
    SampledFieldSourceV1,
    SampledIntensitySourceV1,
    SimulatedPartV1,
    SimulationModelRefV1,
    SpeckleSourceV1,
)
from .._vendor.shared_contracts.optical_design.dof import DofAxisV1, DofMotionV1, dof_local_axis, dof_motion
from .._vendor.shared_contracts.optical_design.optical_model import CompoundModelV1, ProjectOpticalModelV1
from ..contracts.components import (
    ActuatorSpec,
    EngineeringComponentSnapshot,
    PowerRating,
    PulseTrain,
    EmitterField,
    EmitterGrid,
    EmitterIntensityMap,
    EmitterSpeckle,
    ExtendedEmitter,
    SourceSpec,
    SpectralDensity,
    SpectralLine,
    rating_polarization,
)
from ..contracts.geometry import Pose3D
from ..optics.interference.spectrum import bin_width_nm, quadrature_lines
from ..contracts.materials import MaterialOpticalModel
from ..contracts.optics import CircularAperture, RectangularAperture
from ..contracts.polarization import JonesVector
from ..config import NumericalConfig
from ..contracts.project import Observation, ProjectSnapshot, Target, TargetNotAssessable
from ..contracts.request import EngineConfiguration, ValidationRequest
from ..manifest import hash_exact
from ..run_control import checkpoint
from ..validation.alignment import ALIGNMENT_METRICS
from ..validation.exposures import ELEMENT_METRICS
from ..validation.metrics import COHERENCE_METRICS, CONTROL_METRICS, LAYOUT_METRICS, OBSERVATION_METRICS, PAIR_METRICS
from ..contracts.metrics import MetricDescriptor
from ..validation.descriptors import metric_descriptors as _metric_descriptors
from ..validation.mechanical import MECHANICAL_METRICS
from ..validation.detection import DETECTION_METRICS
from ..validation.nonlinear import NONLINEAR_METRICS
from ..validation.scope import out_of_scope_metric
from ..validation.target_parameters import parameter_problems as _parameter_problems
from .catalog import _validation_message, load_project, load_public_catalog, pinned_catalog_version
from .sample_emission import bare_component, complete_request, is_sample
from .compile import (
    environment_air,
    APERTURE_STOP,
    COMPOUND,
    IDEAL_THIN_LENS,
    POLARIZATION_ELEMENT,
    SURFACE_SEQUENCE,
    SURFACE_SET,
    CompiledElement,
    ProjectResolver,
    attenuator_position_parameter,
    compile_declared,
    compile_element,
    declared_model_notices,
    with_attenuator_position,
)
from .design_point import DEFAULT_DESIGN_POINT, DesignPointTolerances
from .errors import LocalRunError, LocalRunErrorCode, LocalRunProblem, fail

# Linear polarization angles whose Jones components are written exactly (no libm trig), so the Engine input
# hash of the common cases is identical on every platform. Other angles use math.cos / math.sin.
EIGHTH_TURN_DEG = 45.0
#: The roles of a user-defined terminal instrument: lowered as a device whose absorbing observation planes state its optics.
TERMINAL_INSTRUMENT_ROLES = frozenset({OpticalRoleV1.DETECTOR, OpticalRoleV1.DUMP})
#: Roles of an optical element (every role but a source and a terminal instrument).
OPTICAL_ELEMENT_ROLES = frozenset(OpticalRoleV1) - TERMINAL_INSTRUMENT_ROLES - {OpticalRoleV1.SOURCE}
#: The ``optical_role`` values a declared model may play, by model kind (a polarization element: by its element).
DECLARED_MODEL_ROLES = {
    SURFACE_SEQUENCE: OPTICAL_ELEMENT_ROLES,
    SURFACE_SET: OPTICAL_ELEMENT_ROLES,
    "RETARDER": frozenset({OpticalRoleV1.WAVEPLATE, OpticalRoleV1.OTHER_OPTICAL}),
    "LINEAR_POLARIZER": frozenset({OpticalRoleV1.POLARIZER, OpticalRoleV1.OTHER_OPTICAL}),
    "CIRCULAR_POLARIZER": frozenset({OpticalRoleV1.POLARIZER, OpticalRoleV1.OTHER_OPTICAL}),
    APERTURE_STOP: frozenset({OpticalRoleV1.APERTURE, OpticalRoleV1.OTHER_OPTICAL}),
    IDEAL_THIN_LENS: frozenset({OpticalRoleV1.LENS, OpticalRoleV1.OTHER_OPTICAL}),
    "TABULATED_THIN_LENS": frozenset({OpticalRoleV1.LENS, OpticalRoleV1.OTHER_OPTICAL}),
    "NEUTRAL_DENSITY": frozenset({OpticalRoleV1.FILTER, OpticalRoleV1.OTHER_OPTICAL}),
    "FARADAY_ROTATOR": frozenset({OpticalRoleV1.OTHER_OPTICAL}),
    "DIFFRACTION_GRATING": frozenset({OpticalRoleV1.OTHER_OPTICAL}),
    COMPOUND: OPTICAL_ELEMENT_ROLES,
    "FACET_SOLID": OPTICAL_ELEMENT_ROLES,
    "NONLINEAR_CRYSTAL": OPTICAL_ELEMENT_ROLES,
}


def declared_model_roles(model: Any) -> frozenset[OpticalRoleV1]:
    """The ``optical_role`` values a component with the declared ``model`` (``ProjectSimulationModelV1``) may play: an
    element of real surfaces (``SURFACE_SEQUENCE``, ``SURFACE_SET``: what it does follows from its surfaces and coatings,
    a polarizing cube or a Fresnel rhomb included) and a ``COMPOUND`` unit any optical element role; a retarder a
    waveplate, a linear polarizer a polarizer; an aperture stop an aperture; an ideal or tabulated thin lens a lens; a
    neutral-density attenuator a filter. ``other_optical`` fits every kind."""

    key = model.element.value if model.model_kind == POLARIZATION_ELEMENT else model.model_kind
    return DECLARED_MODEL_ROLES[key]


def target_metrics() -> dict[str, str]:
    """Target metrics the local optical runtime evaluates: metric name -> unit.

    Observation metrics (``beam_radius_mm``, ``waist_distance_x_mm``, ``power_W``, ...) need an
    ``observation_id``; two-beam metrics (``fringe_period_nm``, ...) also need ``source_id`` / ``source_id_b``.
    The two-beam metrics in ``COHERENCE_METRICS`` also read the source's declared coherence (``linewidth_nm`` or
    ``coherence_length_mm``); without it the line is taken as monochromatic, stated in the value's ``value_basis`` entry,
    and ``coherence_length_mm`` has no value. Where neither source of the pair declares its coherence -- in a Project the
    only such pair -- the values are at a degree of coherence of 1, upper ends (``UPPER_BOUND``) also behind coatings of
    bounded group delay, with no ``coherence_factor_min`` or ``coherence_factor_max``, and a target on one of them is
    judged from 0 up to that value. At the other source's coherence length where that source declares one (a pair of two
    sources sharing a ``coherence_group``, which only the low-level Engine API forms), a target is judged from 0 up to
    that value; where the two paths' group-delay dispersions differ, or behind coatings of bounded group delay, the values
    are ``ESTIMATE`` values at the traced delay, judged up to the value at the pair's ``coherence_factor_max`` (the
    largest any shorter coherence length of the other source gives; 1 behind the bounded coatings).

    Every metric's full meaning -- definition, sign convention, operand order, reference frame, selectors, model basis
    and when it is withheld -- is in ``metric_descriptors()`` (``os-optics results metrics``). In short: ``_x`` /
    ``_y`` are the beam's own transverse axes (the Gaussian beam is x/y independent), not design-frame axes.
    ``beam_radius_mm`` is the 1/e^2 intensity radius at the plane (geometric mean of x and y). ``waist_distance_x_mm`` /
    ``_y_mm`` is the plane position minus the waist position along the beam: 0 means the waist is on the plane, negative
    means the waist lies further downstream. ``power_W`` is the power arriving at the plane.
    ``optical_path_difference_mm`` is ``OPL(a) - OPL(b)`` with ``a`` the beam of ``source_id`` (or
    ``parameters.path_id_a``); ``optical_path_difference_abs_mm`` is its magnitude. ``image_parity`` is +1 / -1 for a
    non-mirrored / mirrored image, ``image_rotation_deg`` 0 upright, 180 inverted (at an image of the source plane).
    """

    metrics = {name: spec[0] for name, spec in OBSERVATION_METRICS.items()}
    metrics.update({name: spec[0] for name, spec in PAIR_METRICS.items()})
    metrics.update({name: spec[0] for name, spec in ALIGNMENT_METRICS.items()})
    metrics.update(ELEMENT_METRICS)
    metrics.update({name: spec[0] for name, spec in MECHANICAL_METRICS.items()})
    metrics.update(DETECTION_METRICS)
    metrics.update(NONLINEAR_METRICS)
    return dict(sorted(metrics.items()))


def metric_descriptors() -> tuple[MetricDescriptor, ...]:
    """The ``MetricDescriptor`` of every metric in ``target_metrics()`` (sorted by name)."""

    return _metric_descriptors()


def pure_optical_engine_configuration(budget: Any = None, settings: Any = None) -> EngineConfiguration:
    """The Engine configuration of the local optical workflow: physics only, deterministic, paths kept, with the
    Project's ``compute_budget`` and ``numerical_settings`` applied field by field over ``NumericalConfig``'s defaults.

    ``budget`` is a ``ComputeBudgetV1`` (or its JSON), ``settings`` a ``NumericalSettingsV1`` (or its JSON), each ``None``
    for the defaults. The budgets decide how much work the run does; the settings are the validity limits, levels and
    match windows a reported number depends on (``numerical_settings.SETTING_NAMES``, ``design_point`` flattened to its
    ``design_point_*`` fields). The result-hash quantization, the reported resolutions and the laboratory reach stay the
    Engine's. The resolved numbers are part of the request and therefore of its exact hash
    (``LoweredProject.engine_input_hash``), so a run is current only for the Project *and* the budget and settings it was
    made with.
    """

    values = {**check_compute_budget(budget), **check_numerical_settings(settings)}
    # (validated construction, never model_copy: the contract's ranges and NumericalConfig's own bounds apply)
    numerical = NumericalConfig(**values) if values else NumericalConfig()
    return EngineConfiguration(numerical=numerical, run_physics=True, run_layout=False, run_control=False, include_paths=True,
                               random_seed=None)


@dataclass(frozen=True)
class LoweredProject:
    """A Project lowered to the Engine: the request, its exact input hash, the public catalog binding (``catalog_version``
    is ``None`` when no component is a catalog product: no catalog was used) and the compiled elements (``elements``, in
    Project order, catalog products and declared models alike: what ``validity.py`` checks and the view draws)."""

    request: ValidationRequest
    engine_input_hash: str
    catalog_version: str | None
    simulated_parts: tuple[SimulatedPartV1, ...]
    #: Every distinct vacuum wavelength the Project's sources emit (all lines of every source), ascending.
    source_wavelengths_nm: tuple[float, ...]
    elements: tuple[CompiledElement, ...] = ()
    #: The optical-solid report of the compiled elements (``os_optics.optical_solids``), checked before any tracing.
    solids: Any = None
    #: The Engine material ``air`` of the Project's environment (``compile.environment_air``): the medium outside the elements
    #: and in every ``AIR`` gap, as the request states it (paraxial and cavity analyses read the same medium).
    air: Any = None
    #: The tolerances of published design points in force (``numerical_settings.design_point`` over the Engine defaults),
    #: resolved once: what ``validity.py``, ``compile.py`` and ``advisories.py`` apply.
    design_point: DesignPointTolerances = DEFAULT_DESIGN_POINT
    #: The ``NumericalConfig`` fields the Project's ``numerical_settings`` states (the others are the Engine defaults).
    stated_settings: tuple[str, ...] = ()
    #: The Project inputs the simulation did not read (``inputs_not_simulated``), in Project order.
    inputs_not_simulated: tuple[dict[str, str], ...] = ()
    #: Why each declared requirement that cannot be read as written is not assessed (``TARGET_INVALID``,
    #: ``TARGET_PARAMETER_INVALID``, ``UNKNOWN_SOURCE_REFERENCE``; location ``target <target_id>[ parameters.<key>]``), in
    #: Project order. Each such target is in the request as a ``Target`` with ``not_assessable`` (``_not_assessable_target``).
    target_problems: tuple[LocalRunProblem, ...] = ()
    #: What the runtime used instead of a stated frame symmetry or design body that contradicts its declared model
    #: (``compile.declared_model_notices``), in ``optical_models`` order.
    model_notices: tuple[dict[str, Any], ...] = ()


def _pose(pose) -> Pose3D:
    return Pose3D(position_mm=pose.position_mm, quaternion_wxyz=pose.quaternion_wxyz)


def _linear_jones(angle_deg: float) -> JonesVector:
    turns = angle_deg / EIGHTH_TURN_DEG
    if turns == math.floor(turns):
        eighth = int(turns) % 8
        half = math.sqrt(0.5)
        cos_sin = ((1.0, 0.0), (half, half), (0.0, 1.0), (-half, half), (-1.0, 0.0), (-half, -half), (0.0, -1.0), (half, -half))[eighth]
        return JonesVector(ex=(cos_sin[0], 0.0), ey=(cos_sin[1], 0.0))
    return JonesVector.linear(angle_deg)


_DECIMAL = Context(prec=28, rounding=ROUND_HALF_EVEN)
_PI = Decimal("3.141592653589793238462643383279502884")
_SERIES_TERMS = 30
_HALF_TURN_DEG = 180


def _cos_sin_deg(degrees: float) -> tuple[Decimal, Decimal]:
    """``(cos, sin)`` of ``degrees`` (|degrees| <= 90) by their power series in exact decimals: no libm function, so the
    hashed Engine input is the same in every runtime."""

    with localcontext(_DECIMAL):
        if degrees % EIGHTH_TURN_DEG == 0.0:  # 0, +-45, +-90: exact
            half = Decimal("0.5").sqrt()
            return {0: (Decimal(1), Decimal(0)), 1: (half, half), 2: (Decimal(0), Decimal(1)), -1: (half, -half),
                    -2: (Decimal(0), Decimal(-1))}[int(degrees // EIGHTH_TURN_DEG)]
        x = Decimal(degrees) * _PI / _HALF_TURN_DEG
        cos, sin, term_c, term_s = Decimal(0), Decimal(0), Decimal(1), x
        for k in range(_SERIES_TERMS):
            cos += term_c
            sin += term_s
            term_c = -term_c * x * x / ((2 * k + 1) * (2 * k + 2))
            term_s = -term_s * x * x / ((2 * k + 2) * (2 * k + 3))
        return cos, sin


def source_jones(polarization: LinearPolarizationV1 | EllipticalPolarizationV1) -> JonesVector:
    """The Engine Jones vector ``(E_x, E_y)`` (local +X, +Y; propagation +Z; field ``exp[i(k.r - w t)]``) of a source
    polarization. Linear at ``angle_deg``: ``(cos a, sin a)``. Elliptical with orientation ``theta`` and ellipticity
    ``chi`` (``EllipticalPolarizationV1``): ``(cos theta cos chi - i sin theta sin chi, sin theta cos chi + i cos theta
    sin chi)``, i.e. the rotation by ``theta`` of ``(cos chi, i sin chi)``, whose normalized Stokes ``s3`` is
    ``sin(2 chi)`` (the results' ``polarization_ellipticity_deg`` is ``chi`` again); ``chi = -45`` deg gives
    ``(1, -i) / sqrt 2``, right-hand circular (clockwise for an observer facing the source,
    ``JonesVector.right_circular``; ``JonesVector.elliptical`` uses the same sign). ``chi = 0`` is the linear state at
    ``theta``, bit for bit."""

    if polarization.kind == "LINEAR" or polarization.ellipticity_deg == 0.0:
        return _linear_jones(polarization.angle_deg if polarization.kind == "LINEAR" else polarization.orientation_deg)
    cos_t, sin_t = _cos_sin_deg(polarization.orientation_deg)
    cos_c, sin_c = _cos_sin_deg(polarization.ellipticity_deg)
    with localcontext(_DECIMAL):
        parts = (cos_t * cos_c, -sin_t * sin_c, sin_t * cos_c, cos_t * sin_c)
    ex_re, ex_im, ey_re, ey_im = (float(part) + 0.0 for part in parts)  # + 0.0: no negative zero
    return JonesVector(ex=(ex_re, ex_im), ey=(ey_re, ey_im))


def source_polarization(polarization: Any) -> tuple[JonesVector, float | None]:
    """The Engine Jones vector and degree of polarization (None: fully polarized) of a source polarization: a linear or
    elliptical one is fully polarized (``source_jones``); ``PARTIAL`` gives its polarized part's Jones vector and its degree
    (None at 1); ``UNPOLARIZED`` the Jones vector of local +X and degree 0 (unpolarized light: equal powers along local +X
    and +Y, ``optics/polarization/partial.py``)."""

    if polarization.kind == "UNPOLARIZED":
        return _linear_jones(0.0), 0.0
    if polarization.kind == "PARTIAL":
        dop = float(polarization.degree_of_polarization)
        return source_jones(polarization.polarized), (None if dop >= 1.0 else dop)
    return source_jones(polarization), None


def _emitter_component(component: OpticalComponentInstanceV1, model: ExtendedEmitterSourceV1) -> EngineeringComponentSnapshot:
    surface = model.emitting_surface
    size = (surface.width_mm, surface.height_mm) if surface.shape == "RECTANGLE" else (surface.diameter_mm, surface.diameter_mm)
    spec = ExtendedEmitter(
        source_id=component.instance_id,
        spectrum=tuple(SpectralLine(wavelength_nm=line.wavelength_nm, weight=line.power_w) for line in model.emission_lines),
        power_W=model.total_power_w, shape=surface.shape, size_x_mm=size[0], size_y_mm=size[1], angular_law=model.angular_law,
        sample_count=model.sampling.sample_count, seed=model.sampling.seed,
    )
    return EngineeringComponentSnapshot(instance_id=component.instance_id, role=OpticalRoleV1.SOURCE.value, pose=_pose(component.nominal_pose), emitter=spec)


def _emitter_grid(grid: Any) -> EmitterGrid:
    return EmitterGrid(columns=grid.column_count, rows=grid.row_count, pitch_x_mm=grid.pitch_mm[0], pitch_y_mm=grid.pitch_mm[1])


def _flat(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(float(v) for row in rows for v in row)


#: The Engine's spelling of a speckle illumination's size: a disc's diameter twice, a rectangle's sides, a Gaussian's radius twice.
def _illumination_size(illumination: Any) -> tuple[float, float]:
    if illumination.shape == "UNIFORM_DISC":
        return illumination.diameter_mm, illumination.diameter_mm
    if illumination.shape == "UNIFORM_RECTANGLE":
        return illumination.width_mm, illumination.height_mm
    return illumination.radius_mm, illumination.radius_mm


def _sampled_component(component: OpticalComponentInstanceV1,
                       model: SampledIntensitySourceV1 | SampledFieldSourceV1 | SpeckleSourceV1) -> EngineeringComponentSnapshot:
    """A sampled source as an Engine emitter (``ExtendedEmitter``): an intensity map's emitting rectangle and law, or a coherent
    field's grid (``COLLIMATED``: its rays follow the field's wavevector)."""

    lines = tuple(SpectralLine(wavelength_nm=line.wavelength_nm, weight=line.power_w) for line in model.emission_lines)
    common: dict[str, Any] = {"source_id": component.instance_id, "spectrum": lines, "power_W": model.total_power_w, "shape": "RECTANGLE"}
    if isinstance(model, SampledIntensitySourceV1):
        m = model.intensity_map
        width, height = m.emitting_size_mm()
        parameter = model.cosine_power_exponent if model.angular_law == "COSINE_POWER" else model.cone_half_angle_deg
        spec = ExtendedEmitter(
            **common, size_x_mm=width, size_y_mm=height, angular_law=model.angular_law, angular_parameter=parameter,
            sample_count=model.sampling.sample_count, seed=model.sampling.seed,
            intensity_map=EmitterIntensityMap(grid=_emitter_grid(m.grid), values=_flat(m.values), interpolation=m.interpolation))
    elif isinstance(model, SampledFieldSourceV1):
        grid = model.grid
        spec = ExtendedEmitter(
            **common, size_x_mm=grid.column_count * grid.pitch_mm[0], size_y_mm=grid.row_count * grid.pitch_mm[1],
            angular_law="COLLIMATED", sample_count=model.ray_sampling.sample_count, seed=model.ray_sampling.seed,
            field=EmitterField(grid=_emitter_grid(grid), amplitude=_flat(model.amplitude), phase_rad=_flat(model.phase_rad),
                               jones=source_jones(model.polarization)))
    else:
        grid = model.grid
        screen = model.phase_screen
        spec = ExtendedEmitter(
            **common, size_x_mm=grid.column_count * grid.pitch_mm[0], size_y_mm=grid.row_count * grid.pitch_mm[1],
            angular_law="COLLIMATED", sample_count=model.ray_sampling.sample_count, seed=model.ray_sampling.seed,
            speckle=EmitterSpeckle(grid=_emitter_grid(grid), illumination=model.illumination.shape,
                                   illumination_size_mm=_illumination_size(model.illumination), phase_rms_rad=screen.phase_rms_rad,
                                   correlation_length_mm=screen.correlation_length_mm, seed=screen.seed,
                                   realizations=model.realizations, jones=source_jones(model.polarization)))
    return EngineeringComponentSnapshot(instance_id=component.instance_id, role=OpticalRoleV1.SOURCE.value, pose=_pose(component.nominal_pose), emitter=spec)


def continuous_emission(density: Any) -> dict[str, Any]:
    """The Engine emission of a Project's continuous spectrum (``SpectralDensityV1``): its ``SpectralDensity`` (the band the
    contract resolves, the shape kind in the Engine's spelling), the quadrature lines it stands for
    (``optics/interference/spectrum.py``) and their linewidth, the bin width."""

    shape = density.shape.model_dump(mode="json")
    shape["kind"] = shape["kind"].lower()
    engine = SpectralDensity(shape=shape, band_nm=density.band(), line_count=density.line_count, quadrature=density.quadrature.lower())
    return {"spectrum": tuple(SpectralLine(wavelength_nm=w, weight=p) for w, p in quadrature_lines(engine)),
            "spectral_density": engine, "linewidth_nm": bin_width_nm(engine)}


def _source_component(component: OpticalComponentInstanceV1, model: GaussianBeamSourceV1) -> EngineeringComponentSnapshot:
    if is_sample(model):  # (its emitter follows from the excitation trace: local/sample_emission.py)
        return bare_component(component, _pose(component.nominal_pose))
    if isinstance(model, ExtendedEmitterSourceV1):
        return _emitter_component(component, model)
    if isinstance(model, (SampledIntensitySourceV1, SampledFieldSourceV1, SpeckleSourceV1)):
        return _sampled_component(component, model)
    if model.spectral_density is not None:
        emission = continuous_emission(model.spectral_density)
    elif model.spectral_lines is None:
        emission = {"wavelength_nm": model.wavelength_nm}
    else:
        emission = {"spectrum": tuple(SpectralLine(wavelength_nm=line.wavelength_nm, weight=line.power_w) for line in model.spectral_lines)}
    linewidth_nm = emission.pop("linewidth_nm", model.linewidth_nm)
    jones, degree_of_polarization = source_polarization(model.polarization)
    spec = SourceSpec(
        source_id=component.instance_id,
        **emission,
        power_W=model.total_power_w,
        waist_radius_x_mm=model.waist_radius_mm,
        waist_radius_y_mm=model.waist_radius_y_mm if model.waist_radius_y_mm is not None else model.waist_radius_mm,
        waist_position_z_mm=model.waist_position_mm,
        m2_x=model.m2,
        m2_y=model.m2_y if model.m2_y is not None else model.m2,
        polarization=jones,
        degree_of_polarization=degree_of_polarization,
        linewidth_nm=linewidth_nm,
        coherence_length_mm=model.coherence_length_mm,
        pulse=None if model.pulse is None else PulseTrain(repetition_rate_hz=model.pulse.repetition_rate_hz,
                                                          pulse_duration_s=model.pulse.pulse_duration_s,
                                                          duration_definition=model.pulse.duration_definition,
                                                          total_pulses=model.pulse.total_pulses),
    )
    return EngineeringComponentSnapshot(instance_id=component.instance_id, role=OpticalRoleV1.SOURCE.value, pose=_pose(component.nominal_pose), source=spec)


def element_component(component: OpticalComponentInstanceV1, element: CompiledElement,
                      power_ratings: tuple[PowerRating, ...] = ()) -> EngineeringComponentSnapshot:
    """The Engine component of a compiled element placed at the Project component's nominal pose."""

    return EngineeringComponentSnapshot(instance_id=component.instance_id, role=component.optical_role.value, pose=_pose(component.nominal_pose),
                                        surfaces=element.surfaces, power_ratings=power_ratings)


def _decimal_number(value: Any) -> float | None:
    from decimal import Decimal

    return None if value is None else float(Decimal(value))


def engine_power_ratings(ratings: Any, origin: str, surface_ids: tuple[str, ...] = (),
                         surface_kind: str | None = None) -> tuple[PowerRating, ...]:
    """Published / declared ``PowerRatingV1`` facts as Engine ``PowerRating`` inputs (decimals to their nearest binary64;
    the polarization condition also in machine-readable form where its text states one, ``rating_polarization``), for a
    surface of ``surface_kind`` (``METALLIC_REFLECTOR``, ``OTHER``; ``None``: a reflector of unknown kind)."""

    return tuple(PowerRating(
        regime=r.regime, value=_decimal_number(r.value), unit=r.unit, wavelength_nm=_decimal_number(r.wavelength_nm),
        pulse_duration_s=_decimal_number(r.pulse_duration_s), repetition_rate_hz=_decimal_number(r.repetition_rate_hz),
        beam_diameter_mm=_decimal_number(r.beam_diameter_mm), beam_diameter_definition=r.beam_diameter_definition,
        pulse_count=r.pulse_count, test_protocol=r.test_protocol, value_definition=r.value_definition,
        polarization_condition=r.polarization_condition,
        polarization=rating_polarization(r.polarization_condition), status=r.status.value, basis=r.basis,
        origin=origin, surface_ids=surface_ids, surface_kind=surface_kind) for r in ratings or ())


def catalog_surface_kind(product: Any) -> str:
    """The kind of surface a catalog product's ratings are for: ``METALLIC_REFLECTOR`` when the product's coating is a
    metallic reflector, else ``OTHER``."""

    return "METALLIC_REFLECTOR" if str(product.specifications.coating) == "METALLIC_REFLECTOR" else "OTHER"


def declared_surface_kind(model: ProjectOpticalModelV1, project: ProjectDocumentV2) -> str | None:
    """The kind of surface a declared model's ratings are for, from what the Project states: ``METALLIC_REFLECTOR`` when a
    surface carries a metallic-reflector coating; ``None`` when a surface reflects without a stated coating (a reflector of
    unknown kind), a neutral-density element reflects part of the light, or the model is an aperture stop (its plate's
    material is not stated); ``OTHER`` otherwise (dielectric coatings, uncoated interfaces, transmissive elements)."""

    inner = model.model
    kind = getattr(inner, "model_kind", None)
    if kind in ("SURFACE_SEQUENCE", "SURFACE_SET"):
        coatings = {c.coating_id: str(c.kind) for c in project.model_coatings or ()}
        unknown = False
        for surface in inner.surfaces:
            coating = coatings.get(surface.coating_id) if surface.coating_id is not None else None
            if coating == "METALLIC_REFLECTOR":
                return "METALLIC_REFLECTOR"
            if str(surface.traced) == "REFLECTED" and coating is None:
                unknown = True
        return None if unknown else "OTHER"
    if kind == "NEUTRAL_DENSITY":
        reflectance = getattr(inner, "reflectance", None)
        return None if reflectance is not None and _decimal_number(reflectance) > 0.0 else "OTHER"  # type: ignore[operator]
    if kind == "APERTURE_STOP":
        return None
    return "OTHER"


def _element_ratings(element: CompiledElement, project: ProjectDocumentV2, product: Any = None) -> tuple[PowerRating, ...]:
    """The power ratings of a compiled element: the catalog product's, or the declared model's (a compound unit's members',
    each on its own surfaces), with the kind of surface they are for (``catalog_surface_kind``,
    ``declared_surface_kind``)."""

    if product is not None:
        return engine_power_ratings(product.specifications.power_ratings, f"catalog product {product.optical_sku}",
                                    surface_kind=catalog_surface_kind(product))
    if element.is_compound:
        ratings: list[PowerRating] = []
        for member in element.members:
            declared = project.optical_model(member.optical_model_id) if member.optical_model_id is not None else None
            if declared is not None and declared.power_ratings:
                ratings.extend(engine_power_ratings(declared.power_ratings, f"model {declared.model_id} (member {member.member_id})",
                                                    tuple(s.surface_id for s in member.surfaces),
                                                    declared_surface_kind(declared, project)))
        return tuple(ratings)
    declared = project.optical_model(element.optical_model_id) if element.optical_model_id is not None else None
    if declared is None or not declared.power_ratings:
        return ()
    return engine_power_ratings(declared.power_ratings, f"model {element.optical_model_id}",
                                surface_kind=declared_surface_kind(declared, project))


#: The unit a position parameter may state (none: the same unit implied).
POSITION_UNITS = {"nd_rotation_deg": "deg", "nd_offset_mm": "mm", "axis_rotation_deg": "deg"}


def inputs_not_simulated(project: ProjectDocumentV2) -> tuple[dict[str, str], ...]:
    """The inputs of ``project`` the simulation does not read (module docstring), in Project order: each
    ``{"location", "field", "key"}`` -- ``materials`` entries by their ``material_id`` (location ``materials``), then
    per component its ``selected_parameters`` entries that are not a position parameter, by their ``semantic_key``
    (location ``component <instance_id>``)."""

    rows = [{"location": "materials", "field": "materials", "key": str(getattr(material, "material_id", index + 1))}
            for index, material in enumerate(project.materials or ())]
    for component in project.components:
        rows.extend({"location": f"component {component.instance_id}", "field": "selected_parameters", "key": parameter.semantic_key}
                    for parameter in component.selected_parameters if parameter.semantic_key not in POSITION_UNITS)
    return tuple(rows)


def _positioned(component: OpticalComponentInstanceV1, element: CompiledElement, where: str
                ) -> tuple[CompiledElement, list[LocalRunProblem]]:
    """The element with its attenuator position applied, and a problem for every position parameter of the component that
    is not this element's (or not a decimal in its unit). Other ``selected_parameters`` entries are not simulation input
    (``inputs_not_simulated``)."""

    key = attenuator_position_parameter(element)
    problems: list[LocalRunProblem] = []
    for parameter in component.selected_parameters:
        if parameter.semantic_key not in POSITION_UNITS:
            continue  # (not simulation input: listed by inputs_not_simulated)
        if key is None or parameter.semantic_key != key:
            expected = f"; this element takes only {key!r}" if key is not None else ""
            problems.append(LocalRunProblem(
                LocalRunErrorCode.SELECTED_PARAMETERS_UNSUPPORTED,
                f"selected parameter {parameter.semantic_key!r} is not simulation input here{expected} (a source is described by "
                "source_model, a catalog product by its published model; an attenuator's position by its model's position_parameter)",
                where))
            continue
        if parameter.value_kind.value != "DECIMAL" or parameter.unit_code not in (None, POSITION_UNITS[key]):
            problems.append(LocalRunProblem(
                LocalRunErrorCode.SELECTED_PARAMETERS_UNSUPPORTED,
                f"{key} is a DECIMAL in {POSITION_UNITS[key]} (unit_code {POSITION_UNITS[key]!r} or none)", where))
            continue
        element = with_attenuator_position(element, _decimal_number(parameter.value))  # type: ignore[arg-type]
    return element, problems


def _aperture(aperture: CircularApertureV1 | RectangularApertureV1 | None) -> CircularAperture | RectangularAperture | None:
    if aperture is None:
        return None
    if aperture.shape == "circle":
        return CircularAperture(radius_mm=aperture.radius_mm, center_offset_mm=aperture.center_offset_mm, obstruction_radius_mm=aperture.obstruction_radius_mm)
    return RectangularAperture(half_width_mm=aperture.half_width_mm, half_height_mm=aperture.half_height_mm, center_offset_mm=aperture.center_offset_mm)


def _target_metric_problem(target: TargetV1) -> LocalRunProblem | None:
    where = f"target {target.target_id}"
    if target.metric in LAYOUT_METRICS or target.metric in CONTROL_METRICS:
        return LocalRunProblem(
            LocalRunErrorCode.TARGET_INVALID,
            f"metric {target.metric!r} needs mechanical envelopes or actuators, which a pure-optical design does not have",
            where,
        )
    if out_of_scope_metric(target.metric) is None and not any(
            target.metric in registry for registry in (OBSERVATION_METRICS, PAIR_METRICS, ALIGNMENT_METRICS, ELEMENT_METRICS,
                                                       MECHANICAL_METRICS, NONLINEAR_METRICS, DETECTION_METRICS)):
        return LocalRunProblem(
            LocalRunErrorCode.TARGET_INVALID,
            f"unknown metric {target.metric!r}; supported metrics ('os-optics results metrics' describes each): "
            f"{', '.join(target_metrics())}",
            where,
        )
    return None


def _detection_target_problem(target: TargetV1, project: ProjectDocumentV2) -> LocalRunProblem | None:
    """Why a detection-metric target cannot be read: its plane states no detector (``ObservationV1.detection``), it names a
    source or a wavelength (a detector reads all the light on its plane), or ``lock_in_snr`` on a plane without a lock-in."""

    where = f"target {target.target_id}"
    plane = next((o for o in project.observations if o.observation_id == target.observation_id), None)
    if plane is None or plane.detection is None:
        return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, f"metric {target.metric!r} reads the detector of its observation plane: "
                               "give the target an observation_id whose plane states detection (a detector)", where)
    if target.source_id is not None or target.source_id_b is not None or target.wavelength_nm is not None:
        return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, f"metric {target.metric!r} reads every source and wavelength the "
                               "detector receives: leave source_id and wavelength_nm unset", where)
    if target.metric == "lock_in_snr" and plane.detection.lock_in is None:
        return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, f"metric 'lock_in_snr' needs a lock-in on observation "
                               f"{plane.observation_id} (detection.lock_in: modulation, time constant, filter order)", where)
    return None


#: A retained DOF's motion -> the Engine actuator's ``dof``; the unit of each.
ACTUATOR_DOF = {DofMotionV1.TRANSLATION: "translate", DofMotionV1.ROTATION: "rotate"}
DOF_UNITS = {"translate": "mm", "rotate": "deg"}


def dof_actuators(component: OpticalComponentInstanceV1) -> tuple[ActuatorSpec, ...]:
    """The Engine actuators of a component's retained DOFs that state a travel, one per DOF, named by its axis (``Tx`` ...
    ``Rz``), about the component's local origin, carrying the stated hard limits, repeatability and drift
    (``validation/mechanical.py`` compares them with the ``dof_*`` targets). A DOF without a travel lowers to nothing, so a
    Project that states no limits keeps its Engine input and hash."""

    actuators = []
    for dof in component.retained_dofs:
        if dof.travel is None:
            continue
        actuators.append(ActuatorSpec(
            actuator_id=dof.axis.value, dof=ACTUATOR_DOF[dof_motion(dof.axis)], axis_local=dof_local_axis(dof.axis),
            travel_min=dof.travel.min, travel_max=dof.travel.max,
            hard_limit_min=None if dof.hard_limits is None else dof.hard_limits.min,
            hard_limit_max=None if dof.hard_limits is None else dof.hard_limits.max,
            repeatability=dof.repeatability,
            drift_per_kelvin=None if dof.drift is None else dof.drift.per_kelvin,
            drift_per_hour=None if dof.drift is None else dof.drift.per_hour,
        ))
    return tuple(actuators)


def _dof_target_problem(target: TargetV1, project: ProjectDocumentV2) -> LocalRunProblem | None:
    """Why a ``dof_*`` target names no retained DOF it can read, or asks nothing (``validation/mechanical.py``); None when it
    is evaluable. A DOF that states no limits is not a problem here: the run reports the target undetermined."""

    where = f"target {target.target_id}"
    _unit, kind, motion = MECHANICAL_METRICS[target.metric]
    parameters = {p.key: p.value for p in target.parameters}
    component = next((c for c in project.components if c.instance_id == target.device_instance_id), None)
    if component is None:
        return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, f"metric {target.metric} names a retained DOF: device_instance_id "
                               "is the component and parameters.axis its axis", where)
    axis = parameters.get("axis")
    retained = [dof.axis.value for dof in component.retained_dofs]
    if axis not in retained:
        held = ", ".join(retained) or "none"
        return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, f"parameters.axis must name a retained DOF of component "
                               f"{component.instance_id} (retained: {held})", where)
    actual = ACTUATOR_DOF[dof_motion(DofAxisV1(axis))]
    if actual != motion:
        stem = target.metric.rsplit("_", 1)[0]
        return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID,
                               f"{axis} is in {DOF_UNITS[actual]}: use the metric {stem}_{DOF_UNITS[actual]}", where)
    if kind == "travel":
        low, high = parameters.get("required_min"), parameters.get("required_max")
        if low is None and high is None:
            return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, "state the adjustment the design needs: parameters.required_min "
                                   "and / or required_max (relative to the nominal pose)", where)
        if low is not None and high is not None and float(low) > float(high):
            return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, "parameters.required_min must not exceed required_max", where)
    if kind == "drift":
        spans = [parameters.get(key) for key in ("temperature_span_K", "duration_h")]
        if all(span is None for span in spans):
            return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, "state the span the drift is taken over: "
                                   "parameters.temperature_span_K and / or duration_h", where)
        if any(span is not None and float(span) <= 0.0 for span in spans):
            return LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, "a drift span (temperature_span_K, duration_h) is positive", where)
    return None


def _target_problems(target: TargetV1, project: ProjectDocumentV2, source_ids: set[str]) -> list[LocalRunProblem]:
    """Why a declared requirement cannot be read as written (empty when it can): a source reference that is not a source,
    a metric or selector the requirement cannot be read with, a parameter its metric does not take."""

    where = f"target {target.target_id}"
    unknown = [s for s in (target.source_id, target.source_id_b) if s is not None and s not in source_ids]
    if unknown:
        return [LocalRunProblem(LocalRunErrorCode.UNKNOWN_SOURCE_REFERENCE, f"source reference {unknown} is not a source component instance_id", where)]
    metric_problem = _target_metric_problem(target)
    if metric_problem is None and target.metric in MECHANICAL_METRICS:
        metric_problem = _dof_target_problem(target, project)
    if metric_problem is None and target.metric in DETECTION_METRICS:
        metric_problem = _detection_target_problem(target, project)
    if metric_problem is not None:
        return [metric_problem]
    return [LocalRunProblem(LocalRunErrorCode.TARGET_PARAMETER_INVALID, message, f"{where} parameters.{key}")
            for key, message in _parameter_problems(target.metric, {p.key: p.value for p in target.parameters})]


def _not_assessable_target(target: TargetV1, problems: list[LocalRunProblem]) -> Target:
    """The Engine target of a requirement that cannot be read as written: its id, its metric and why (the first problem's
    code; every problem's message). It selects no beam, plane, element or analysis, so the physics of the run is that of
    the Project without it, and the run reports it ``UNKNOWN`` (assessment ``INDETERMINATE``) with that code."""

    return Target(target_id=target.target_id, metric=target.metric, not_assessable=TargetNotAssessable(
        code=problems[0].code.value, message="; ".join(p.message for p in problems)))


def project_materials(elements: tuple[CompiledElement, ...] | list[CompiledElement], air: MaterialOpticalModel | None = None
                      ) -> tuple[MaterialOpticalModel, ...]:
    """The Engine materials the compiled elements use, each once, sorted by id (the ``ProjectSnapshot.materials``), with the
    environment's air as the Engine material ``air`` (``compile.environment_air``: it replaces the Engine's unit-index
    built-in, so every native run refracts and accumulates optical path in physical air; vacuum at zero pressure). A
    ``VACUUM`` gap is the Engine's built-in ``vacuum``."""

    by_id: dict[str, MaterialOpticalModel] = {} if air is None else {air.material_id: air}
    for element in elements:
        for material in element.materials:
            by_id.setdefault(material.material_id, material)
    return tuple(by_id[key] for key in sorted(by_id))


def _unplaced_models(project: ProjectDocumentV2) -> list[ProjectOpticalModelV1]:
    """The Project's declared models that no component uses, directly or as a member of a unit it uses."""

    placed: set[str] = set()
    pending = [c.optical_model_id for c in project.components if c.optical_model_id is not None]
    while pending:
        model_id = pending.pop()
        if model_id in placed:
            continue
        placed.add(model_id)
        declared = project.optical_model(model_id)
        if declared is not None and isinstance(declared.model, CompoundModelV1):
            pending.extend(member.model_id for member in declared.model.members)
    return [m for m in project.optical_models or () if m.model_id not in placed]


def solid_check(elements: Any, project: Any, *, refuse: bool = True) -> Any:
    """The optical-solid check of compiled ``elements`` at the nominal poses of ``project``'s components (the capability
    ``solid_check``): the ``os_optics.optical_solids`` report, refused (``LocalRunError`` carrying the whole report) when
    ``refuse`` and a finding is an ERROR."""

    from ..optical_solids import inspect_elements

    solids = inspect_elements(elements, {c.instance_id: c.nominal_pose for c in project.components})
    if refuse and not solids.valid:
        # each finding's structured fields travel with its problem, and the whole report with the refusal
        raise LocalRunError(solids.local_run_problems(), optical_solids=solids.as_dict())
    return solids


def _declared_context(project: Any) -> str:
    """The digest of what a declared model compiles from beside itself: the Project's models, materials and coatings."""

    return hash_exact({"optical_models": [m.model_dump(mode="json") for m in project.optical_models or ()],
                       "model_materials": [m.model_dump(mode="json") for m in project.model_materials or ()],
                       "model_coatings": [c.model_dump(mode="json") for c in project.model_coatings or ()]})


def _compiled(cache: dict[Any, Any] | None, key: tuple[Any, ...], compile_: Any) -> Any:
    """``compile_()``, or the element ``cache`` holds under ``key`` (a compiled element depends on its model, materials,
    coatings, catalog release and model kind, never on its pose)."""

    if cache is None:
        return compile_()
    element = cache.get(key)
    if element is None:
        element = cache[key] = compile_()
    return element


def lower_project(project: Any, catalog: Any = None, *, model_kind: str | None = None, check_solids: bool = True,
                  require_source: bool = True, inspect_solids: bool = True,
                  compiled: dict[Any, Any] | None = None, engine_state: Any = None) -> LoweredProject:
    """Lower ``project`` against ``catalog`` (default: the release the Project's components pin, else the active
    catalog; ``load_public_catalog``).

    Both accept the same inputs as ``load_project`` / ``load_public_catalog``; a Project without catalog components
    loads no catalog. ``model_kind`` selects the simulation model of every catalog component when a product publishes
    several. Raises ``LocalRunError`` listing every unsupported input.

    Before the Engine request is built, the compiled optical bodies are checked as solids (``os_optics.optical_solids``,
    for every design intent; no lab mechanics): every profile defined over its outline, a positive thickness between
    consecutive surfaces over the whole outline, media that chain, no two placed optical bodies (compound members
    included) sharing volume, every member inside its unit's envelope. Each ERROR finding is a ``LocalRunProblem`` with its
    code (``SAG_DOMAIN_INSIDE_OUTLINE``, ``SURFACES_INTERSECT``, ``MEDIUM_BOUNDARY_INCONSISTENT``, ``OPTICAL_BODIES_OVERLAP``,
    ``MEMBER_OUTSIDE_ENVELOPE``, ``OPTICAL_BODIES_TOUCH`` where light would cross from one body into the other), location
    (``component <id>[ member <m>][ surface <s>]``) and ``details`` (the other body, position, depth); the refusal carries
    the whole report (``LocalRunError.optical_solids``). ``check_solids=False``
    reports the findings in ``LoweredProject.solids`` without refusing (``json_api.inspect_optical_solids``).
    ``require_source=False`` also lowers a Project without a light source (a paraxial analysis that uses none).
    ``inspect_solids=False`` leaves the check out (``LoweredProject.solids`` is ``None``): an evaluation that does not
    request the capability ``solid_check`` (``os_optics.local.EvaluationSession``). ``compiled``: a dictionary the caller
    keeps across lowerings, holding the compiled elements of the latest Project lowered with it, by what they compile
    from (the product, catalog release and model kind, or the declared model with the Project's models, materials and
    coatings), the environment and the design-point tolerances. A lowering compiles missing elements and drops unused cache
    entries. Moving an element without changing these inputs reuses its compilation. ``engine_state``:
    the ``EvaluationState`` of the evaluations that follow (an ``EvaluationSession``'s), through which the excitation
    trace of a ``FLUORESCENCE`` / ``RAMAN`` sample identifies the Engine build (``sample_emission.complete_request``).
    """

    project = load_project(project)
    if project.configurations:
        ids = ", ".join(c.configuration_id for c in project.configurations)
        raise fail(LocalRunErrorCode.CONFIGURATION_NOT_SELECTED,
                   f"the Project declares configurations ({ids}): each is evaluated as its own Project -- run them all with "
                   "'os-optics project run' (one with '--configuration ID'; run_configured_project), or name the one to "
                   "evaluate or analyse: '--configuration ID' of 'project evaluate', 'project paraxial', 'project cavity' and "
                   "'project network', the request field configuration_id of a paraxial or cavity analysis, "
                   "configuration_id= of evaluate_project and the network export, or ProjectDocumentV2.configured(id)",
                   "configurations")
    if catalog is not None or any(c.product is not None for c in project.components):
        catalog = load_public_catalog(catalog, catalog_version=pinned_catalog_version(project) if catalog is None else None)
    environment = project.effective_environment
    air = environment_air(environment)
    environment_key = (environment.temperature_c, environment.pressure_pa)
    problems: list[LocalRunProblem] = []
    # (the Engine configuration first: the design-point tolerances in it are what every element compiles under)
    stated_settings = tuple(check_numerical_settings(project.numerical_settings))
    try:
        configuration = pure_optical_engine_configuration(project.compute_budget, project.numerical_settings)
    except ValidationError as exc:
        problems.append(LocalRunProblem(LocalRunErrorCode.ENGINE_INPUT_INVALID, _validation_message(exc)))
        configuration = pure_optical_engine_configuration()
    design_point = DesignPointTolerances.of(configuration.numerical, stated_settings)
    resolver = ProjectResolver(project, design_point)

    sources: dict[str, Any] = {}
    instruments: dict[str, OpticalComponentInstanceV1] = {}
    elements: list[CompiledElement] = []
    ratings: dict[str, tuple[PowerRating, ...]] = {}
    context: str | None = None  # (the declared models' compile context, when a cache keeps compiled elements)
    used: set[Any] = set()  # (the keys of the caller's compiled elements this lowering reads or writes)
    for component in project.components:
        checkpoint()  # (a monitored run: one component lowered -- os_optics.run_control; nothing else changes)
        where = f"component {component.instance_id}"
        element_like = component.optical_role is not OpticalRoleV1.SOURCE and (component.product is not None or component.optical_model_id is not None)
        misapplied = [] if element_like else [p.semantic_key for p in component.selected_parameters if p.semantic_key in POSITION_UNITS]
        if misapplied:  # (a position parameter is simulation input: on a component that has no such position it is refused)
            problems.append(LocalRunProblem(
                LocalRunErrorCode.SELECTED_PARAMETERS_UNSUPPORTED,
                f"selected parameter {', '.join(repr(key) for key in misapplied)} positions a variable attenuator; this component "
                "is not one (an attenuator's position is the selected parameter its model names in position_parameter)",
                where,
            ))
        if component.optical_role is OpticalRoleV1.SOURCE:
            if component.product is not None:
                problems.append(LocalRunProblem(LocalRunErrorCode.SOURCE_PRODUCT_UNSUPPORTED, "catalog source products are not published; describe the source with a custom_part and source_model", where))
            elif component.source_model is None:
                problems.append(LocalRunProblem(LocalRunErrorCode.SOURCE_MODEL_MISSING, "a source needs an explicit source_model (GAUSSIAN_BEAM: wavelength, power, waist, M^2, polarization; EXTENDED_EMITTER: wavelength, power, emitting surface, angular law, sampling; SAMPLED_INTENSITY: wavelength, power, intensity map, angular law, sampling; SAMPLED_FIELD: wavelength, grid, amplitude, phase, polarization, ray sampling; SPECKLE: wavelength, power, illumination, phase screen, grid, polarization, ray sampling)", where))
            else:
                sources[component.instance_id] = component.source_model
            continue
        if component.product is None:
            if component.optical_role in TERMINAL_INSTRUMENT_ROLES:
                if any(o.device_instance_id == component.instance_id and o.absorbing for o in project.observations):
                    instruments[component.instance_id] = component
                else:
                    problems.append(LocalRunProblem(
                        LocalRunErrorCode.CUSTOM_PART_UNSUPPORTED,
                        f"a custom {component.optical_role.value} has no public simulation model: attach an absorbing observation "
                        "plane to it (device_instance_id, absorbing = true) as its sensitive or absorbing face",
                        where,
                    ))
                continue
            if component.optical_model_id is None:
                problems.append(LocalRunProblem(
                    LocalRunErrorCode.CUSTOM_PART_UNSUPPORTED,
                    f"a custom {component.optical_role.value} needs its optical model: declare it in the Project's optical_models "
                    "(with its model_materials / model_coatings) and name it with optical_model_id",
                    where,
                ))
                continue
            declared = project.optical_model(component.optical_model_id)
            assert declared is not None  # ProjectDocumentV2 resolves every optical_model_id
            allowed = declared_model_roles(declared.model)
            if component.optical_role not in allowed:
                problems.append(LocalRunProblem(
                    LocalRunErrorCode.OPTICAL_MODEL_ROLE_MISMATCH,
                    f"model {declared.model_id} is a {declared.model.model_kind} model; it may play optical_role "
                    f"{', '.join(sorted(r.value for r in allowed))}, not {component.optical_role.value}",
                    where,
                ))
                continue
            try:
                if compiled is not None and context is None:
                    context = _declared_context(project)
                key = ("declared", component.instance_id, declared.model_id, context, design_point, environment_key)
                used.add(key)
                element, position_problems = _positioned(component, _compiled(compiled, key, lambda: compile_declared(
                    component.instance_id, declared, resolver, where=where)), where)
                problems.extend(position_problems)
                elements.append(element)
                ratings[component.instance_id] = _element_ratings(element, project)
            except LocalRunError as exc:
                problems.extend(exc.problems)
            except ValidationError as exc:
                problems.append(LocalRunProblem(LocalRunErrorCode.ENGINE_INPUT_INVALID, _validation_message(exc), where))
            continue
        ref = component.product
        if ref.catalog_version is not None and ref.catalog_version != catalog.catalog_version:
            problems.append(LocalRunProblem(
                LocalRunErrorCode.CATALOG_VERSION_MISMATCH,
                f"designed against public catalog {ref.catalog_version}, but the runtime catalog is {catalog.catalog_version}",
                where,
            ))
            continue
        product = catalog.get(ref.optical_sku)
        if product is None:
            problems.append(LocalRunProblem(LocalRunErrorCode.OPTICAL_SKU_NOT_PUBLISHED, f"{ref.optical_sku} is not published in public catalog {catalog.catalog_version}", where))
            continue
        if component.optical_role.value not in product.design_roles:
            allowed = ", ".join(sorted(product.design_roles)) or "none"
            problems.append(LocalRunProblem(
                LocalRunErrorCode.PRODUCT_ROLE_MISMATCH,
                f"{ref.optical_sku} is a {product.category.value} product ({product.form.value}); it may play optical_role {allowed}, "
                f"not {component.optical_role.value}",
                where,
            ))
            continue
        try:
            # (the release by its version and the object the caller's cache keeps alive with it)
            key = ("product", component.instance_id, ref.optical_sku, catalog.catalog_version, id(catalog), model_kind, design_point, environment_key)
            used.add(key)
            element, position_problems = _positioned(component, _compiled(compiled, key, lambda: compile_element(
                component.instance_id, product, catalog, model_kind=model_kind, environment=environment, design_point=design_point)), where)
            problems.extend(position_problems)
            elements.append(element)
            ratings[component.instance_id] = _element_ratings(element, project, product)
        except LocalRunError as exc:
            problems.extend(exc.problems)
        except ValidationError as exc:
            problems.append(LocalRunProblem(LocalRunErrorCode.ENGINE_INPUT_INVALID, _validation_message(exc), where))
    if compiled is not None:
        # (the dictionary holds the compiled elements of the latest lowering: what this Project does not use is dropped)
        for key in [key for key in compiled if key not in used]:
            del compiled[key]

    # a declared model that no component places (directly or as a unit's member) is still checked, exactly as a component
    # would compile it, so a Project never carries an unusable model that check_project passed
    for declared in _unplaced_models(project):
        checkpoint()
        try:
            compile_declared(declared.model_id, declared, resolver, where=f"optical model {declared.model_id}")
        except LocalRunError as exc:
            problems.extend(exc.problems)

    wavelengths = tuple(sorted({wavelength for model in sources.values() for wavelength in model.wavelengths_nm}))
    if require_source and not sources and not any(p.code in (LocalRunErrorCode.SOURCE_MODEL_MISSING, LocalRunErrorCode.SOURCE_PRODUCT_UNSUPPORTED)
                                                       for p in problems):
        problems.append(LocalRunProblem(LocalRunErrorCode.SOURCE_MISSING, "the project has no light source; add an optical_role 'source' component with a source_model"))

    components: dict[str, EngineeringComponentSnapshot] = {}
    for instance_id, model in sources.items():
        component = next(c for c in project.components if c.instance_id == instance_id)
        try:
            components[instance_id] = _source_component(component, model)
        except ValidationError as exc:
            problems.append(LocalRunProblem(LocalRunErrorCode.ENGINE_INPUT_INVALID, _validation_message(exc), f"component {instance_id}"))
    for instance_id, component in instruments.items():
        components[instance_id] = EngineeringComponentSnapshot(instance_id=instance_id, role=component.optical_role.value,
                                                               pose=_pose(component.nominal_pose))
    by_id = {c.instance_id: c for c in project.components}
    for element in elements:
        try:
            components[element.instance_id] = element_component(by_id[element.instance_id], element, ratings.get(element.instance_id, ()))
        except ValidationError as exc:
            problems.append(LocalRunProblem(LocalRunErrorCode.ENGINE_INPUT_INVALID, _validation_message(exc), f"component {element.instance_id}"))
    for component in project.components:
        # a retained DOF that states its travel: the stage's stated limits, for the dof_* targets (the trace does not read them)
        actuators = dof_actuators(component) if component.instance_id in components else ()
        if actuators:
            components[component.instance_id] = components[component.instance_id].model_copy(update={"actuators": actuators})

    observations: list[Observation] = []
    # (a declared crystal is an Engine material per element that cuts it, and the Engine traces light in a crystal only between
    # that crystal's faces: an observation plane inside one is refused here, as the Engine refuses one in an Engine-request crystal)
    crystals = {m.material_id for m in project.model_materials or () if m.is_crystal}
    gradients = {m.material_id for m in project.model_materials or () if m.is_gradient}
    for observation in project.observations:
        if observation.medium in crystals:
            problems.append(LocalRunProblem(
                LocalRunErrorCode.BIREFRINGENT_MEDIUM_UNSUPPORTED,
                f"observation {observation.observation_id} lies inside the uniaxial crystal {observation.medium}: light is observed "
                "outside crystals; place the observation plane outside the crystal and give it the medium there (for example air)",
                f"observation {observation.observation_id}", details={"material_id": observation.medium}))
            continue
        if observation.medium in gradients:
            problems.append(LocalRunProblem(
                LocalRunErrorCode.GRADIENT_INDEX_UNSUPPORTED,
                f"observation {observation.observation_id} lies inside the gradient-index medium {observation.medium}: light is traced "
                "between the medium's end faces in one step and observed outside it; place the observation plane outside the rod and "
                "give it the medium there (for example air)",
                f"observation {observation.observation_id}", details={"material_id": observation.medium}))
            continue
        try:
            observations.append(Observation(
                observation_id=observation.observation_id,
                device_id=observation.device_instance_id,
                pose=_pose(observation.pose),
                aperture=_aperture(observation.aperture),
                acceptance_half_angle_deg=observation.acceptance_half_angle_deg,
                absorbing=observation.absorbing,
                medium=observation.medium,
            ))
        except ValidationError as exc:
            problems.append(LocalRunProblem(LocalRunErrorCode.ENGINE_INPUT_INVALID, _validation_message(exc), f"observation {observation.observation_id}"))

    targets: list[Target] = []
    target_problems: list[LocalRunProblem] = []
    source_ids = {c.instance_id for c in project.components if c.optical_role is OpticalRoleV1.SOURCE}
    for target in project.targets:
        where = f"target {target.target_id}"
        found = _target_problems(target, project, source_ids)
        if found:
            # (a requirement that cannot be read as written is carried, not refused: it selects nothing and is UNKNOWN)
            target_problems.extend(found)
            targets.append(_not_assessable_target(target, found))
            continue
        try:
            targets.append(Target(
                target_id=target.target_id,
                metric=target.metric,
                device_id=target.device_instance_id,
                observation_id=target.observation_id,
                source_id=target.source_id,
                source_id_b=target.source_id_b,
                wavelength_nm=target.wavelength_nm,
                wavelength_nm_b=target.wavelength_nm_b,
                target_value=target.target_value,
                tolerance_abs=target.tolerance_abs,
                tolerance_rel=target.tolerance_rel,
                minimum=target.minimum,
                maximum=target.maximum,
                unit=target.unit,
                parameters={p.key: p.value for p in target.parameters},
                source_revision=target.source_revision,
            ))
        except ValidationError as exc:
            found = [LocalRunProblem(LocalRunErrorCode.TARGET_INVALID, _validation_message(exc), where)]
            target_problems.extend(found)
            targets.append(_not_assessable_target(target, found))

    solids = None
    if elements and not problems and inspect_solids:
        solids = solid_check(elements, project, refuse=check_solids)

    if problems:
        raise LocalRunError(problems)

    # Author order is preserved; it is part of the Engine input and therefore of its hash.
    ordered = tuple(components[c.instance_id] for c in project.components)
    try:
        request = ValidationRequest(
            project=ProjectSnapshot(components=ordered, observations=tuple(observations), targets=tuple(targets),
                                    materials=project_materials(elements, air)),
            configuration=configuration,
        )
        # (a FLUORESCENCE / RAMAN sample's emitter from the excitation its plane receives, and each plane's detection)
        request = complete_request(project, request, engine_state=engine_state)
    except ValidationError as exc:
        raise LocalRunError([LocalRunProblem(LocalRunErrorCode.ENGINE_INPUT_INVALID, _validation_message(exc))]) from None
    simulated_parts = tuple(
        SimulatedPartV1(
            instance_id=element.instance_id,
            optical_sku=element.optical_sku,
            catalog_version=catalog.catalog_version,
            simulation_model=SimulationModelRefV1(model_kind=element.model.model_kind, model_version=element.model.model_version),
        )
        for element in elements if element.optical_sku is not None
    )
    return LoweredProject(
        request=request,
        engine_input_hash=hash_exact(request.model_dump(mode="json")),
        catalog_version=catalog.catalog_version if catalog is not None else None,
        simulated_parts=simulated_parts,
        source_wavelengths_nm=wavelengths,
        elements=tuple(elements),
        solids=solids,
        air=air,
        design_point=design_point,
        stated_settings=stated_settings,
        inputs_not_simulated=inputs_not_simulated(project),
        target_problems=tuple(target_problems),
        model_notices=tuple(n for declared in project.optical_models or () for n in declared_model_notices(declared)),
    )
