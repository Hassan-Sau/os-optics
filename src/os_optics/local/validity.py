"""Does light stay inside what each element's model is characterized for? (Checked after the run.)

The lowering compiles every catalog product into the Engine's general physics (``compile.py``) and the Engine traces
it without any product-specific rule. Published models are characterized only for a wavelength band, for the bands
of their coatings and for a range of angles of incidence; outside them the Engine still computes something (a
material's dispersion formula, a coating's value held between its bands), but that number is not published data.
So after the single Engine run, this module reads the Engine's own recorded paths and ray paths -- nothing is
re-traced -- and reports every use outside a published characterization:

For every catalog element and every recorded path segment that ends on one of its surfaces with an interaction
(``refract``, ``tir``, ``reflect``, ``split_transmit``, ``retarder``, ``polarizer``, ``faraday_rotator``, ``thin_lens``,
``neutral_density``), with
the path's vacuum wavelength ``lambda`` (compared exactly, as the decimal of its shortest repr, inclusive) and the chief
ray's angle of incidence ``theta`` in the medium it arrives from (index ``n``):

* ``MODEL_WAVELENGTH_OUTSIDE_BAND``: ``lambda`` is outside the model's ``valid_band_nm``;
* ``COATING_WAVELENGTH_OUTSIDE_BANDS``: the surface is coated and ``lambda`` lies in none of the coating's bands (a
  filter's transition gap, or beyond its outermost band) or outside its tabulated spectrum or angle-resolved table;
* ``COATING_AOI_OUTSIDE_RANGE``: the surface is coated and the chief ray's angle, read as the coating's
  ``aoi_reference`` says, is outside the coating's ``valid_aoi_deg`` by more than ``AOI_TOLERANCE_DEG`` (a float
  allowance: an exact 45 deg fold is inside a 45-45 deg range). ``AIR_EQUIVALENT``: the equivalent angle in air,
  ``asin(min(1, n sin theta))`` (``n sin theta`` is invariant through a thin film, so a coating on the back face of a plate
  is judged by the same number as on its front); ``INCIDENT_MEDIUM``: ``theta`` itself (a cube splitter's diagonal coating
  specified at 45 deg inside the glass). An angle-resolved table (``angular_spectrum``, always ``INCIDENT_MEDIUM``) is
  also never used beyond its own angles (``valid_aoi_deg`` lies inside them; a single design angle's tolerance does not
  reach past the table);
* ``ELEMENT_AOI_OUTSIDE_RANGE``: the same angle check on a polarization element or a Faraday rotator (its model's
  ``valid_aoi_deg``).

Angle checks are evaluated on the chief ray. The rays of an extended emitter are each their own chief ray, and every one of them is
checked, whether or not it is among the rays kept as whole paths (``optics/rays/sampled.py``, ``SampledScope``): per surface against the range of
angles they met it at, whose two ends the issue names; an issue of theirs counts its rays, and reaches every plane and element the rays of that line
reach. The rays of the light a surface scattered out of a beam (``optics/rays/scatter_trace.py``) are checked the same way, origin
by origin (``ScatterLine.scope``): the wavelength, coating and angle checks on every surface they met, and the data ranges where
they stopped; an issue of theirs reaches the planes that origin's rays cross (its own ``sampled_light`` line, ``scatter_surface_id``,
``parent_path_id``) and the elements they meet. The edge and body checks (``RAY_LEFT_ELEMENT_THROUGH_EDGE``,
``RAY_CROSSES_ELEMENT_BODY``) are not applied to scattered rays: a diffuse law sends light into every direction, through the edges
of the scattering element itself, so these would state the scattering model rather than a use of an element's model.

Single design points. A product specification often states one design wavelength ("633 nm": a laser-line coating, a waveplate for one
laser) or one design angle ("45 deg": a splitter or fold-mirror coating), and the published model then carries that
point as a zero-width range (``min == max``). Such a point names a laser line or a nominal fold, not a characterization
that fails 0.2 nm or 0.01 deg away, so for the forms below it is matched within a tolerance. The tolerances are caller
settings: the Project states them in ``numerical_settings.design_point``, the Engine defaults (given below) hold for what
it does not state, and lowering resolves them once (``design_point.DesignPointTolerances``, kept on every compiled
element and on ``LoweredProject.design_point``; the run records them in ``manifest.numerical_tolerances``). The same
values are written into each coating's declared validity in the Engine request (``compile.engine_coating``), so they are
part of ``engine_input_hash``:

* wavelength (``valid_band_nm`` of a model, a single-wavelength band of a coating): within
  ``design_point_wavelength_rel`` (default 0.5 %) of the stated wavelength -- a laser-line coating holds its reflectance or
  transmittance over several percent around its line, and a zero-order retarder's retardance (which the Engine scales as
  ``lambda0 / lambda``) moves by 0.5 % of its value, 0.0025 waves for a half-wave plate. A multi-order retarder (no
  ``RETARDANCE_INVERSE_WAVELENGTH`` approximation: its order is not published) is matched within
  ``design_point_retarder_wavelength_rel`` (default 0.05 %): its retardance moves ``(m + delta0)`` times faster, and
  0.05 % keeps a plate of order up to 10 within 0.005 waves while admitting the standard laser lines (HeNe 632.8 nm for
  "633 nm", Rb 780.24 nm for "780 nm");
* angle (a coating's single-angle ``valid_aoi_deg``): within the coating kind's ``design_point_angle_<kind>_deg`` of it,
  whatever product carries the coating (the anti-reflection back face of a dichroic mirror too). Defaults: metallic
  reflectors 5 deg (their reflectance changes by well under 1 % there), dielectric high reflectors and anti-reflection
  stacks 3 deg (a thin-film band shifts by ``lambda (sin^2 theta - sin^2 theta0) / (2 n_eff^2)``, under 2 % of the
  wavelength for ``n_eff`` >= 1.6, inside their broad bands), beamsplitter coatings 2 deg (their R / T split and s / p
  split change by a few percent per degree). Interference filter coatings (dichroic, bandpass, edge-pass, notch) shift
  their whole spectrum with angle, ``lambda(theta) = lambda0 sqrt(1 - sin^2(theta) / n_eff^2)``, so their tolerance is
  the angle on either side of the design angle within which that shift stays within
  ``design_point_interference_shift_fraction`` (default a quarter) of the narrowest spectral width of its published
  bands -- each band, and each gap between two bands (a transition edge the bands leave out), relative to its centre
  wavelength -- with the conservative (low) effective index ``INTERFERENCE_N_EFF`` = 1.7, rounded down to 0.1 deg and at
  most ``design_point_interference_angle_max_deg`` (default 3 deg; ``interference_angle_tolerance_deg``; the catalog
  derives the angle ranges of normal-incidence interference filters the same way). With the defaults a 90 nm transition
  gap at 1000 nm between broad bands keeps 3 deg; a 3 nm gap at 268 nm keeps 0.7 deg. A coating whose bands leave no gap
  keeps its angle exactly. The angle is read as the coating's ``aoi_reference`` says.

A finding at such a point carries its facts in ``ModelScopeIssue.details`` (``design_point_facts``): the published
point, the values the light met it at, how the angle is read, the tolerance applied, the setting that holds it and
whether the Project stated it.

Filters keep their stated wavelength points exactly: a product of the ``FILTER`` category and a dichroic, bandpass,
edge-pass, notch or neutral-density coating, whose transmission edges move with wavelength on the scale of their
bandwidth. A neutral-density coating keeps its stated angle exactly. Polarization elements state ranges, not points, for
their angle.

Where the published data ran out on an element's surface:

* ``MATERIAL_DATA_OUTSIDE_RANGE``: a path the Engine stopped there, termination ``MATERIAL_OUT_OF_RANGE`` (the wavelength
  is outside a medium's published dispersion band; the Engine never extrapolates it);
* ``COATING_DATA_OUTSIDE_RANGE``: a beam the Engine carried on through a coating outside its data, with the data's values
  at the nearest stated point, or past stated phases that do not cover its wavelength (the Engine's
  ``COATING_OUT_OF_RANGE`` issue names the surface and the path: everything that path reaches behind the surface is
  affected); a sampled ray of an extended emitter stops there (termination ``COATING_OUT_OF_RANGE``).

And where the beam meets an element in a way the Engine's single-packet trace does not follow (the Engine follows each
beam along its chief ray: one power, one Gaussian beam, the surfaces the chief ray reaches):

* ``BEAM_PARTIALLY_INTERCEPTED``: pupil rays of a path (``validate_with_ray_paths``) that would have travelled on with
  the chief ray -- they reach the chief's next surface inside its clear aperture, so the path's power and Gaussian beam
  count them -- meet a surface of this element first (``NOT_FOLLOWED`` with ``counted_with_chief``). Part of the beam
  falls on an element the chief ray misses (or on another of its surfaces), so the delivered power and beam are those
  of an unobstructed beam. Every such ray counts (each carries ``1 / pupil_ray_count`` of the sampled power): a beam
  clipped by the chief's own surface is the Engine's aperture truncation instead, and pupil rays that miss the chief's
  surface are already counted as lost by it;
* ``RAY_LEFT_ELEMENT_THROUGH_EDGE``: a chief-ray segment inside a catalog element's medium (it entered through one of the
  element's surfaces) does not end on one of that element's surfaces: the light leaves the element through its edge
  (side wall) or a body face that is not a traced surface, which the published model does not describe; the Engine keeps
  it in the glass medium. The medium a segment runs in names the body it belongs to: behind a joint of a compound unit
  (``CompoundModelV1.joints``) the light is in the next member's crystal or glass (an optical contact: that member's
  faces and the shared face bound it) or in the cement layer (bounded by the two joined faces), not in the member whose
  face it crossed. A chief ray in a radial gradient-index rod is recorded as straight pieces along its curved path; a
  piece that ends inside the medium where the next piece of the same medium starts is part of one run from face to face
  (light reaching the rod's side is the Engine's own ``GRADIENT_INDEX_UNSUPPORTED``);
* ``RAY_CROSSES_ELEMENT_BODY``: a chief-ray segment the Engine traces in air passes through an element's solid body for
  more than ``BODY_CROSSING_TOLERANCE_MM`` without meeting one of its surfaces: the beam enters the element through its
  edge or a face that is not a traced surface, and the Engine sees air. The body of a surface sequence is each non-air
  medium between two consecutive surfaces (their full profiles) inside the element's round or rectangular outline; the
  body of a surface set is its convex solid (the segment is clipped exactly against its half-spaces and, for a round
  plate, its side cylinder);
* ``ELEMENT_MET_FROM_BEHIND``: a transmitting face-normal element (a dichroic mirror, a plate splitter, a beam sampler:
  its coated face both transmits and reflects) reflects a chief ray that arrives from behind its coated face
  (travelling along the element's local +Z) at a coating that is not lossless at the path's wavelength (the band
  covering it, a single-wavelength band matched within its tolerance included). Only the coated face is judged: light
  reaching a back face from outside meets that face's own front. Light from behind is otherwise in scope. The power transmittance of a linear coating stack is the same in both directions
  (reciprocity), so the transmitted branch uses the published value. Where the coating is lossless (reflectance +
  transmittance = 1 within ``LOSSLESS_TOLERANCE``, for each polarization it states, as in every band a
  ``LOSSLESS_COATING`` model completes with 1 minus the specified value), the reflectance from behind is 1 minus that
  transmittance, i.e. the published reflectance (Stokes relations), so the reflected branch uses it too. A coating
  that absorbs does not determine its reflectance from behind: only that reflected branch is outside the evidence, and
  the issue names it. A beam combiner meets its element from both sides, so the remedy never asks to turn an element
  a chief ray also meets on its coated face. A reflector's back is not a transmitting face: the Engine blocks it
  (below).

Issues are aggregated per element, surface and code, listing the offending values. A beam whose chief ray misses an
element entirely, hits its opaque rim or the back of a reflector is the Engine's own finding (the beam passes by or
stops, and targets downstream fail).

A violation never changes the Engine result; it makes the local validation status ``OUT_OF_MODEL_SCOPE``
(``LocalValidationRecordV1.validation_status`` may be stricter than the simulation status) and is listed in
``LocalOpticalRun.model_scope_issues``.

What a finding affects. Every finding names the observation planes and elements its light reaches behind the point of
the finding, and the declared targets that read that light -- on a plane, a beam (path and crossing) or a sampled line
the target's selection reads (its source, wavelength, path, crossing and pair; a target that selects no beam reads every
beam on its plane); on an element, a beam of the target's source and wavelength -- and whose metric reads a kind of
quantity the finding touches (``MODEL_SCOPE_AFFECTS``, per code; ``validation/quantity_kinds.py``, per metric). One
wavelength outside a lens's band leaves another wavelength's spot on the same plane decided. The four coefficient codes
(``COATING_WAVELENGTH_OUTSIDE_BANDS``, ``COATING_AOI_OUTSIDE_RANGE``, ``ELEMENT_AOI_OUTSIDE_RANGE``,
``ELEMENT_MET_FROM_BEHIND``) touch power and polarization; every other code touches geometry as well. A target on a
metric fixed by the geometry alone (a position, a direction, a beam radius, a path length) behind a coefficient finding,
and a target on a declared limit behind any finding, keeps the Engine's assessment. A plane's recorded quantity is its
power, which every code touches. The light a surface scatters out of a beam behind the point of the finding is light
the finding reaches.

Models the Project declares (``optical_models``) are checked by exactly these rules against their own bands, coatings
and bodies (a model's band and coatings are its characterization). A ``COMPOUND`` unit is checked member by member: each
member is its own optical body at its pose in the unit, and an issue of a member names it (``member_id``; its surfaces
are ``<member_id>-<surface_id>``). An ideal thin lens is checked against its band; an aperture stop is wavelength
independent. A tabulated thin lens's validity domain (its table's wavelengths, field angle, beam size, polarization and
the metrics it was validated for) travels with its Engine interaction (``TabulatedThinLensInteraction``): the Engine
checks it itself and reports every use outside it as an ``OUT_OF_MODEL_SCOPE`` issue of the run
(``MODEL_WAVELENGTH_OUTSIDE_BAND``, ``MODEL_FIELD_ANGLE_EXCEEDED``, ``MODEL_BEAM_SIZE_EXCEEDED``,
``MODEL_POLARIZATION_NOT_VALIDATED``: one per path, surface and code, naming the path in its ``path_id``) or target
(``METRIC_NOT_VALIDATED_FOR_ELEMENT``). Each such run issue is also listed here (``TABULATED_DOMAIN_CODES``, in the
Engine's words) with what it affects: from the path's first crossing of the lens surface (for the field angle, the first
crossing beyond the validated angle), the observations, targets and elements that light and the beams branched from it
behind that point reach, as for the issues above. The run does not record a beam's size or polarization per crossing, so
for ``MODEL_BEAM_SIZE_EXCEEDED`` and ``MODEL_POLARIZATION_NOT_VALIDATED`` on a path that crosses the lens more than once
this starts at the first crossing, which may be earlier than the one outside the domain. An issue whose path the run does
not list applies from the first crossing of the lens on every path that crosses it.

A Faraday rotator's wavelength domain (its table's) travels the same way: the Engine stops light outside it at the element
and reports ``MODEL_WAVELENGTH_OUTSIDE_BAND`` (``OUT_OF_MODEL_SCOPE``), listed here with what that light would have
reached. Its ``valid_aoi_deg`` is checked here like a polarization element's.

A diffraction grating has no model-scope issue here: its directions hold at every wavelength and angle, so nothing behind it is
out of scope; where its response does not hold the Engine says so (``GRATING_RESPONSE_UNAVAILABLE``) and the ledger books the
power (``UNMODELLED_RESPONSE``), which leaves the power on the planes unknown and the geometry decided.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from ..run_control import checkpoint
from .._vendor.shared_contracts.common.sag_shapes import SagForm, sag_form, sag_value
from .._vendor.shared_contracts.design_catalog import (
    AIR,
    AngleRangeV1,
    CoatingAoiReference,
    CoatingBandV1,
    CoatingKind,
    OpticalCategory,
    PolarizationElementModelV1,
    PrescriptionSurfaceV1,
    PublicCoatingV1,
    SimulationApproximation,
    SurfaceSequenceModelV1,
    SurfaceSetModelV1,
    TracedBranches,
    WavelengthBandV1,
)
from ..config import NumericalConfig
from ..contracts.basis import ValidityCheck
from ..contracts.project import Target
from ..contracts.results import ValidationResult
from ..errors.status import AssumedInput
from ..math.pose import Pose
from ..optical_view.surface_mesh import held_point
from ..optics.rays.sampled import SampledScope, SampledTrace
from ..optics.rays.scatter_trace import ScatterTrace
from ..optics.surfaces.tabulated_lens import DOMAIN_LIMIT_ABS_TOLERANCE_DEG, beyond_limit
from ..validation.exposures import ELEMENT_METRICS
from ..validation.metrics import selected_light
from ..validation.quantity_kinds import ALL_QUANTITY_KINDS, QuantityKind, metric_quantity_kinds
from ..validation.ray_paths import NOT_FOLLOWED, RayPathTrace
from .bodies import clip_segment
from .compile import ENGINE_SURROUNDINGS, VACUUM, surrounding_index
from .compile import (DIFFRACTION_GRATING, ENGINE_AIR, FARADAY_ROTATOR, MEMBER_SURFACE_SEPARATOR, OUTLINE_TOLERANCE_MM, CompiledElement,
                      POLARIZATION_ELEMENT, TABULATED_THIN_LENS, SurfaceEnvelope)
from .design_point import (ANGLE_SETTINGS, DEFAULT_DESIGN_POINT, INTERFERENCE_ANGLE_MAX_SETTING, INTERFERENCE_SHIFT_SETTING,
                           RETARDER_WAVELENGTH_SETTING, WAVELENGTH_SETTING, DesignPointTolerances)
from .lowering import LoweredProject

# Float allowance on a published angle-of-incidence range: the Engine's recorded AOI of an exact 45 deg fold is 45 deg up
# to rounding of the unit quaternion and the arccos, far below any optical significance.
AOI_TOLERANCE_DEG = 1e-6
# Single design points of non-filter forms (module docstring): the tolerances are caller settings
# (``numerical_settings.design_point``), resolved once per lowering into the ``DesignPointTolerances`` each compiled element
# carries (``CompiledElement.design_point``); nothing here holds a value of its own.
PERCENT = 100
FILTER_COATING_KINDS = frozenset({CoatingKind.DICHROIC, CoatingKind.BANDPASS, CoatingKind.EDGEPASS, CoatingKind.NOTCH,
                                  CoatingKind.NEUTRAL_DENSITY})
# Interference filter coatings: a single design angle is matched within the angle that shifts their spectrum by at most
# the shift fraction in force of the narrowest spectral width their bands leave (module docstring), with the conservative
# effective index of hard-coated stacks, rounded down to INTERFERENCE_ANGLE_STEP_DEG and at most the cap in force.
INTERFERENCE_COATING_KINDS = frozenset({CoatingKind.DICHROIC, CoatingKind.BANDPASS, CoatingKind.EDGEPASS, CoatingKind.NOTCH})
INTERFERENCE_N_EFF = 1.7
INTERFERENCE_ANGLE_STEP_DEG = 0.1
GRAZING_AOI_DEG = 90.0
# Float allowance when rounding a derived tolerance down to INTERFERENCE_ANGLE_STEP_DEG (2.9999999 deg stays 3.0 deg).
ROUNDING_ALLOWANCE = 1e-9
# Float allowance on the length a chief-ray segment runs inside an element's body: a segment that starts or ends on the
# element's surface touches the body only at a point (up to the rounding of the hit point, ~1e-12 mm).
BODY_CROSSING_TOLERANCE_MM = 1e-6
# Cumulative path distances use the same local-coordinate roundoff allowance as compiled
# surface boundaries, so coincident samples do not spuriously inherit a downstream scope issue.
SCOPE_DISTANCE_TOLERANCE_MM = OUTLINE_TOLERANCE_MM
# Engine segment end events at which light interacts with a surface (and so with its published data).
INTERACTION_EVENTS = frozenset({"refract", "tir", "reflect", "split_transmit", "retarder", "polarizer", "faraday_rotator", "thin_lens",
                                "neutral_density", "grating", "complex_transmission"})
MATERIAL_OUT_OF_RANGE = "MATERIAL_OUT_OF_RANGE"
COATING_OUT_OF_RANGE = "COATING_OUT_OF_RANGE"

MODEL_WAVELENGTH_OUTSIDE_BAND = "MODEL_WAVELENGTH_OUTSIDE_BAND"
COATING_WAVELENGTH_OUTSIDE_BANDS = "COATING_WAVELENGTH_OUTSIDE_BANDS"
COATING_AOI_OUTSIDE_RANGE = "COATING_AOI_OUTSIDE_RANGE"
ELEMENT_AOI_OUTSIDE_RANGE = "ELEMENT_AOI_OUTSIDE_RANGE"
MATERIAL_DATA_OUTSIDE_RANGE = "MATERIAL_DATA_OUTSIDE_RANGE"
COATING_DATA_OUTSIDE_RANGE = "COATING_DATA_OUTSIDE_RANGE"
BEAM_PARTIALLY_INTERCEPTED = "BEAM_PARTIALLY_INTERCEPTED"
RAY_LEFT_ELEMENT_THROUGH_EDGE = "RAY_LEFT_ELEMENT_THROUGH_EDGE"
RAY_CROSSES_ELEMENT_BODY = "RAY_CROSSES_ELEMENT_BODY"
ELEMENT_MET_FROM_BEHIND = "ELEMENT_MET_FROM_BEHIND"
MODEL_FIELD_ANGLE_EXCEEDED = "MODEL_FIELD_ANGLE_EXCEEDED"
MODEL_BEAM_SIZE_EXCEEDED = "MODEL_BEAM_SIZE_EXCEEDED"
MODEL_POLARIZATION_NOT_VALIDATED = "MODEL_POLARIZATION_NOT_VALIDATED"
LOCAL_Z = (0.0, 0.0, 1.0)
# Engine branch kinds of a reflected child path: a coated transmitting interface's reflection, an ideal splitter's.
REFLECTION_BRANCHES = frozenset({"coating_reflection", "beam_splitter_reflection"})
# Allowance on reflectance + transmittance = 1 of a lossless coating: a tabulated spectrum's values are linearly
# interpolated and each rounded down to 1e-18, so a lossless curve sums to 1 only up to that rounding.
LOSSLESS_TOLERANCE = Decimal("1e-9")
# Engine surface ids of observation planes: a plane samples the beam where it crosses it; the light travels on unchanged.
OBSERVATION_SURFACE_PREFIX = "observation/"
# Crossings of a body segment with a conic / polynomial / cylindrical surface are bracketed on this many equal steps of the
# segment and bisected to BODY_CROSSING_TOLERANCE_MM (planes and spheres are cut exactly).
SAG_CROSSING_STEPS = 64
SAG_BISECTION_STEPS = 60
# The filter ahead of the clip of a sampled ray: the z range a body's surfaces can reach beyond their vertices, in outline reaches.
SAG_FILTER_REACHES = 2.0
#: The end events of those elements' crossings.
ENGINE_CHECKED_EVENTS = frozenset({"thin_lens", "faraday_rotator"})
#: Every model-scope issue code, in reporting order.
MODEL_SCOPE_CODES = (
    MODEL_WAVELENGTH_OUTSIDE_BAND,
    COATING_WAVELENGTH_OUTSIDE_BANDS,
    COATING_AOI_OUTSIDE_RANGE,
    ELEMENT_AOI_OUTSIDE_RANGE,
    MATERIAL_DATA_OUTSIDE_RANGE,
    COATING_DATA_OUTSIDE_RANGE,
    BEAM_PARTIALLY_INTERCEPTED,
    RAY_LEFT_ELEMENT_THROUGH_EDGE,
    RAY_CROSSES_ELEMENT_BODY,
    ELEMENT_MET_FROM_BEHIND,
    MODEL_FIELD_ANGLE_EXCEEDED,
    MODEL_BEAM_SIZE_EXCEEDED,
    MODEL_POLARIZATION_NOT_VALIDATED,
)
#: The elements whose wavelength domain the Engine checks itself (module docstring).
ENGINE_CHECKED_DOMAINS = frozenset({TABULATED_THIN_LENS, FARADAY_ROTATOR})
#: The Engine's own validity-domain issue codes of those elements (module docstring).
TABULATED_DOMAIN_CODES = (
    MODEL_WAVELENGTH_OUTSIDE_BAND,
    MODEL_FIELD_ANGLE_EXCEEDED,
    MODEL_BEAM_SIZE_EXCEEDED,
    MODEL_POLARIZATION_NOT_VALIDATED,
)
#: Finding code -> the kinds of quantity it touches (``validation/quantity_kinds.py``). A finding about a coefficient -- a
#: coating or a polarization element used outside the wavelengths or angles its data are published for, a coating whose
#: data ran out and whose values at the nearest stated point the beam is carried on with, a reflection from behind an
#: absorbing coating -- touches the power and the polarization of the light behind it: the Engine traces the same
#: surfaces, points and directions whatever the coefficient is (no coating coefficient enters a ray's direction or a
#: Gaussian beam's state). Every other code touches every kind (the model band, material data that ran out and stopped the
#: path, a beam cut beside its chief ray, light through an edge or a body, a tabulated lens outside its domain), and so
#: does a code this table does not hold (``finding_quantity_kinds``).
MODEL_SCOPE_AFFECTS: dict[str, frozenset[QuantityKind]] = {
    **{code: ALL_QUANTITY_KINDS for code in MODEL_SCOPE_CODES},
    **{code: frozenset({QuantityKind.POWER, QuantityKind.POLARIZATION})
       for code in (COATING_WAVELENGTH_OUTSIDE_BANDS, COATING_AOI_OUTSIDE_RANGE, COATING_DATA_OUTSIDE_RANGE, ELEMENT_AOI_OUTSIDE_RANGE,
                    ELEMENT_MET_FROM_BEHIND)},
}


def finding_quantity_kinds(code: str) -> frozenset[QuantityKind]:
    """The kinds of quantity a finding of ``code`` touches; every kind for a code the table does not hold."""

    return MODEL_SCOPE_AFFECTS.get(code, ALL_QUANTITY_KINDS)


def finding_applies(code: str, metric: str) -> bool:
    """Whether a finding of ``code`` on the light a target reads affects a target on ``metric``: the metric reads a kind
    of quantity the finding touches."""

    return bool(finding_quantity_kinds(code) & metric_quantity_kinds(metric))


@dataclass(frozen=True)
class ModelScopeIssue:
    """One model validity violation: element ``instance_id`` (and ``surface_id`` when it concerns one surface; ``member_id``
    when it concerns one member of a ``COMPOUND`` unit)."""

    instance_id: str
    code: str
    message: str
    surface_id: str | None = None
    member_id: str | None = None
    affected_observation_ids: tuple[str, ...] = ()
    affected_target_ids: tuple[str, ...] = ()
    affected_element_ids: tuple[str, ...] = ()
    #: The structured facts of a finding at a single design point (``design_point_facts``): ``{"design_point": {...}}``;
    #: ``None`` for every other finding.
    details: dict | None = None
    #: The beams the finding's light reaches on the observation planes, ``(observation_id, path_id, crossing_index)`` in
    #: order: each arrival behind the point of the finding. An analysis of the light on a plane reads it
    #: (``results.imaging``).
    affected_arrivals: tuple[tuple[str, str, int], ...] = ()
    #: The sampled lines the finding's light reaches on the observation planes, ``(observation_id, source_id,
    #: spectral_index)`` in order: an extended emitter's line whose rays met the element, on the planes its rays cross; the
    #: line of the light one surface scattered (``scatter_surface_id``, ``parent_path_id``) whose rays met the element, on
    #: the planes that origin's rays cross; and the line of the light a surface scattered out of a beam the finding reaches,
    #: behind the point of the finding (on the finding's own path where that path meets the scattering surface at or after
    #: that point; on a path branched after it, wherever the light scatters). A sampled-light row of such a line reads it
    #: (``results.basis.scoped_observation``).
    affected_sampled_lines: tuple[tuple[str, str, int], ...] = ()
    #: The validity conditions the finding states (``finding_validity``): per value the light met the element at, the
    #: parameter, the published limits, that value, ``within`` false, the setting that holds a tolerance, where and the code
    #: -- the entry a value the finding qualifies carries in its ``value_basis``.
    validity: tuple[ValidityCheck, ...] = ()
    #: Per check of ``validity`` (in its order), the light that met the element at that check's value: the targets that
    #: read it (``affected_target_ids``, by the rule the finding names its targets by), its arrivals
    #: ``(observation_id, path_id, crossing_index)`` and its sampled lines ``(observation_id, source_id, spectral_index)``.
    #: A target's value carries the checks of the light it reads, an imaging field those of its own arrivals and a
    #: sampled-light row those of its own line.
    validity_reach: tuple[tuple[tuple[str, ...], tuple[tuple[str, str, int], ...], tuple[tuple[str, str, int], ...]], ...] = ()

    def as_dict(self) -> dict:
        data = {"instance_id": self.instance_id, "code": self.code, "message": self.message,
                "affected_observation_ids": list(self.affected_observation_ids),
                "affected_target_ids": list(self.affected_target_ids),
                "affected_element_ids": list(self.affected_element_ids)}
        if self.surface_id is not None:
            data["surface_id"] = self.surface_id
        if self.member_id is not None:
            data["member_id"] = self.member_id
        if self.details is not None:
            data["details"] = self.details
        if self.affected_arrivals:
            data["affected_arrivals"] = [{"observation_id": oid, "path_id": path_id, "crossing_index": crossing}
                                         for oid, path_id, crossing in self.affected_arrivals]
        if self.affected_sampled_lines:
            data["affected_sampled_lines"] = _line_rows(self.affected_sampled_lines)
        if self.validity:
            data["validity"] = [check.model_dump(mode="json") for check in self.validity]
        if self.validity_reach:
            data["validity_reach"] = [
                {"affected_target_ids": list(target_ids),
                 "affected_arrivals": [{"observation_id": oid, "path_id": path_id, "crossing_index": crossing}
                                       for oid, path_id, crossing in arrivals],
                 **({"affected_sampled_lines": _line_rows(lines)} if lines else {})}
                for target_ids, arrivals, lines in self.validity_reach]
        return data


def _line_rows(lines: tuple[tuple[str, str, int], ...]) -> list[dict]:
    return [{"observation_id": oid, "source_id": source_id, "spectral_index": index} for oid, source_id, index in lines]


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def equivalent_air_aoi_deg(aoi_deg: float, refractive_index: float) -> float:
    """``asin(min(1, n sin theta))`` in degrees: the angle in air with the same ``n sin theta`` (Snell invariant)."""

    return math.degrees(math.asin(min(1.0, refractive_index * math.sin(math.radians(aoi_deg)))))


def _inside(angles: AngleRangeV1, aoi_deg: float, tolerance_deg: float = 0.0) -> bool:
    slack = AOI_TOLERANCE_DEG + tolerance_deg
    return float(angles.min_deg) - slack <= aoi_deg <= float(angles.max_deg) + slack


def _filter_form(element: CompiledElement, coating: PublicCoatingV1 | None = None) -> bool:
    return element.identity.category is OpticalCategory.FILTER or (coating is not None and coating.kind in FILTER_COATING_KINDS)


def _multi_order_retarder(element: CompiledElement) -> bool:
    """A retarder whose order is not published (no ``RETARDANCE_INVERSE_WAVELENGTH`` approximation)."""

    model = element.model
    return (isinstance(model, PolarizationElementModelV1) and model.retardance_waves is not None
            and SimulationApproximation.RETARDANCE_INVERSE_WAVELENGTH not in model.approximations)


def wavelength_tolerance_setting(element: CompiledElement, coating: PublicCoatingV1 | None = None) -> str | None:
    """The setting (``NumericalConfig`` field) that holds the tolerance a single-wavelength band of ``element`` (or of its
    ``coating``) is matched within; ``None`` for filters, which keep their wavelength points exactly."""

    if _filter_form(element, coating):
        return None
    return RETARDER_WAVELENGTH_SETTING if coating is None and _multi_order_retarder(element) else WAVELENGTH_SETTING


def wavelength_tolerance(element: CompiledElement, coating: PublicCoatingV1 | None = None) -> Decimal:
    """The relative tolerance a single-wavelength band of ``element`` (or of its ``coating``) is matched within (module
    docstring), from the tolerances the element was compiled under (``CompiledElement.design_point``); 0 for filters."""

    setting = wavelength_tolerance_setting(element, coating)
    if setting is None:
        return Decimal(0)
    point = element.design_point
    return point.retarder_wavelength_rel if setting == RETARDER_WAVELENGTH_SETTING else point.wavelength_rel


def _in_band(band: WavelengthBandV1, wavelength: Decimal, tolerance: Decimal) -> bool:
    if band.contains(wavelength):
        return True
    point = Decimal(band.min_nm)
    return tolerance > 0 and point == Decimal(band.max_nm) and abs(wavelength - point) <= tolerance * point


def _covering_band(element: CompiledElement, coating: PublicCoatingV1, wavelength: Decimal, aoi_deg: float | None = None
                   ) -> CoatingBandV1 | None:
    """The coating band that covers ``wavelength``: the band containing it, or a single-wavelength band matched within
    the element's tolerance (``wavelength_tolerance``); for an angle-resolved table, its values at ``wavelength`` and the
    angle of incidence ``aoi_deg`` (in the incident medium; None without an angle or outside the table)."""

    if coating.angular_spectrum is not None:
        return coating.band_at(wavelength, _decimal(aoi_deg)) if aoi_deg is not None else None
    band = coating.band_at(wavelength)
    if band is not None:
        return band
    tolerance = wavelength_tolerance(element, coating)
    return next((b for b in coating.bands or () if Decimal(b.min_nm) == Decimal(b.max_nm)
                 and _in_band(WavelengthBandV1(min_nm=b.min_nm, max_nm=b.max_nm, status=b.status), wavelength, tolerance)), None)


def _coating_covers(element: CompiledElement, coating: PublicCoatingV1, wavelength: Decimal) -> bool:
    if coating.angular_spectrum is not None:  # the table's wavelengths (its angles are checked with the angle of incidence)
        return coating.covers(wavelength)
    return _covering_band(element, coating, wavelength) is not None


def _shift_limit_deg(design_deg: float, ratio: float) -> float:
    """How far (deg) the angle can move from ``design_deg`` before an interference spectrum designed there has moved to
    ``ratio`` times its design wavelengths (``lambda(theta) / lambda(theta0)``; < 1 toward larger angles, > 1 toward
    smaller ones), or to 90 / 0 deg when it never gets that far."""

    n2 = INTERFERENCE_N_EFF * INTERFERENCE_N_EFF
    sin2 = n2 * (1.0 - ratio * ratio * (1.0 - math.sin(math.radians(design_deg)) ** 2 / n2))
    if sin2 > 1.0:
        return GRAZING_AOI_DEG - design_deg
    if sin2 < 0.0:
        return design_deg
    return abs(math.degrees(math.asin(math.sqrt(sin2))) - design_deg)


def _interference_shift_limit_deg(coating: PublicCoatingV1, design_deg: float, design_point: DesignPointTolerances) -> float | None:
    """The angle (deg), on either side of ``design_deg``, within which an interference coating's spectrum shifts by at most
    the shift fraction in force of the narrowest of its bands and of the gaps between them (each relative to its centre
    wavelength); ``None`` when its bands leave no gap."""

    def relative(low: float, high: float) -> float:  # a width relative to its centre wavelength
        return (high - low) / ((high + low) / 2.0)

    bands = sorted((float(Decimal(b.min_nm)), float(Decimal(b.max_nm))) for b in coating.bands or ())
    widths = [relative(end, start) for (_, end), (start, _) in zip(bands, bands[1:]) if start > end]  # the gaps
    if not widths:
        return None
    widths += [relative(low, high) for low, high in bands if high > low]  # the bands (a single wavelength is matched exactly)
    shift = design_point.interference_shift_fraction * min(widths)
    return min(_shift_limit_deg(design_deg, 1.0 - shift), _shift_limit_deg(design_deg, 1.0 + shift))


def interference_angle_tolerance_deg(coating: PublicCoatingV1, design_deg: float,
                                     design_point: DesignPointTolerances = DEFAULT_DESIGN_POINT) -> float:
    """The tolerance (deg) of an interference filter coating's single design angle (module docstring): on either side,
    the angle within which its spectrum shifts by at most ``design_point.interference_shift_fraction`` of the narrowest of
    its bands and of the gaps between them (each relative to its centre wavelength), rounded down to
    ``INTERFERENCE_ANGLE_STEP_DEG`` and at most ``design_point.interference_angle_max_deg``; 0 when its bands leave no gap."""

    limit = _interference_shift_limit_deg(coating, design_deg, design_point)
    if limit is None:
        return 0.0
    tolerance = min(design_point.interference_angle_max_deg, limit)
    return round(math.floor(tolerance / INTERFERENCE_ANGLE_STEP_DEG + ROUNDING_ALLOWANCE) * INTERFERENCE_ANGLE_STEP_DEG, 6)


def _single_angle(angles: AngleRangeV1 | None) -> bool:
    return angles is not None and Decimal(angles.min_deg) == Decimal(angles.max_deg)


def coating_angle_tolerance_deg(coating: PublicCoatingV1 | None, angles: AngleRangeV1 | None, design_point: DesignPointTolerances) -> float:
    """The tolerance (deg) a single-angle range ``angles`` of ``coating`` is matched within under ``design_point`` (module
    docstring), per coating kind; 0 for a range, a neutral-density coating, an interference coating whose bands leave no gap
    or a surface without a coating (a polarization element)."""

    if coating is None or not _single_angle(angles):
        return 0.0
    if coating.kind in INTERFERENCE_COATING_KINDS:
        return interference_angle_tolerance_deg(coating, float(Decimal(angles.min_deg)), design_point)  # type: ignore[union-attr]
    return design_point.angle_deg(coating.kind)


def angle_tolerance_setting(coating: PublicCoatingV1 | None, angles: AngleRangeV1 | None, design_point: DesignPointTolerances) -> str | None:
    """The setting (``NumericalConfig`` field) that holds the tolerance of ``coating``'s single design angle: its kind's;
    for an interference coating the shift fraction, or the cap when the cap is the smaller; ``None`` where the point is
    kept exactly."""

    if coating is None or not _single_angle(angles):
        return None
    if coating.kind in INTERFERENCE_COATING_KINDS:
        limit = _interference_shift_limit_deg(coating, float(Decimal(angles.min_deg)), design_point)  # type: ignore[union-attr]
        if limit is None:
            return None
        return INTERFERENCE_ANGLE_MAX_SETTING if design_point.interference_angle_max_deg <= limit else INTERFERENCE_SHIFT_SETTING
    return ANGLE_SETTINGS.get(coating.kind)


def angle_tolerance_deg(element: CompiledElement, envelope: SurfaceEnvelope) -> float:
    """The tolerance (deg) a single-angle range of ``envelope`` is matched within (``coating_angle_tolerance_deg``), from
    the tolerances ``element`` was compiled under (``CompiledElement.design_point``)."""

    return coating_angle_tolerance_deg(envelope.coating, envelope.valid_aoi_deg, element.design_point)


def _nm(values) -> str:
    return ", ".join(f"{v!r} nm" for v in sorted(values))


def _deg(values) -> str:
    return ", ".join(f"{v:.6g} deg" for v in sorted(values))


def _band_text(low: str, high: str, tolerance: Decimal) -> str:
    """A published band in words; a single wavelength matched within a tolerance says so."""

    if Decimal(low) != Decimal(high) or tolerance == 0:
        return f"{low}-{high} nm"
    return f"{low} nm (a single design wavelength, matched within +-{float(tolerance * PERCENT):g} %)"


def _beams(values) -> str:
    """Offending beams ``(path_id, wavelength_nm, ...)``, one per path, in path order."""

    return ", ".join(f"{path_id} ({wavelength!r} nm)" for path_id, wavelength, *_ in sorted(values))


class _Collector:
    """Offending values per (body, surface, code), in first-seen order (floats, or per-path tuples). A body is an element,
    or a member of a compound unit: ``(instance_id, member_id)``."""

    def __init__(self, result: ValidationResult, owners) -> None:
        self.values: dict[tuple[tuple[str, str | None], str | None, str], set] = {}
        self.affected: dict[tuple, set[str]] = {}
        self.result = result
        self.paths = {p.path_id: p for p in result.paths}
        self.owners = owners
        self.devices: dict[tuple, set[str]] = {}
        self.context_devices: set[str] = set()
        self.context: set[str] | None = None
        # Per finding, the light it reaches beam by beam (``_TargetReads`` matches a target's selection against it): the
        # arrivals ``(observation_id, path_id, crossing_index)``, the sampled lines ``(observation_id, source_id,
        # spectral_index)`` and the elements with the beam that reaches them ``(instance_id, source_id, wavelength_nm)``
        # (``None`` source and wavelength: any beam).
        self.arrivals: dict[tuple, set[tuple[str, str, int]]] = {}
        self.lines: dict[tuple, set[tuple[str, str, int]]] = {}
        self.device_beams: dict[tuple, set[tuple[str, str | None, float | None]]] = {}
        self.context_arrivals: set[tuple[str, str, int]] = set()
        self.context_lines: set[tuple[str, str, int]] = set()
        self.context_device_beams: set[tuple[str, str | None, float | None]] = set()
        # Per finding and value, the same light: which beams met the element at that value (``_validity_reach``).
        self.value_light: dict[tuple, dict[object, tuple[set, set, set]]] = {}
        #: The beam the check at hand reads, ``(source_id, wavelength_nm)``; ``None``: not known (any beam).
        self.beam: tuple[str, float] | None = None
        self.location: tuple[str, float] | None = None
        self.resolved_location: tuple[str, float] | None = None
        # (built on the first resolve: each path's branches, each path's arrivals and start length -- so a resolve visits
        # the paths that branch from one path, never every path of the run)
        self._children: dict[str, list[str]] | None = None
        self._arrivals: dict[str, list[tuple[str, float, int]]] = {}
        # (per beam path, the sampled lines of the light a surface scattered out of it: plane, line and scattering surface)
        self._scattered: dict[str, list[tuple[str, str, int, str]]] = {}
        self._starts: dict[str, float] = {}

    @staticmethod
    def start_length(path):
        return path.geometric_path_length_mm - sum(s.length_mm or 0.0 for s in path.segments)

    def after(self, path_id: str, minimum_length: float) -> None:
        self.location = (path_id, minimum_length)

    def _start(self, path_id: str) -> float:
        start = self._starts.get(path_id)
        if start is None:
            start = self._starts[path_id] = self.start_length(self.paths[path_id])
        return start

    def _descendants(self, path_id: str, minimum_length: float) -> list[str]:
        """``path_id`` and every path that branched from it (directly or through other branches) at or after
        ``minimum_length`` along it: a path whose own start, or the start of a path between it and ``path_id``, lies before
        that point does not inherit the scope (the branch tree is walked down from ``path_id``)."""
        if self._children is None:
            self._children = {}
            for pid, path in self.paths.items():
                if path.parent_path_id is not None:
                    self._children.setdefault(path.parent_path_id, []).append(pid)
            for observation in self.result.observations:
                for arrival in observation.arrivals:
                    self._arrivals.setdefault(arrival.beam.path_id, []).append(
                        (observation.observation_id, arrival.beam.geometric_length_mm, arrival.crossing_index))
                for line in observation.sampled_light:
                    if line.scatter_surface_id is not None and line.parent_path_id is not None:
                        self._scattered.setdefault(line.parent_path_id, []).append(
                            (observation.observation_id, line.source_id, line.spectral_index, line.scatter_surface_id))
        if path_id not in self.paths:
            return []
        threshold = minimum_length - SCOPE_DISTANCE_TOLERANCE_MM
        found, stack, seen = [], [path_id], {path_id}
        while stack:
            pid = stack.pop()
            found.append(pid)
            for child in self._children.get(pid, ()):
                if child not in seen and not self._start(child) < threshold:
                    seen.add(child)
                    stack.append(child)
        return found

    def _meets_after(self, path_id: str, surface_id: str, minimum_length: float) -> bool:
        """Whether ``path_id`` meets ``surface_id`` at or after ``minimum_length`` along it."""

        distance = self._start(path_id)
        for segment in self.paths[path_id].segments:
            distance += segment.length_mm or 0.0
            if segment.end_surface_id == surface_id and distance >= minimum_length - SCOPE_DISTANCE_TOLERANCE_MM:
                return True
        return False

    def _resolve(self) -> None:
        """Scope starts at a particular point, not the start of an entire path.

        A descendant only inherits it if it branched after that point. Cumulative
        lengths also distinguish repeated crossings and samples before an optic.
        """
        if self.location is None or self.location == self.resolved_location:
            return
        path_id, minimum_length = self.location
        self.resolved_location = self.location
        start = self.paths.get(path_id)
        self.beam = None if start is None else (start.source_id, start.wavelength_nm)
        descendants = self._descendants(path_id, minimum_length)
        self.context_arrivals = {(oid, pid, crossing) for pid in descendants for oid, length, crossing in self._arrivals.get(pid, ())
                                 if length > minimum_length + SCOPE_DISTANCE_TOLERANCE_MM}
        # The light a surface scattered out of these beams at or after that point: on the finding's own path, where the
        # path meets the scattering surface (any of its crossings there), on a path branched after it, anywhere.
        self.context_lines = {(oid, source_id, j) for pid in descendants for oid, source_id, j, surface_id in self._scattered.get(pid, ())
                              if pid != path_id or self._meets_after(pid, surface_id, minimum_length)}
        self.context = {oid for oid, _, _ in self.context_arrivals} | {oid for oid, _, _ in self.context_lines}
        self.context_devices = set()
        self.context_device_beams = set()
        for pid in descendants:
            path = self.paths[pid]
            distance = self._start(pid)
            for segment in path.segments:
                distance += segment.length_mm or 0.0
                owner = self.owners.get(segment.end_surface_id or "")
                if owner is not None and distance >= minimum_length - SCOPE_DISTANCE_TOLERANCE_MM:
                    self.context_devices.add(owner[0].instance_id)
                    self.context_device_beams.add((owner[0].instance_id, path.source_id, path.wavelength_nm))

    def add(self, element: CompiledElement, surface_id: str | None, code: str, value: object) -> None:
        self._resolve()
        key = ((element.instance_id, element.member_id), surface_id, code)
        self.values.setdefault(key, set()).add(value)
        self.affected.setdefault(key, set()).update(self.context or ())
        self.arrivals.setdefault(key, set()).update(self.context_arrivals)
        self.lines.setdefault(key, set()).update(self.context_lines)
        self.devices.setdefault(key, set()).update(self.context_devices | {element.instance_id})
        own = (element.instance_id, *self.beam) if self.beam is not None else (element.instance_id, None, None)
        beams = self.context_device_beams | {own}
        self.device_beams.setdefault(key, set()).update(beams)
        arrivals, lines, device_beams = self.value_light.setdefault(key, {}).setdefault(value, (set(), set(), set()))
        arrivals.update(self.context_arrivals)
        lines.update(self.context_lines)
        device_beams.update(beams)


def _check_interaction(found: _Collector, element: CompiledElement, envelope: SurfaceEnvelope, wavelength_nm: float, aoi_deg: float | None,
                       refractive_index: float) -> None:
    if envelope.traced in (TABULATED_THIN_LENS, DIFFRACTION_GRATING):  # the Engine checks its own domain (module docstring)
        return
    wavelength = _decimal(wavelength_nm)
    band = element.valid_band_nm
    if band is not None and envelope.traced not in ENGINE_CHECKED_DOMAINS and not _in_band(band, wavelength, wavelength_tolerance(element)):
        found.add(element, None, MODEL_WAVELENGTH_OUTSIDE_BAND, wavelength_nm)
    coating = envelope.coating
    if coating is not None and not _coating_covers(element, coating, wavelength):
        found.add(element, envelope.surface_id, COATING_WAVELENGTH_OUTSIDE_BANDS, wavelength_nm)
    if aoi_deg is not None:
        outside = aoi_outside_range(element, envelope, aoi_deg, refractive_index, wavelength_nm)
        if outside is not None:
            found.add(element, envelope.surface_id, *outside)


def aoi_outside_range(element: CompiledElement, envelope: SurfaceEnvelope, aoi_deg: float, refractive_index: float,
                      wavelength_nm: float | None = None) -> tuple[str, float] | None:
    """``(code, angle)`` when the chief ray meets the surface outside the angles it is characterized for (its
    ``valid_aoi_deg`` read as the coating's ``aoi_reference`` says, with the coating kind's tolerance; an angle-resolved
    table never beyond its own angles): ``COATING_AOI_OUTSIDE_RANGE`` / ``ELEMENT_AOI_OUTSIDE_RANGE``; ``None`` inside.
    ``aoi_deg``: the angle in the medium the light arrives from, of absolute index ``refractive_index`` at the vacuum
    wavelength ``wavelength_nm``: the equivalent angle is the one in the air the element operates in (``CompiledElement.air``,
    the Project's environment), ``n sin theta = n_air sin theta_air``. Without ``wavelength_nm`` the index is taken as relative
    to that air already."""

    if envelope.valid_aoi_deg is None:
        return None
    coating = envelope.coating
    if not _incident_medium(envelope) and wavelength_nm is not None:
        refractive_index = refractive_index / surrounding_index(element.air, wavelength_nm)
    angle = aoi_deg if _incident_medium(envelope) else equivalent_air_aoi_deg(aoi_deg, refractive_index)
    if not _inside(envelope.valid_aoi_deg, angle, angle_tolerance_deg(element, envelope)):
        return (ELEMENT_AOI_OUTSIDE_RANGE if envelope.traced in (POLARIZATION_ELEMENT, FARADAY_ROTATOR) else COATING_AOI_OUTSIDE_RANGE), angle
    if coating is not None and coating.angular_spectrum is not None:  # never beyond the table's own angles
        low, high = coating.angular_spectrum.aoi_span_deg
        if not float(low) - AOI_TOLERANCE_DEG <= angle <= float(high) + AOI_TOLERANCE_DEG:
            return COATING_AOI_OUTSIDE_RANGE, angle
    return None


def surface_aoi_checks(elements: tuple[CompiledElement, ...]) -> dict[str, Callable[[float, float], str | None]]:
    """For every compiled surface characterized for a range of angles (a coated surface, a polarization element), keyed by
    its Engine surface id: ``check(aoi_deg, refractive_index) -> code | None`` with the rule of ``aoi_outside_range`` (the
    same rule the run's validity applies). For callers that follow light without tracing it
    (``optics.polarization.routing.PolarizationRouter``)."""

    checks: dict[str, Callable[[float, float], str | None]] = {}
    for element in elements:
        for body in element.bodies():
            for engine_id, envelope in body.envelope_by_engine_id().items():
                if envelope.valid_aoi_deg is None:
                    continue

                def check(aoi_deg: float, refractive_index: float, wavelength_nm: float | None = None, body: CompiledElement = body,
                          envelope: SurfaceEnvelope = envelope) -> str | None:
                    outside = aoi_outside_range(body, envelope, aoi_deg, refractive_index, wavelength_nm)
                    return None if outside is None else outside[0]

                checks[engine_id] = check
    return checks


def _incident_medium(envelope: SurfaceEnvelope | None) -> bool:
    """Whether the surface's angle range is stated in the medium the light arrives from (``aoi_reference``)."""

    coating = envelope.coating if envelope is not None else None
    return coating is not None and coating.aoi_reference is CoatingAoiReference.INCIDENT_MEDIUM


def _message(element: CompiledElement, envelope: SurfaceEnvelope | None, code: str, values: set) -> str:
    """The issue in words: the element by its instance id and product name (its SKU is in the run's simulated_parts), a
    declared model by its name (a compound unit's member by its member id too)."""

    if code in TABULATED_DOMAIN_CODES and envelope is not None and envelope.traced in ENGINE_CHECKED_DOMAINS:
        # the Engine's own words, one sentence per distinct finding, in path order (values: (path_id, wavelength_nm, message))
        sentences = dict.fromkeys(message if message.endswith(".") else f"{message}." for *_, message in sorted(values))
        return " ".join(sentences)
    member = f" member {element.member_id}" if element.member_id is not None else ""
    sku = f"{element.instance_id}{member} ({element.label})"
    catalog = element.identity.is_catalog
    published = "published model" if catalog else "model"
    band = element.valid_band_nm
    band_text = f"{band.min_nm}-{band.max_nm} nm" if band is not None else "every wavelength"
    surface = f"surface {envelope.surface_id}" if envelope is not None else "the element"
    if code == MODEL_WAVELENGTH_OUTSIDE_BAND:
        remedy = "or a product whose band covers it." if catalog else "or a model whose band covers it."
        return (f"{_nm(values)} light interacts with {sku}, whose {published} is characterized for "
                f"{_band_text(band.min_nm, band.max_nm, wavelength_tolerance(element))} only. Use a source wavelength inside that band, "  # type: ignore[union-attr]
                f"{remedy}")
    if code == COATING_WAVELENGTH_OUTSIDE_BANDS:
        coating = envelope.coating  # type: ignore[union-attr]
        table = coating.spectrum or coating.angular_spectrum  # type: ignore[union-attr]
        if table is not None:
            low, high = table.wavelength_nm[0], table.wavelength_nm[-1]
            kind = "spectrum" if coating.spectrum is not None else "angle-resolved table"  # type: ignore[union-attr]
            return (f"{_nm(values)} light meets the coating of {sku} {surface}, whose {kind} is {'published' if catalog else 'stated'} for "
                    f"{low}-{high} nm only; beyond it its reflectance / transmittance are not known. Use a wavelength inside that range, "
                    f"or {'a product whose coating covers it' if catalog else 'a coating that covers it'}.")
        tolerance = wavelength_tolerance(element, coating)  # type: ignore[arg-type]
        bands = ", ".join(_band_text(b.min_nm, b.max_nm, tolerance) for b in coating.bands or ())  # type: ignore[union-attr]
        return (f"{_nm(values)} light meets the coating of {sku} {surface}, which is {'published' if catalog else 'stated'} for {bands} only; between or "
                f"beyond these bands (e.g. a filter's transition edge) its reflectance / transmittance are not known. Use a wavelength inside a "
                f"band, or {'a product whose coating covers it' if catalog else 'a coating that covers it'}.")
    if code in (COATING_AOI_OUTSIDE_RANGE, ELEMENT_AOI_OUTSIDE_RANGE):
        angles = envelope.valid_aoi_deg  # type: ignore[union-attr]
        what = f"the coating of {sku} {surface}" if envelope.coating is not None else sku  # type: ignore[union-attr]
        reading = "angle of incidence in the medium the light arrives from" if _incident_medium(envelope) else "equivalent angle of incidence in air"
        tolerance = angle_tolerance_deg(element, envelope)  # type: ignore[arg-type]
        accepted = f" (a single design angle, matched within +-{tolerance:g} deg)" if tolerance else ""
        return (f"the chief ray meets {what} at {_deg(values)} ({reading}); the {published} is characterized "
                f"for {angles.min_deg}-{angles.max_deg} deg{accepted}. Turn the element so the chief ray's angle of incidence is within "  # type: ignore[union-attr]
                f"that range.")
    if code == MATERIAL_DATA_OUTSIDE_RANGE:
        return (f"the Engine stopped {_nm(values)} light at {sku} {surface}: the wavelength is outside the {'published ' if catalog else ''}dispersion data of the "
                f"element's material (model band {band_text}). Use a wavelength inside the model's band.")
    if code == COATING_DATA_OUTSIDE_RANGE:
        return (f"{_nm(values)} light meets {sku} {surface} outside the Engine's coating range (model band {band_text}): the coating's "
                f"reflectance, transmittance and phases are not stated there, so powers and phases behind this surface are estimates. "
                f"Use a wavelength inside the model's band.")
    if code == BEAM_PARTIALLY_INTERCEPTED:
        rays = ", ".join(f"{count} of {total} pupil rays of beam {path_id} ({wavelength!r} nm)" for path_id, wavelength, count, total in sorted(values))
        return (f"{rays} meet {sku} {surface} while the beam's chief ray does not: part of the beam falls on this element, but the Engine "
                f"follows each beam along its chief ray only, so the power and Gaussian beam downstream are those of an unobstructed beam. "
                f"Move the element or the beam so the beam passes the element completely or clears it.")
    if code == RAY_LEFT_ELEMENT_THROUGH_EDGE:
        ends = ", ".join(f"beam {path_id} ({wavelength!r} nm) {end}" for path_id, wavelength, end in sorted(values))
        return (f"light that entered {sku} through {surface} leaves it through its edge or a face that is not a traced surface, which the "
                f"{published} does not describe: {ends} while still inside the element's medium. Tilt or move the element so the beam "
                f"leaves through its surfaces.")
    if code == ELEMENT_MET_FROM_BEHIND:
        wavelengths = _nm({wavelength for _, wavelength, _ in values})
        lossless = "a product whose coating" if catalog else "a coating that"
        remedy = (f"Use {lossless} is lossless at this wavelength." if any(met_from_front for *_, met_from_front in values) else
                  f"Turn the element so this beam meets the coated face first, or use {lossless} is lossless at this wavelength.")
        return (f"{_beams(values)} is reflected by {sku} at {surface} from behind its coated face: the {'published ' if catalog else ''}coating is not lossless at "
                f"{wavelengths} (reflectance + transmittance < 1), so its reflectance for light from behind is not determined (its transmittance "
                f"is the same from both sides). {remedy}")
    return (f"{_beams(values)} crosses the body of {sku} through its edge or a face that is not a traced surface, without meeting one "
            f"of its surfaces; the Engine traces it in air. Move the element or the beam so the beam passes through the element's surfaces "
            f"or clears it.")


#: ``details.design_point.quantity`` of a finding at a single design point.
ANGLE_OF_INCIDENCE_DEG = "angle_of_incidence_deg"
WAVELENGTH_NM = "wavelength_nm"


def design_point_facts(element: CompiledElement, envelope: SurfaceEnvelope | None, code: str, values: set) -> dict | None:
    """The structured facts of a finding at a single design point (``ModelScopeIssue.details``), ``None`` for any other
    finding: ``{"design_point": {...}}`` with

    * ``quantity``: ``angle_of_incidence_deg`` (``COATING_AOI_OUTSIDE_RANGE`` at a coating published for one angle) or
      ``wavelength_nm`` (``MODEL_WAVELENGTH_OUTSIDE_BAND`` at a model published for one wavelength,
      ``COATING_WAVELENGTH_OUTSIDE_BANDS`` at a coating whose bands are single wavelengths);
    * ``design_values``: the published point(s), ascending; ``observed``: the values the light met it at, ascending;
    * ``reading`` (angles): the coating's ``aoi_reference`` (``AIR_EQUIVALENT`` / ``INCIDENT_MEDIUM``), as ``observed`` is read;
    * ``tolerance_deg`` / ``tolerance_rel``: the tolerance applied (0 where the point is kept exactly); ``setting``: the
      ``NumericalConfig`` field that holds it and ``source``: ``PROJECT`` when the Project's
      ``numerical_settings.design_point`` states it, else ``ENGINE_DEFAULT`` (both absent where no setting applies);
    * ``coating_kind``: the coating's kind (absent for a model band);
    * ``assumed``: ``HELD_AT_NEAREST_STATED_POINT`` where the Engine computed with the coefficient published at the point
      (a metallic, dielectric high-reflector, anti-reflection or beamsplitter coating's single angle; a single wavelength
      of a non-filter form other than a multi-order retarder). Absent where the published value does not stand for the
      light's (an interference coating beyond its edge-shift limit, a filter, a multi-order retarder)."""

    point = element.design_point
    facts: dict
    if code == COATING_AOI_OUTSIDE_RANGE and envelope is not None and envelope.coating is not None and _single_angle(envelope.valid_aoi_deg):
        coating = envelope.coating
        setting = angle_tolerance_setting(coating, envelope.valid_aoi_deg, point)
        facts = {"quantity": ANGLE_OF_INCIDENCE_DEG, "design_values": [float(Decimal(envelope.valid_aoi_deg.min_deg))],  # type: ignore[union-attr]
                 "observed": sorted(float(v) for v in values), "reading": coating.aoi_reference.value,
                 "tolerance_deg": angle_tolerance_deg(element, envelope), "coating_kind": coating.kind.value}
        held = coating.kind in ANGLE_SETTINGS
    elif code == MODEL_WAVELENGTH_OUTSIDE_BAND and envelope is None and element.valid_band_nm is not None \
            and Decimal(element.valid_band_nm.min_nm) == Decimal(element.valid_band_nm.max_nm):
        setting = wavelength_tolerance_setting(element)
        facts = {"quantity": WAVELENGTH_NM, "design_values": [float(Decimal(element.valid_band_nm.min_nm))],
                 "observed": sorted(float(v) for v in values), "tolerance_rel": float(wavelength_tolerance(element))}
        held = setting == WAVELENGTH_SETTING
    elif code == COATING_WAVELENGTH_OUTSIDE_BANDS and envelope is not None and envelope.coating is not None \
            and envelope.coating.spectrum is None and envelope.coating.angular_spectrum is None and envelope.coating.bands \
            and all(Decimal(b.min_nm) == Decimal(b.max_nm) for b in envelope.coating.bands):
        coating = envelope.coating
        setting = wavelength_tolerance_setting(element, coating)
        facts = {"quantity": WAVELENGTH_NM, "design_values": sorted(float(Decimal(b.min_nm)) for b in coating.bands or ()),
                 "observed": sorted(float(v) for v in values), "tolerance_rel": float(wavelength_tolerance(element, coating)),
                 "coating_kind": coating.kind.value}
        held = setting == WAVELENGTH_SETTING
    else:
        return None
    if setting is not None:
        facts["setting"] = setting
        facts["source"] = point.source(setting)
    if held:
        facts["assumed"] = AssumedInput.HELD_AT_NEAREST_STATED_POINT.value
    return {"design_point": facts}


#: The parameter a finding's validity check names, per code (``finding_validity``).
VALIDITY_PARAMETERS = {
    MODEL_WAVELENGTH_OUTSIDE_BAND: WAVELENGTH_NM,
    COATING_WAVELENGTH_OUTSIDE_BANDS: WAVELENGTH_NM,
    MATERIAL_DATA_OUTSIDE_RANGE: WAVELENGTH_NM,
    COATING_DATA_OUTSIDE_RANGE: WAVELENGTH_NM,
    COATING_AOI_OUTSIDE_RANGE: ANGLE_OF_INCIDENCE_DEG,
    ELEMENT_AOI_OUTSIDE_RANGE: ANGLE_OF_INCIDENCE_DEG,
    BEAM_PARTIALLY_INTERCEPTED: "intercepted_pupil_rays",
    RAY_LEFT_ELEMENT_THROUGH_EDGE: "rays_leaving_through_an_untraced_face",
    RAY_CROSSES_ELEMENT_BODY: "rays_crossing_the_body_in_air",
    ELEMENT_MET_FROM_BEHIND: "reflection_from_behind_an_absorbing_coating",
    MODEL_FIELD_ANGLE_EXCEEDED: "field_angle_deg",
    MODEL_BEAM_SIZE_EXCEEDED: "beam_size",
    MODEL_POLARIZATION_NOT_VALIDATED: "polarization",
}
_UNITS = {WAVELENGTH_NM: "nm", ANGLE_OF_INCIDENCE_DEG: "deg", "field_angle_deg": "deg"}


def _limits(element: CompiledElement, envelope: SurfaceEnvelope | None, code: str) -> tuple[float | None, float | None, str | None]:
    """The published range a value of a finding of ``code`` is checked against (``None`` where it is not one range) and the
    setting that holds the tolerance a single point of it is matched within."""

    point = element.design_point
    if code == MODEL_WAVELENGTH_OUTSIDE_BAND and element.valid_band_nm is not None:
        band = element.valid_band_nm
        single = Decimal(band.min_nm) == Decimal(band.max_nm)
        return float(Decimal(band.min_nm)), float(Decimal(band.max_nm)), wavelength_tolerance_setting(element) if single else None
    coating = envelope.coating if envelope is not None else None
    if code == COATING_WAVELENGTH_OUTSIDE_BANDS and coating is not None:
        table = coating.spectrum or coating.angular_spectrum
        if table is not None:
            return float(Decimal(table.wavelength_nm[0])), float(Decimal(table.wavelength_nm[-1])), None
        bands = coating.bands or ()
        single = all(Decimal(b.min_nm) == Decimal(b.max_nm) for b in bands)
        setting = wavelength_tolerance_setting(element, coating) if bands and single else None
        if len(bands) == 1:
            return float(Decimal(bands[0].min_nm)), float(Decimal(bands[0].max_nm)), setting
        return None, None, setting
    if code in (COATING_AOI_OUTSIDE_RANGE, ELEMENT_AOI_OUTSIDE_RANGE) and envelope is not None and envelope.valid_aoi_deg is not None:
        angles = envelope.valid_aoi_deg
        return float(Decimal(angles.min_deg)), float(Decimal(angles.max_deg)), angle_tolerance_setting(coating, angles, point)
    if code == BEAM_PARTIALLY_INTERCEPTED:
        return None, 0.0, None
    return None, None, None


def _observed(code: str, value: object) -> float | None:
    """The number a collected value of a finding states: the wavelength or angle, the intercepted pupil rays of a beam;
    ``None`` for a fact without one."""

    if isinstance(value, (int, float)):
        return float(value)
    if code == BEAM_PARTIALLY_INTERCEPTED:
        return float(value[2])  # type: ignore[index]
    if VALIDITY_PARAMETERS.get(code) == WAVELENGTH_NM and isinstance(value, tuple) and isinstance(value[1], (int, float)):
        return float(value[1])  # (a tabulated lens's or Faraday rotator's issue: (path, wavelength, message))
    return None


def finding_validity(element: CompiledElement, envelope: SurfaceEnvelope | None, code: str, values: set) -> tuple[ValidityCheck, ...]:
    """The validity checks a finding states (``ModelScopeIssue.validity``): one per distinct value the light met the element
    at (a wavelength, an angle of incidence, a beam's intercepted pupil rays), else one for the fact; each with the parameter
    (``VALIDITY_PARAMETERS``), the published limits (``_limits``), ``within`` false, the setting that holds the tolerance of a
    single published point, where (the element, or its Engine surface) and the code."""

    parameter = VALIDITY_PARAMETERS.get(code, code.lower())
    minimum, maximum, setting = _limits(element, envelope, code)
    at = (element.engine_surface_id(envelope.surface_id),) if envelope is not None else (element.instance_id,)
    observed = sorted({number for value in values if (number := _observed(code, value)) is not None})
    return tuple(ValidityCheck(parameter=parameter, unit=_UNITS.get(parameter), minimum=minimum, maximum=maximum, observed=number,
                               within=False, setting=setting, at=at, code=code)
                 for number in (observed or [None]))


def _validity_reach(code: str, value_light: dict[object, tuple[set, set, set]], targets, reads: _TargetReads
                    ) -> tuple[tuple[tuple[str, ...], tuple[tuple[str, str, int], ...], tuple[tuple[str, str, int], ...]], ...]:
    """Per check ``finding_validity`` states (in its order: one per observed number, else one for the fact), the targets
    that read the light met at that check's value and that light's arrivals and sampled lines. ``value_light``: per collected value, the
    arrivals, sampled lines and element beams that met the element at it (``_Collector.value_light``)."""

    numbers = {value: _observed(code, value) for value in value_light}
    observed = sorted({number for number in numbers.values() if number is not None})
    groups: dict[float | None, tuple[set, set, set]] = {}
    for value, (arrivals, lines, beams) in value_light.items():
        if observed and numbers[value] is None:
            continue  # (a value without a number while others have one: no check states it)
        group = groups.setdefault(numbers[value] if observed else None, (set(), set(), set()))
        group[0].update(arrivals)
        group[1].update(lines)
        group[2].update(beams)
    reach = []
    for number in observed or [None]:
        arrivals, lines, beams = groups.get(number, (set(), set(), set()))
        target_ids = tuple(sorted(t.target_id for t in targets if finding_applies(code, t.metric) and reads.reads(t, arrivals, lines, beams)))
        reach.append((target_ids, tuple(sorted(arrivals)), tuple(sorted(lines))))
    return tuple(reach)


@dataclass(frozen=True)
class _Profile:
    """A published surface-sequence surface in floats: ``z(x, y) = vertex + sag`` (``None`` outside its profile), held flat
    beyond its own mechanical ``edge`` when it states one (``surface_mesh.held_point``)."""

    vertex: float
    radius: float | None
    conic: float
    coefficients: tuple[float, ...]
    cylinder_axis: str | None
    linear: float = 0.0  # the axicon term (L rho)
    edge: tuple[float, float] | None = None
    round_edge: bool = True
    form: SagForm | None = None  # a two-variable sag shape

    @property
    def plane(self) -> bool:
        return self.radius is None and not self.coefficients and self.linear == 0.0 and self.form is None

    @property
    def sphere(self) -> bool:
        """A whole sphere (exact crossings); a sphere held flat beyond its own edge is not one."""

        return (self.radius is not None and self.conic == 0.0 and not self.coefficients and self.cylinder_axis is None
                and self.linear == 0.0 and self.edge is None and self.form is None)

    def z(self, x: float, y: float) -> float | None:
        x, y = held_point(x, y, self.edge, self.round_edge)
        rho2 = y * y if self.cylinder_axis == "X" else x * x if self.cylinder_axis == "Y" else x * x + y * y
        sag = 0.0
        if self.form is not None:
            sag, ok = sag_value(self.form, x, y)
            if not ok:
                return None
            if not self.form.additive:
                return self.vertex + sag
        if self.radius is not None:
            curvature = 1.0 / self.radius
            arg = 1.0 - (1.0 + self.conic) * curvature * curvature * rho2
            if arg < 0.0:
                return None
            sag += curvature * rho2 / (1.0 + math.sqrt(arg))
        power = rho2 * rho2
        for coefficient in self.coefficients:
            sag += coefficient * power
            power *= rho2
        if self.linear != 0.0:
            sag += self.linear * math.sqrt(rho2)
        return self.vertex + sag


def _profile(model: SurfaceSequenceModelV1, index: int) -> _Profile:
    surface = model.surfaces[index]
    edge = tuple(float(v) for v in model.surface_edge_mm(index)) if surface.has_own_edge else None
    return _Profile(float(Decimal(surface.vertex_z_mm)), None if surface.radius_mm is None else float(Decimal(surface.radius_mm)),
                    0.0 if surface.conic is None else float(Decimal(surface.conic)),
                    tuple(float(Decimal(a)) for a in surface.aspheric_coefficients or ()), surface.cylinder_axis,
                    0.0 if surface.linear_sag_coefficient is None else float(Decimal(surface.linear_sag_coefficient)),
                    edge, model.is_round, None if surface.sag is None else sag_form(surface.sag))  # type: ignore[arg-type]


@dataclass(frozen=True)
class _Outline:
    """A surface sequence's outline: a circle of ``radius`` or a rectangle of ``half_widths`` about local Z."""

    radius: float | None
    half_widths: tuple[float, float] | None

    def contains(self, x: float, y: float) -> bool:
        if self.radius is not None:
            return math.hypot(x, y) <= self.radius
        return abs(x) <= self.half_widths[0] and abs(y) <= self.half_widths[1]  # type: ignore[index]


def _body_zones(element: CompiledElement) -> tuple[tuple[str, _Profile, _Profile, _Outline], ...]:
    """The solid body of a surface-sequence element in its frame: per non-air medium, (surface id where it starts, front
    surface, back surface, the gap's outline)."""

    model = element.model
    if not isinstance(model, SurfaceSequenceModelV1):
        return ()
    return tuple((model.surfaces[index].surface_id, _profile(model, index), _profile(model, index + 1), _outline(model, index))
                 for index, medium in enumerate(model.media) if medium not in (AIR, VACUUM))


def _outline(model: SurfaceSequenceModelV1, gap: int) -> _Outline:
    """The outline of gap ``gap`` (``SurfaceSequenceModelV1.gap_outline_mm``; the element outline when no surface states its
    own edge)."""

    if not any(s.has_own_edge for s in model.surfaces):
        if model.edge_radius_mm is not None:
            return _Outline(float(Decimal(model.edge_radius_mm)), None)
        return _Outline(None, (float(Decimal(model.edge_half_widths_mm[0])), float(Decimal(model.edge_half_widths_mm[1]))))  # type: ignore[index]
    hx, hy = model.gap_outline_mm(gap)
    return _Outline(float(hx), None) if model.is_round else _Outline(None, (float(hx), float(hy)))


def _roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a t^2 + 2 b t + c`` (a linear equation when ``a`` is 0)."""

    if a == 0.0:
        return [] if b == 0.0 else [-c / (2.0 * b)]
    disc = b * b - a * c
    if disc < 0.0:
        return []
    root = math.sqrt(disc)
    return [(-b - root) / a, (-b + root) / a]


def _surface_cuts(start, d, profile: _Profile, length: float) -> list[float]:
    """Segment parameters where ``start + t d`` crosses a surface: exact for planes and spheres, bracketed on
    ``SAG_CROSSING_STEPS`` steps and bisected to ``BODY_CROSSING_TOLERANCE_MM`` for any other profile."""

    if profile.plane:
        return _roots(0.0, d[2] / 2.0, start[2] - profile.vertex)
    if profile.sphere:
        centre = profile.vertex + profile.radius  # type: ignore[operator]
        rel = (start[0], start[1], start[2] - centre)
        return _roots(sum(v * v for v in d), sum(rel[k] * d[k] for k in range(3)), sum(v * v for v in rel) - profile.radius * profile.radius)  # type: ignore[operator]

    def gap(t: float) -> float | None:
        z = profile.z(start[0] + d[0] * t, start[1] + d[1] * t)
        return None if z is None else start[2] + d[2] * t - z

    cuts = []
    previous_t, previous = 0.0, gap(0.0)
    for step in range(1, SAG_CROSSING_STEPS + 1):
        t = step / SAG_CROSSING_STEPS
        value = gap(t)
        if previous is not None and value is not None and previous * value <= 0.0:
            low, high, f_low = previous_t, t, previous
            for _ in range(SAG_BISECTION_STEPS):
                if (high - low) * length <= BODY_CROSSING_TOLERANCE_MM:
                    break
                middle = (low + high) / 2.0
                f_middle = gap(middle)
                if f_middle is None:
                    break
                if f_low * f_middle <= 0.0:
                    high = middle
                else:
                    low, f_low = middle, f_middle
            cuts.append((low + high) / 2.0)
        previous_t, previous = t, value
    return cuts


def _inside_length(start, end, zone) -> float:
    """Length (mm) of the segment ``start``-``end`` (element frame) inside one body zone: split the segment where it
    crosses the zone's outline or either bounding surface and test each piece's midpoint."""

    _, front, back, outline = zone
    d = [end[k] - start[k] for k in range(3)]
    length = math.sqrt(sum(v * v for v in d))
    if length == 0.0:
        return 0.0
    if outline.radius is not None:
        cuts = _roots(d[0] * d[0] + d[1] * d[1], start[0] * d[0] + start[1] * d[1], start[0] ** 2 + start[1] ** 2 - outline.radius ** 2)
    else:
        cuts = [(sign * half - start[k]) / d[k] for k, half in enumerate(outline.half_widths) for sign in (1.0, -1.0) if d[k] != 0.0]  # type: ignore[arg-type]
    for profile in (front, back):
        cuts += _surface_cuts(start, d, profile, length)
    ts = sorted({0.0, 1.0, *(t for t in cuts if 0.0 < t < 1.0)})
    inside = 0.0
    for t0, t1 in zip(ts, ts[1:]):
        mid = [start[k] + d[k] * (t0 + t1) / 2.0 for k in range(3)]
        if not outline.contains(mid[0], mid[1]):
            continue
        z_front, z_back = front.z(mid[0], mid[1]), back.z(mid[0], mid[1])
        if z_front is not None and z_back is not None and z_front <= mid[2] <= z_back:
            inside += (t1 - t0) * length
    return inside


#: Points along a segment at which a hole bore through a surface-sequence element is checked to hold it.
BORE_SAMPLES = 16


def _in_bore(element: CompiledElement, start, end) -> bool:
    """Does the segment (element frame) run through a hole bored through a surface-sequence element: is every sample
    point of it across the element's vertex span, in local X / Y, inside a hole of every surface (``SurfaceHoleV1``)?"""

    model = element.model
    if not isinstance(model, SurfaceSequenceModelV1) or not all(surface.holes for surface in model.surfaces):
        return False
    from ..optics import apertures
    from .compile import hole_region

    regions = [[hole_region(hole) for hole in surface.holes] for surface in model.surfaces]
    vertices = [float(surface.vertex_z_mm) for surface in model.surfaces]
    low, high = min(vertices), max(vertices)
    held = False
    for k in range(BORE_SAMPLES + 1):
        f = k / BORE_SAMPLES
        x, y, z = (start[i] + f * (end[i] - start[i]) for i in range(3))
        if not low <= z <= high:
            continue
        point = (np.array([x]), np.array([y]))
        if not all(any(bool(apertures.inside(region, *point, 0.0)[0]) for region in holes) for holes in regions):
            return False
        held = True
    return held


def _length_in_body(element: CompiledElement, zones, start, end) -> float:
    """Length (mm) of a segment (element frame) inside the element's solid body (none through a hole bored through it)."""

    if _in_bore(element, start, end):
        return 0.0
    if element.solid is not None:
        interval = clip_segment(element.solid, tuple(start), tuple(end))
        if interval is None:
            return 0.0
        return (interval[1] - interval[0]) * math.sqrt(sum((end[k] - start[k]) ** 2 for k in range(3)))
    return max(_inside_length(tuple(start), tuple(end), zone) for zone in zones)


def _at_observation(segment) -> bool:
    return segment.end_surface_id is not None and segment.end_surface_id.startswith(OBSERVATION_SURFACE_PREFIX)


@dataclass(frozen=True)
class _JoinedBody:
    """One body of a compound unit with joints, for ``RAY_LEFT_ELEMENT_THROUGH_EDGE``: a member (its own medium, its faces at
    unit level and, behind an optical contact, the shared face) or a cement layer (its medium, the two joined faces).
    ``entries``: Engine surface id -> the (body, surface) an issue of light that entered there names."""

    media: frozenset[str]
    entries: dict[str, tuple[CompiledElement, SurfaceEnvelope]]


def _joined_bodies(elements: tuple[CompiledElement, ...]) -> dict[str, list[_JoinedBody]]:
    """Per Engine surface id of a compound unit with joints, the bodies it bounds (``_JoinedBody``)."""

    found: dict[str, list[_JoinedBody]] = {}
    for unit in elements:
        joints = getattr(unit.model, "joints", None) if unit.is_compound else None
        if not joints:
            continue
        present = {s.surface_id: s for s in unit.surfaces}
        members = {m.member_id: m for m in unit.members}
        envelopes = {m.member_id: {e.surface_id: e for e in m.envelopes} for m in unit.members}
        contacts: dict[str, list[tuple[str, str]]] = {}
        bodies: list[_JoinedBody] = []
        for joint in joints:
            a = f"{joint.member_a}{MEMBER_SURFACE_SEPARATOR}{joint.surface_a}"
            b = f"{joint.member_b}{MEMBER_SURFACE_SEPARATOR}{joint.surface_b}"
            if joint.member_a not in members or joint.member_b not in members or a not in present:
                continue
            if joint.cement_material_id is None:
                contacts.setdefault(joint.member_b, []).append((a, b))
            elif b in present:
                cement = present[a].medium_positive
                bodies.append(_JoinedBody(frozenset({cement}), {
                    unit.engine_surface_id(a): (members[joint.member_a], envelopes[joint.member_a][a]),
                    unit.engine_surface_id(b): (members[joint.member_b], envelopes[joint.member_b][b])}))
        for member in unit.members:
            media = frozenset(m for s in member.surfaces for m in (s.medium_negative, s.medium_positive) if m not in ENGINE_SURROUNDINGS)
            own = envelopes[member.member_id]
            entries = {unit.engine_surface_id(sid): (member, e) for sid, e in own.items() if sid in present}
            for a, b in contacts.get(member.member_id, ()):
                if b in own:
                    entries[unit.engine_surface_id(a)] = (member, own[b])
            bodies.append(_JoinedBody(media, entries))
        for body in bodies:
            for engine_id in body.entries:
                found.setdefault(engine_id, []).append(body)
    return found


def _medium_owner(owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]], joined: dict[str, list[_JoinedBody]], entered: str | None,
                  medium: str) -> tuple[CompiledElement, SurfaceEnvelope, frozenset[str]] | None:
    """The body a segment in ``medium`` that entered through ``entered`` runs in: ``(body, surface named, the Engine surfaces
    that bound it)``; ``None`` in air or outside every element. In a compound unit with joints the medium picks the body
    (``_JoinedBody``); elsewhere it is the element whose surface the light entered through."""

    owner = owners.get(entered or "") if medium not in ENGINE_SURROUNDINGS else None
    if owner is None:
        return None
    matching = [body for body in joined.get(entered or "", ()) if medium in body.media]
    if not matching:
        return owner[0], owner[1], frozenset(owner[0].envelope_by_engine_id())
    body, envelope = matching[0].entries[entered or ""]
    return body, envelope, frozenset(engine_id for b in matching for engine_id in b.entries)


def _continues(segments, k: int) -> bool:
    """Does segment ``k`` end inside its medium where the next segment, in the same medium, starts (the straight pieces of
    a chief ray along its curved path in a gradient-index rod)?"""

    segment = segments[k]
    if segment.end_surface_id is not None or segment.end_mm is None or k + 1 >= len(segments):
        return False
    following = segments[k + 1]
    return following.medium_id == segment.medium_id and tuple(following.start_mm) == tuple(segment.end_mm)


def _check_edges(found: _Collector, paths, owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]],
                 joined: dict[str, list[_JoinedBody]] | None = None) -> None:
    """``RAY_LEFT_ELEMENT_THROUGH_EDGE``: chief-ray segments in an element's medium that do not end on its surfaces. A
    segment ending on an observation plane is part of the same run through the medium (the plane only samples the beam),
    and so is a piece of a curved ray in a gradient-index rod (``_continues``). The body is read from the medium
    (``_medium_owner``): behind a joint, the next member or the cement layer."""

    joined = joined or {}
    for path in paths:
        entered = path.branch_surface_id
        distance = found.start_length(path)
        entered_distance = distance
        sampled: list[str] = []  # observation planes the run through the medium crossed so far (named in the message)
        segments = path.segments
        for k, segment in enumerate(segments):
            distance += segment.length_mm or 0.0
            if _at_observation(segment):
                if segment.medium_id not in ENGINE_SURROUNDINGS and segment.end_surface_id not in sampled:
                    sampled.append(segment.end_surface_id)
                continue
            if segment.medium_id not in ENGINE_SURROUNDINGS and _continues(segments, k):
                continue
            owner = _medium_owner(owners, joined, entered, segment.medium_id)
            medium_start_distance = entered_distance
            entered = segment.end_surface_id
            entered_distance = distance
            crossed, sampled = sampled, []
            if owner is None:
                continue
            element, envelope, ends = owner
            if segment.end_surface_id is not None and segment.end_surface_id in ends:
                continue
            end = "leaves the scene" if segment.end_surface_id is None else f"reaches {segment.end_surface_id}"
            if crossed:
                end = f"reaches {', '.join(crossed)}, then {end}"
            found.after(path.path_id, medium_start_distance)
            found.add(element, envelope.surface_id, RAY_LEFT_ELEMENT_THROUGH_EDGE, (path.path_id, path.wavelength_nm, end))


def body_pose(element: CompiledElement, poses: dict[str, Pose]) -> Pose | None:
    """The design-frame pose of an optical body: its element's pose, composed with its member pose in a compound unit."""

    pose = poses.get(element.instance_id)
    if pose is None or element.member_pose is None:
        return pose
    return pose.compose(Pose.from_quat(np.array(element.member_pose.position_mm, dtype=float), np.array(element.member_pose.quaternion_wxyz, dtype=float)))


def _solid_bodies(elements: tuple[CompiledElement, ...], poses: dict[str, Pose]) -> list:
    """Per optical body with a solid: ``(element, design frame to its frame, its zones)``."""

    bodies = []
    for element in (body for e in elements for body in e.bodies()):
        zones = _body_zones(element)
        pose = body_pose(element, poses)
        if (zones or element.solid is not None) and pose is not None:
            bodies.append((element, pose.inverse(), zones))
    return bodies


def _check_bodies(found: _Collector, paths, elements: tuple[CompiledElement, ...], poses: dict[str, Pose]) -> None:
    """``RAY_CROSSES_ELEMENT_BODY``: chief-ray segments in air that run through an element's body."""

    bodies = _solid_bodies(elements, poses)
    if not bodies:
        return
    for path in paths:
        distance = found.start_length(path)
        for segment in path.segments:
            start_distance = distance
            distance += segment.length_mm or 0.0
            if segment.medium_id not in ENGINE_SURROUNDINGS or segment.end_mm is None:
                continue
            for element, to_local, zones in bodies:
                start, end = to_local.apply_points([segment.start_mm, segment.end_mm])
                if _length_in_body(element, zones, [float(v) for v in start], [float(v) for v in end]) > BODY_CROSSING_TOLERANCE_MM:
                    found.after(path.path_id, start_distance)
                    found.add(element, None, RAY_CROSSES_ELEMENT_BODY, (path.path_id, path.wavelength_nm))


def _near_zones(zones, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Which segments (arrays of ends in the element's frame) come near a surface-sequence body: a cheap, generous filter ahead of the exact clip. A sag
    no larger than twice the outline's reach, and the segment's closest approach to the local Z axis."""

    reach = max((zone[3].radius if zone[3].radius is not None else math.hypot(*zone[3].half_widths)) for zone in zones)
    low = min(min(zone[1].vertex, zone[2].vertex) for zone in zones) - SAG_FILTER_REACHES * reach
    high = max(max(zone[1].vertex, zone[2].vertex) for zone in zones) + SAG_FILTER_REACHES * reach
    span = end - start
    across = np.sum(span[:, :2] * span[:, :2], axis=1)
    t = np.clip(np.divide(-np.sum(start[:, :2] * span[:, :2], axis=1), across, out=np.zeros(across.shape), where=across > 0.0), 0.0, 1.0)
    nearest = start[:, :2] + t[:, None] * span[:, :2]
    return (np.sum(nearest * nearest, axis=1) <= reach * reach) & (np.minimum(start[:, 2], end[:, 2]) <= high) & (np.maximum(start[:, 2], end[:, 2]) >= low)


def _rays(count: int, source_id: str, spectral_index: int) -> str:
    return f"{count} rays of {source_id}#{spectral_index}"


def _check_line(found: _Collector, scope: SampledScope, key: tuple[str, int], wavelength_nm: float, planes: set[str],
                owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]]) -> None:
    """The interaction checks of the rays of one sampled line (``key``: source id, spectral index) on every surface they met,
    and the data ranges where they stopped (``scope``), each issue reaching the line on ``planes`` and the elements the rays
    met."""

    found.location = found.resolved_location = None
    found.context_lines = {(oid, *key) for oid in planes}
    found.context = set(planes)
    found.context_arrivals = set()
    found.context_devices = {owners[surface_id][0].instance_id for surface_id, source_id, j, *_ in scope.meets
                             if (source_id, j) == key and surface_id in owners}
    found.beam = (key[0], wavelength_nm)
    found.context_device_beams = {(instance_id, *found.beam) for instance_id in found.context_devices}
    for (surface_id, source_id, j, event, index), (smallest, largest, _count) in scope.meets.items():
        owner = owners.get(surface_id)
        if owner is None or (source_id, j) != key or event not in INTERACTION_EVENTS:
            continue
        _check_interaction(found, owner[0], owner[1], wavelength_nm, smallest, index)
        if largest != smallest:
            _check_interaction(found, owner[0], owner[1], wavelength_nm, largest, index)
    for surface_id, source_id, j, termination in scope.stops:
        owner = owners.get(surface_id)
        if owner is not None and (source_id, j) == key and termination in (MATERIAL_OUT_OF_RANGE, COATING_OUT_OF_RANGE):
            code = MATERIAL_DATA_OUTSIDE_RANGE if termination == MATERIAL_OUT_OF_RANGE else COATING_DATA_OUTSIDE_RANGE
            found.add(owner[0], owner[1].surface_id, code, wavelength_nm)


def _check_scattered(found: _Collector, scattered: ScatterTrace, owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]]) -> None:
    """The checks of ``_check_line`` for the rays of the light each surface scattered, origin by origin (``ScatterLine.scope``):
    an issue of an origin's rays reaches its line on the planes that origin's rays cross, not another origin's of the same
    source line. The edge and body checks are not applied to them (module docstring)."""

    for item in scattered.lines:
        origin = item.origin
        _check_line(found, item.scope, (origin.source_id, origin.spectral_index), origin.wavelength_nm, set(item.rows), owners)


def _check_sampled(found: _Collector, sampled: SampledTrace, owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]], elements: tuple[CompiledElement, ...],
                   poses: dict[str, Pose] | None, joined: dict[str, list[_JoinedBody]] | None = None) -> None:
    """The checks above for every ray of the extended emitters (module docstring), from what the trace kept of them (``SampledScope``)."""

    scope = sampled.scope
    joined = joined or {}
    for line in sampled.lines:
        key = (line.source_id, line.spectral_index)
        # (an issue of these rays reaches what the rays of the line reach)
        _check_line(found, scope, key, line.wavelength_nm, {oid for oid, source_id, j in sampled.rows if (source_id, j) == key}, owners)
        for (entered, ended, source_id, j, medium), count in scope.exits.items():
            owner = _medium_owner(owners, joined, entered, medium) if (source_id, j) == key else None
            if owner is None or (ended is not None and ended in owner[2]):
                continue
            end = "leave the scene" if ended is None else f"reach {ended}"
            found.add(owner[0], owner[1].surface_id, RAY_LEFT_ELEMENT_THROUGH_EDGE, (_rays(count, source_id, j), line.wavelength_nm, end))
        flat = scope.segments.get((*key, ENGINE_AIR))
        if flat and poses is not None:
            ends = np.array(flat).reshape(-1, 2, 3)
            for element, to_local, zones in _solid_bodies(elements, poses):
                start, end = to_local.apply_points(ends[:, 0]), to_local.apply_points(ends[:, 1])
                near = _near_zones(zones, start, end) if zones and element.solid is None else np.ones(start.shape[0], dtype=bool)
                count = sum(1 for k in np.flatnonzero(near)
                            if _length_in_body(element, zones, [float(v) for v in start[k]], [float(v) for v in end[k]]) > BODY_CROSSING_TOLERANCE_MM)
                if count:
                    found.add(element, None, RAY_CROSSES_ELEMENT_BODY, (_rays(count, *key), line.wavelength_nm))


def _transmitting_coated_face(element: CompiledElement) -> str | None:
    """The surface id of a face-normal element's coated face (at the origin, normal local +Z) when it also transmits (a
    dichroic, a plate splitter); ``None`` otherwise."""

    model = element.model
    if not element.identity.is_reflector:
        return None
    split = TracedBranches.TRANSMITTED_AND_REFLECTED
    if isinstance(model, SurfaceSequenceModelV1):
        return model.surfaces[-1].surface_id if model.surfaces[-1].traced is split else None
    if isinstance(model, SurfaceSetModelV1):
        return next((s.surface_id for s in model.surfaces if s.traced is split and all(Decimal(v) == 0 for v in s.origin_mm)
                     and tuple(Decimal(v) for v in s.normal) == (0, 0, 1)), None)
    return None


def _lossless(band: CoatingBandV1) -> bool:
    """Reflectance + transmittance = 1 (within ``LOSSLESS_TOLERANCE``) for unpolarized light and, when the band states
    them, for s and for p."""

    pairs = [(band.reflectance, band.transmittance)]
    if band.polarization_resolved:
        pairs += [(band.reflectance_s, band.transmittance_s), (band.reflectance_p, band.transmittance_p)]
    return all(abs(Decimal(r) + Decimal(t) - 1) <= LOSSLESS_TOLERANCE for r, t in pairs)  # type: ignore[arg-type]


def _incident_segment(parent, branch):
    """The parent's chief-ray segment that ends where ``branch`` starts (the light the branch reflects)."""

    if parent is None or not branch.segments:
        return None
    start = np.array(branch.segments[0].start_mm)
    candidates = [s for s in parent.segments if s.end_surface_id == branch.branch_surface_id and s.end_mm is not None
                  and s.end_mm != s.start_mm]  # a plane crossed on the element's surface ends a zero-length segment
    return min(candidates, key=lambda s: float(np.linalg.norm(np.subtract(s.end_mm, start))), default=None)


def _check_back_reflections(found: _Collector, result: ValidationResult, owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]],
                            poses: dict[str, Pose]) -> None:
    """``ELEMENT_MET_FROM_BEHIND``: chief rays a transmitting face-normal element reflects from behind (arriving along its
    local +Z) at a coating that is not lossless at their wavelength (its reflectance from behind is not determined)."""

    axes = {}  # the coated face only: light reaching a back (anti-reflection) face from outside meets that face's own front
    for engine_id, (element, envelope) in owners.items():
        pose = body_pose(element, poses)
        if pose is not None and envelope.surface_id == _transmitting_coated_face(element):
            axes[engine_id] = (element, envelope, pose.apply_dirs(np.array(LOCAL_Z)))
    if not axes:
        return
    met_from_front: set[str] = set()  # elements a chief ray in air meets travelling along local -Z (onto the coated face)
    for path in result.paths:
        for segment in path.segments:
            hit = axes.get(segment.end_surface_id or "")
            if hit is not None and segment.medium_id in ENGINE_SURROUNDINGS and segment.end_mm is not None:
                if float(np.subtract(segment.end_mm, segment.start_mm) @ hit[2]) < 0.0:
                    met_from_front.add(hit[0].engine_surface_id(hit[1].surface_id))
    paths = {path.path_id: path for path in result.paths}
    for branch in result.paths:
        hit = axes.get(branch.branch_surface_id or "")
        if hit is None or branch.branch_kind not in REFLECTION_BRANCHES:
            continue
        element, envelope, axis = hit
        incident = _incident_segment(paths.get(branch.parent_path_id or ""), branch)
        if incident is None or float(np.subtract(incident.end_mm, incident.start_mm) @ axis) <= 0.0:
            continue
        band = (_covering_band(element, envelope.coating, _decimal(branch.wavelength_nm), incident.angle_of_incidence_deg)
                if envelope.coating is not None else None)
        if band is None or _lossless(band):  # uncoated: bare Fresnel; outside the coating's bands: reported on its own
            continue
        found.after(branch.path_id, found.start_length(branch))
        found.add(element, envelope.surface_id, ELEMENT_MET_FROM_BEHIND,
                  (branch.path_id, branch.wavelength_nm, element.engine_surface_id(envelope.surface_id) in met_from_front))


def _check_interceptions(found: _Collector, ray_paths: RayPathTrace, owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]]) -> None:
    """``BEAM_PARTIALLY_INTERCEPTED``: pupil rays counted with the chief that meet another catalog surface first."""

    for ray_path in ray_paths.paths:
        counts: dict[str, int] = {}
        cutoffs: dict[str, float] = {}
        path = found.paths[ray_path.path_id]
        for ray in ray_path.pupil_rays:
            if ray.end == NOT_FOLLOWED and ray.counted_with_chief and ray.next_surface_id in owners:
                counts[ray.next_surface_id] = counts.get(ray.next_surface_id, 0) + 1  # type: ignore[index]
                last = ray.steps[-1] if ray.steps else len(ray.points_mm) - 1  # (the step the ray was last recorded at)
                cutoff = found.start_length(path) + sum(s.length_mm or 0.0 for s in path.segments[:max(0, last)])
                cutoffs[ray.next_surface_id] = min(cutoffs.get(ray.next_surface_id, cutoff), cutoff)
        for surface_id, count in counts.items():
            element, envelope = owners[surface_id]
            found.after(ray_path.path_id, cutoffs[surface_id])
            found.add(element, envelope.surface_id, BEAM_PARTIALLY_INTERCEPTED,
                      (ray_path.path_id, ray_path.wavelength_nm, count, ray_path.pupil_ray_count))


def _tabulated_crossing(found: _Collector, path, surface_id: str, issue) -> float:
    """Where on ``path`` the Engine judged a tabulated thin lens's or Faraday rotator's domain (cumulative geometric length):
    its first crossing of ``surface_id`` -- for ``MODEL_FIELD_ANGLE_EXCEEDED`` the first one beyond the validated angle,
    by the Engine's own rule. A crossing not found in the recorded segments is taken at the path's start (so nothing the
    path reaches is left out)."""

    limit = issue.numerical_context.get("max_field_angle_deg")
    angle_rule = issue.code.value == MODEL_FIELD_ANGLE_EXCEEDED and isinstance(limit, (int, float))
    distance = found.start_length(path)
    first = None
    for segment in path.segments:
        distance += segment.length_mm or 0.0
        if segment.end_surface_id != surface_id or segment.end_event not in ENGINE_CHECKED_EVENTS:
            continue
        if not angle_rule:
            return distance
        first = distance if first is None else first
        angle = segment.angle_of_incidence_deg
        if angle is not None and beyond_limit(angle, float(limit), DOMAIN_LIMIT_ABS_TOLERANCE_DEG):
            return distance
    return first if first is not None else found.start_length(path)


def _check_tabulated_domains(found: _Collector, result: ValidationResult, owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]]) -> None:
    """The Engine's validity-domain issues of tabulated thin lenses and Faraday rotators (module docstring), each from the
    crossing it judged."""

    paths = {p.path_id: p for p in result.paths}
    for issue in result.issues:
        location = issue.location or ""
        if issue.code.value not in TABULATED_DOMAIN_CODES or issue.status.value != "OUT_OF_MODEL_SCOPE" or not location.startswith("surface:"):
            continue
        surface_id = location.removeprefix("surface:")
        owner = owners.get(surface_id)
        if owner is None or owner[1].traced not in ENGINE_CHECKED_DOMAINS:
            continue  # (the Engine reports these codes only at these elements, and every lowered one is an owner)
        path = paths.get(str(issue.numerical_context.get("path_id")))
        # an issue whose path the run does not list: every path that crosses the lens, from its first crossing
        crossing = [path] if path is not None else [p for p in result.paths if any(
            s.end_surface_id == surface_id and s.end_event in ENGINE_CHECKED_EVENTS for s in p.segments)]
        for p in crossing:
            found.after(p.path_id, _tabulated_crossing(found, p, surface_id, issue))
            found.add(owner[0], owner[1].surface_id, issue.code.value, (p.path_id, p.wavelength_nm, issue.message))


def _check_held_coatings(found: _Collector, result: ValidationResult, owners) -> None:
    """``COATING_DATA_OUTSIDE_RANGE`` for every beam the Engine carried on through a coating of an element outside its data
    (module docstring): its ``COATING_OUT_OF_RANGE`` issue at the surface names the path; what the path reaches behind its
    first crossing of the surface is affected. An issue whose path the run does not list is taken at every path that
    meets the surface. A ``WARNING`` of that code is not such a beam: the pupil rays beside a chief ray inside the data,
    held at its edge, carry no power of the beam (``Tracer._live_coating``)."""

    paths = {p.path_id: p for p in result.paths}
    for issue in result.issues:
        location = issue.location or ""
        if issue.code.value != COATING_OUT_OF_RANGE or not location.startswith("surface:") or issue.status.value == "WARNING":
            continue
        surface_id = location.removeprefix("surface:")
        owner = owners.get(surface_id)
        if owner is None:
            continue
        path = paths.get(str(issue.numerical_context.get("path_id")))
        meeting = [path] if path is not None else [p for p in result.paths if any(s.end_surface_id == surface_id for s in p.segments)]
        for p in meeting:
            distance = found.start_length(p)
            for segment in p.segments:
                distance += segment.length_mm or 0.0
                if segment.end_surface_id == surface_id:
                    break
            else:
                distance = found.start_length(p)  # (not in the recorded segments: nothing the path reaches is left out)
            found.after(p.path_id, distance)
            found.add(owner[0], owner[1].surface_id, COATING_DATA_OUTSIDE_RANGE, p.wavelength_nm)


class _TargetReads:
    """Whether a target reads the light a finding reaches: the beams and sampled lines its selection reads on its plane
    (``validation.metrics.selected_light``), and on the element it names (``device_id``) the beams of its source and
    wavelength -- every beam on it for an element metric (``ELEMENT_METRICS``: a rating's utilization reads the element's
    exposure by all of them, whatever source or wavelength the target states)."""

    def __init__(self, result: ValidationResult, numerical: NumericalConfig) -> None:
        self.observations = {o.observation_id: o for o in result.observations}
        self.numerical = numerical
        self._light: dict[str, tuple[frozenset, frozenset]] = {}

    def light(self, t: Target) -> tuple[frozenset[tuple[str, str, int]], frozenset[tuple[str, str, int]]]:
        """The arrivals ``(observation_id, path_id, crossing_index)`` and sampled lines ``(observation_id, source_id,
        spectral_index)`` the target reads."""

        found = self._light.get(t.target_id)
        if found is None:
            observation = self.observations.get(t.observation_id) if t.observation_id is not None else None
            arrivals, lines = selected_light(t, observation, self.numerical)
            oid = t.observation_id
            found = self._light[t.target_id] = (frozenset((oid, a.beam.path_id, a.crossing_index) for a in arrivals),
                                                frozenset((oid, line.source_id, line.spectral_index) for line in lines))
        return found

    def _beam(self, t: Target, source_id: str | None, wavelength_nm: float | None) -> bool:
        if source_id is None:  # (a check that does not know its beam: any beam)
            return True
        return ((t.source_id is None or source_id == t.source_id)
                and (t.wavelength_nm is None or abs(wavelength_nm - t.wavelength_nm) <= self.numerical.interference_wavelength_tol_nm))  # type: ignore[operator]

    def reads(self, t: Target, arrivals: set, lines: set, device_beams: set) -> bool:
        if not isinstance(t, Target):  # (a result's target row states no selection: every beam on its plane and element)
            return ((t.device_id is not None and any(device == t.device_id for device, *_ in device_beams))
                    or any(oid == t.observation_id for oid, *_ in (*arrivals, *lines)))
        if t.device_id is not None and any(device == t.device_id and (t.metric in ELEMENT_METRICS or self._beam(t, source_id, wavelength))
                                           for device, source_id, wavelength in device_beams):
            return True
        if t.observation_id is None:
            return False
        selected_arrivals, selected_lines = self.light(t)
        return not selected_arrivals.isdisjoint(arrivals) or not selected_lines.isdisjoint(lines)


def element_scope_issues(result: ValidationResult, elements: tuple[CompiledElement, ...], *, ray_paths: RayPathTrace | None = None,
                         poses: dict[str, Pose] | None = None, tabulated_domains: bool = True,
                         targets: Iterable[Target] | None = None, numerical: NumericalConfig | None = None) -> tuple[ModelScopeIssue, ...]:
    """Every model validity violation of ``elements`` in ``result`` (empty = within scope), in element order (a compound
    unit's issues member by member).

    ``BEAM_PARTIALLY_INTERCEPTED`` needs the run's ``ray_paths`` and ``RAY_CROSSES_ELEMENT_BODY`` the elements' design-frame
    ``poses`` (by instance id); ``model_scope_issues`` always passes both. ``tabulated_domains``: also list the Engine's own
    validity-domain issues of tabulated thin lenses and Faraday rotators (``TABULATED_DOMAIN_CODES``) with what they affect.
    ``targets``: the targets a finding can affect, each by the plane or device it names (default: the targets ``result``
    assessed, whose rows state no selection: each reads every beam on its plane); ``model_scope_issues`` passes the request's declared targets, so a finding names every declared target
    that reads its light, whether or not the evaluation assessed it. A finding names a target when the target reads the
    light the finding reaches -- on its plane, a beam (path and crossing) or a sampled line its selection reads
    (``validation.metrics.selected_light``: its source, wavelength, path and crossing, its pair; no selection reads every
    beam on the plane; a polarization metric whose path selects one polarization component reads both components of its
    route); on the element it names, the beams of its source and wavelength, and for an element metric (``ELEMENT_METRICS``:
    ``power_rating_utilization``) every beam on the element, whatever source or wavelength the target states (the element's
    exposure by all of them) -- and its metric reads a kind of quantity the finding's code touches (``finding_applies``): a
    target on a beam's position behind a coating used outside its bands keeps the Engine's own assessment, and so does a
    target whose selection on the same plane reads only another source's or wavelength's beam. ``numerical``: the settings the run's targets select their beams with (the wavelength tolerance; default the
    Engine's)."""

    owners: dict[str, tuple[CompiledElement, SurfaceEnvelope]] = {}
    for element in elements:
        for body in element.bodies():
            for engine_id, envelope in body.envelope_by_engine_id().items():
                owners[engine_id] = (body, envelope)
    found = _Collector(result, owners)
    sampled = None if ray_paths is None else ray_paths.sampled
    # (an extended emitter's rays are read all together, by _check_sampled)
    beams = result.paths if sampled is None else [p for p in result.paths if p.source_id not in {line.source_id for line in sampled.lines}]
    for path in beams:
        checkpoint()  # (one more path checked: the run's monitor may stop the run here, os_optics/run_control.py)
        distance = found.start_length(path)
        for segment in path.segments:
            distance += segment.length_mm or 0.0
            owner = owners.get(segment.end_surface_id or "")
            if owner is None or segment.end_event not in INTERACTION_EVENTS:
                continue
            found.after(path.path_id, distance)
            _check_interaction(found, owner[0], owner[1], path.wavelength_nm, segment.angle_of_incidence_deg, segment.refractive_index)
        stopped = owners.get(path.termination_surface_id or "")
        if stopped is not None and path.termination in (MATERIAL_OUT_OF_RANGE, COATING_OUT_OF_RANGE):
            found.after(path.path_id, distance)
            code = MATERIAL_DATA_OUTSIDE_RANGE if path.termination == MATERIAL_OUT_OF_RANGE else COATING_DATA_OUTSIDE_RANGE
            found.add(stopped[0], stopped[1].surface_id, code, path.wavelength_nm)
    _check_held_coatings(found, result, owners)
    if tabulated_domains:
        _check_tabulated_domains(found, result, owners)
    if ray_paths is not None:
        _check_interceptions(found, ray_paths, owners)
    joined = _joined_bodies(elements)
    _check_edges(found, beams, owners, joined)
    if poses is not None:
        _check_bodies(found, beams, elements, poses)
        _check_back_reflections(found, result, owners, poses)
    if sampled is not None:
        _check_sampled(found, sampled, owners, elements, poses, joined)
    if ray_paths is not None and ray_paths.scattered is not None:
        _check_scattered(found, ray_paths.scattered, owners)
    issues: list[ModelScopeIssue] = []
    reads = _TargetReads(result, numerical if numerical is not None else NumericalConfig())
    candidates = tuple(result.targets if targets is None else targets)
    for element in (body for e in elements for body in e.bodies()):
        envelopes = {e.surface_id: e for e in element.envelopes}
        body_key = (element.instance_id, element.member_id)
        keys = [key for key in found.values if key[0] == body_key]
        keys.sort(key=lambda key: (MODEL_SCOPE_CODES.index(key[2]), key[1] or ""))
        for key in keys:
            _, surface_id, code = key
            envelope = envelopes.get(surface_id) if surface_id is not None else None
            observations = tuple(sorted(found.affected.get(key, ())))
            affected = tuple(sorted(t.target_id for t in candidates
                                    if finding_applies(code, t.metric)
                                    and reads.reads(t, found.arrivals.get(key, set()), found.lines.get(key, set()),
                                                    found.device_beams.get(key, set()))))
            issues.append(ModelScopeIssue(element.instance_id, code, _message(element, envelope, code, found.values[key]), surface_id,
                                          element.member_id, observations, affected, tuple(sorted(found.devices.get(key, ()))),
                                          details=design_point_facts(element, envelope, code, found.values[key]),
                                          affected_arrivals=tuple(sorted(found.arrivals.get(key, ()))),
                                          affected_sampled_lines=tuple(sorted(found.lines.get(key, ()))),
                                          validity=finding_validity(element, envelope, code, found.values[key]),
                                          validity_reach=_validity_reach(code, found.value_light.get(key, {}), candidates, reads)))
    return tuple(issues)


def element_poses(lowered: LoweredProject) -> dict[str, Pose]:
    """The design-frame pose of every lowered component, by instance id."""

    return {c.instance_id: Pose.from_quat(c.pose.position_mm, c.pose.quaternion_wxyz) for c in lowered.request.project.components}


def model_scope_issues(lowered: LoweredProject, result: ValidationResult, ray_paths: RayPathTrace, *,
                       tabulated_domains: bool = True) -> tuple[ModelScopeIssue, ...]:
    """Every model validity violation of the run's elements, catalog and declared (empty = within scope), including the
    Engine's own validity-domain issues of tabulated thin lenses unless ``tabulated_domains`` is false. Each finding names
    the declared targets on the planes and devices it affects (``affected_target_ids``): the relation is the Project's, so
    an evaluation that assessed no target, or a selection of them, names the same targets as the complete run."""

    return element_scope_issues(result, lowered.elements, ray_paths=ray_paths, poses=element_poses(lowered),
                                tabulated_domains=tabulated_domains, targets=lowered.request.project.targets,
                                numerical=lowered.request.configuration.numerical)
