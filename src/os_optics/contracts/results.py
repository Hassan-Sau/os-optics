"""Result contracts: ValidationResult and its parts.

All numeric fields are finite or ``None``; a ``None`` value always comes with a status/issue that
explains why.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import model_serializer, Field, model_validator

from ..errors.issues import ValidationIssue
from ..errors.status import QuantityState, RequirementAssessment, Status, assessment_of
from .base import OmitAbsentFields, StrictModel, Vector2, Vector3, schema_description
from .basis import BasisCarrier, ValidityCheck, ValueBasisField
from .detection import DetectionResult
from .evaluation import EvaluationScope
from .manifest import ValidationManifest
from .optics import Aperture, BeamState, OpticalPath
from .outcome import ResultOutcome

# ------------------------------------------------------------------------------------------ targets


class TargetResult(BasisCarrier, OmitAbsentFields, StrictModel):
    absent_omitted: ClassVar[tuple[str, ...]] = ("value_basis", "validity")

    target_id: str
    device_id: str | None = None
    observation_id: str | None = None
    metric: str
    value: float | None = None
    target_value: float | None = None
    tolerance: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    error: float | None = None
    unit: str | None = None
    status: Status
    assessment: RequirementAssessment | None = Field(
        None,
        description="The requirement assessment of this declared target, derived from its status: MET, UNMET, "
        "INDETERMINATE or NOT_APPLICABLE. It answers whether the design meets what the customer asked for; status "
        "stays the diagnostic severity. A Project with no targets has no target results at all, so no requirement "
        "is assessed -- which is not universal success.",
    )
    source_revision: str | None = None
    message: str = ""
    issue_codes: tuple[str, ...] = ()
    value_basis: ValueBasisField = None
    validity: tuple[ValidityCheck, ...] = Field(
        (), description="The validity checks of the model the metric needs that failed or stay open for this case: a Gaussian "
        "beam beyond an accuracy limit of the first-order model (the target then has no value), or a detector's linear range "
        "or bandwidth that light left out as negligible may exceed (within=null; the target is then graded on what holds "
        "without the detector model). Each states the quantity, the value observed where known, the limit and its setting, "
        "where. Omitted when empty.")

    @model_validator(mode="before")
    @classmethod
    def _derive_assessment(cls, data: object) -> object:
        """Derive the assessment from the status when it is not given (a result stored before the field existed)."""

        if isinstance(data, dict) and data.get("status") is not None:
            expected = assessment_of(Status(data["status"]))
            if data.get("assessment") is not None and data["assessment"] != expected:
                raise ValueError("target assessment contradicts its status")
            return {**data, "assessment": expected}
        return data

    def model_copy(self, *, update=None, deep=False):
        # Scope checks refine an already evaluated target. Pydantic's model_copy skips
        # validators, so a status refinement must update its derived assessment too.
        if update is not None and "status" in update:
            update = {**update, "assessment": assessment_of(Status(update["status"]))}
        return super().model_copy(update=update, deep=deep)


# ------------------------------------------------------------------------------------- observations


class PolarizationStateSummary(StrictModel):
    """The polarization state of a beam: of a traced beam, the ellipse of its Jones vector (degree of polarization 1); of
    the partially polarized light on a route (``PolarizationMixture``), the ellipse of its polarized part, with ``s1..s3``
    normalized by the total ``S0`` (so ``sqrt(s1^2 + s2^2 + s3^2)`` is the degree of polarization). Stokes parameters in
    the beam's transverse basis ``(e1, e2)`` (``BeamState.basis``):
    ``S0 = |Ex|^2 + |Ey|^2``, ``S1 = |Ex|^2 - |Ey|^2``, ``S2 = 2 Re(conj(Ex) Ey)``, ``S3 = 2 Im(conj(Ex) Ey)``; ``s1..s3`` are
    normalized by ``S0``. ``s3 = -1`` is right circular (``JonesVector.right_circular``: clockwise viewed facing the source),
    ``+1`` left circular. ``ellipticity_deg = asin(s3) / 2`` (0 linear, -45 right circular). The major axis is given as a 3D
    unit vector (``major_axis``, design frame; sign arbitrary) and as its angle in the observation plane
    (``azimuth_in_plane_deg``: from the plane's u axis toward v, in [0, 180); ``None`` for circular light)."""

    s1: float
    s2: float
    s3: float
    ellipticity_deg: float
    handedness: Literal["LINEAR", "RIGHT", "LEFT"]
    major_axis: Vector3 | None = None
    azimuth_in_plane_deg: float | None = None
    degree_of_polarization: float = Field(
        1.0, description="1 for a traced beam (a Jones vector is fully polarized; a component of a partially polarized source "
        "included); the degree of polarization of the light on a route in a PolarizationMixture.")


class FirstOrderTransfer(StrictModel):
    """First-order (paraxial, real-ray finite-difference) ray-transfer matrix from the source plane to this crossing.

    ``matrix[i][j] = d out_i / d in_j`` with inputs ``(x0, y0, a0, b0)``: a parallel displacement of the chief ray at the
    source origin along the source frame x / y (mm) and a tilt toward the source x / y (rad); outputs ``(u, v, l_u, l_v)``:
    the hit point on the observation plane (mm) and the direction cosines along the plane's u / v axes. The ``A`` block
    ``[[du/dx0, du/dy0], [dv/dx0, dv/dy0]]`` is the transverse magnification matrix when this plane is an image of the
    source plane (``B`` block ~ 0) and the focusing state otherwise. Computed by central differences of eight
    differential rays traced with the chief ray (``first_order_position_step_mm``, ``first_order_angle_step_rad``).

    Image orientation (viewed along the propagation direction at both ends): ``reflection_count`` counts the reflections
    in the path's lineage; ``handedness_preserved`` is ``reflection_count`` even (an odd count mirrors the image).
    ``image_rotation_deg`` is the rotation of the source frame onto the plane frame seen along the propagation (0 upright,
    180 inverted; plane frame taken as ``(u, v)`` when the beam travels along the plane normal, else ``(u, -v)``), only when
    the handedness is preserved and the A block is not singular; ``mirror_axis_deg`` is the in-plane mirror-line angle for a
    mirrored image. ``transverse_magnification`` is ``sqrt(|det A|)`` (mm/mm).
    """

    matrix: tuple[tuple[float, float, float, float], ...]
    reflection_count: int
    handedness_preserved: bool
    image_rotation_deg: float | None = None
    mirror_axis_deg: float | None = None
    transverse_magnification: float | None = None


class GratingOrderStep(StrictModel):
    """One grating order on a beam's way: the grating's surface and the order the beam left it in."""

    surface_id: str
    order: int


class SurfaceCoincidence(OmitAbsentFields, StrictModel):
    """A plane through a physical surface's chief-ray hit point (within ``ray_min_advance_mm``). ``ARRIVING``: the beam's power
    at that surface before the interaction, which is the arrival's value. ``RETURNING``: the beam that leaves the surface back
    across the plane, a separate arrival with its own key."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("crossing_power_upper_bound_W",)

    surface_id: str
    role: Literal["ARRIVING", "RETURNING"]
    surface_event: Literal["REFLECTIVE", "TRANSMISSIVE", "ABSORBING"] = Field(
        description="What the surface does to the arriving beam. REFLECTIVE: the power arriving at the surface and the power "
        "of the beam that leaves it are different quantities; a target reads the arrival.")
    crossing_power_upper_bound_W: float | None = Field(
        None, ge=0.0, description="Auxiliary: the most power of this beam that crosses the literal plane before the surface, where "
        "that is bounded (a flat surface tilted to the plane and a valid Gaussian beam). It never replaces the arrival's power "
        "and no target reads it. Omitted elsewhere.")


class ArrivalResult(BasisCarrier, OmitAbsentFields, StrictModel):
    """One beam arriving at an observation plane (chief ray + Gaussian + polarization)."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("power_out_of_scope_at", "group_path_length_mm", "group_delay_dispersion_mm2",
                                                 "group_delay_unknown_at", "grating_orders", "power_upper_bound_W", "relative_path_mm",
                                                 "surface_coincidence", "polarization_component", "power_basis", "power_sampling_resolution_W",
                                                 "power_taken_before_at", "power_taken_before_upper_W", "power_taken_before_lower_W",
                                                 "reference_ray_promoted_at", "value_basis")

    beam: BeamState
    uv_mm: Vector2 = Field(description="Chief-ray position in the plane frame (u = frame x, v = frame y).")
    aoi_deg: float
    within_acceptance: bool | None = Field(
        None, description="Whether the chief-ray AOI is within the plane's acceptance cone (None: the plane declares none). A beam "
        "outside it is not detected: on an absorbing plane detected_power_W is 0 (the plane stops it on its active area, the power "
        "ledger's OUTSIDE_ACCEPTANCE), and a transparent plane receives no power from it.")
    acceptance_margin_deg: float | None = None
    active_area_power_fraction: float | None = Field(None, description="Gaussian power fraction inside the active area.")
    capture_margin_mm: float | None = Field(
        None, description="Distance from the 1/e^2 beam edge (along the radial direction) to the active-area edge; negative = clipped."
    )
    pupil_ray_count: int = 0
    pupil_rays_arrived: int = 0
    pupil_centroid_uv_mm: Vector2 | None = None
    pupil_rms_radius_mm: float | None = None
    crossing_index: int = Field(0, ge=0, description="0 for the path's first crossing of this plane, 1 for its second (folded / double pass), ...")
    detected_power_W: float | None = Field(
        None, description="On an absorbing plane: the power it detects from this beam (the power ledger's DETECTED): the beam's power on "
        "the active area (beam.power_W x active_area_power_fraction), or the whole beam without an active area; 0 for a beam outside "
        "the acceptance cone. None on a transparent plane, where beam.power_W is the power crossing it. beam.power_W is always the "
        "incident beam's power.")
    detected_power_min_W: float | None = Field(
        None, description="A lower bound of detected_power_W when that value is not exact: an annular beam (behind a central "
        "obstruction) counted whole on an active area that may miss up to capture_min_power_fraction of it. The detected power lies "
        "between this and detected_power_W; a power target whose limit falls in between is UNKNOWN (INCOMPLETE_ENVELOPE). None: "
        "detected_power_W is exact.")
    parent_path_id: str | None = None
    lineage: tuple[str, ...] = Field((), description="'<surface_id>:<event>' interactions from the source up to this plane.")
    direction_uvn: Vector3 | None = Field(None, description="Chief-ray direction in the plane frame (components along u, v, n).")
    polarization_state: PolarizationStateSummary | None = None
    first_order: FirstOrderTransfer | None = None
    first_order_issue: str | None = Field(None, description="Why first_order is absent.")
    ray_opl_defined: bool = Field(True, description="False after an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating: per-ray optical path lengths (wavefront data) are not modelled and beam.phase_rad is None.")
    relative_path_mm: float | None = Field(
        None, description="Only where ray_opl_defined is false: the sum of n x length with the elements that state no optical path counted as "
        "nothing. It is not an optical path length; only its difference between two beams that crossed the same such passes means "
        "something (the fringes of their coherent sum). The results readers do not report it.")
    coherence_group: str | None = Field(None, description="Beams of one coherence group are mutually coherent (default: one group per source).")
    coherence_length_mm: float | None = Field(None, description="The source's declared coherence length (None: undeclared).")
    group_path_length_mm: float | None = Field(
        None, description="The chief ray's group optical path from its source: the sum of n_g x length over the media it "
        "crossed (n_g = n - lambda dn/dlambda; equal to beam.optical_path_length_mm in media without dispersion), without "
        "the passes listed in group_delay_unknown_at. Two beams' degree of coherence follows the difference of their group "
        "paths, the delay of their envelopes. Omitted when a medium's group index is not available.")
    group_delay_dispersion_mm2: float | None = Field(
        None, description="The chief ray's group-delay dispersion from its source: the sum of k2 x length over the media it "
        "crossed (k2 = lambda^3 / (2 pi) d^2n/dlambda^2; divided by c^2 the group-delay dispersion in s^2). Omitted with "
        "group_path_length_mm.")
    group_delay_unknown_at: tuple[str, ...] = Field(
        (), description="The passes on the beam's way whose group delay the design does not state, as "
        "'<surface_id>:<event>:<side>@<angle of incidence, deg>' (side + or -: from the surface's positive or negative "
        "side; a thin lens also with the hit point '(u,v)' in mm, its thickness varying across it; a pass whose delay may "
        "differ between polarizations also with the beam's polarization state there, '[s1,s2,s3]': its normalized Stokes "
        "parameters along the pass's own axes -- s and p of a coating met obliquely, a retarder's fast and slow axes, a "
        "polarizer's transmission and blocking axes when it passes light along both): an element without a thickness (an "
        "ideal or tabulated thin lens, a splitter plate, a polarizer, a retarder, a Faraday rotator, a diffraction grating, an attenuator, an aperture that attenuates) "
        "and a coating without a "
        "stated constant phase, also where it is met beyond the critical angle ('coated_tir'). A pass through a coating "
        "that states its phase at the line (a phase table) ends with '#phase', and one whose data bound its group delay "
        "with that bound, '<=<bound>fs' (CoatingModel.group_delay_bound_fs), after it. Where a pass the two beams do not "
        "share has no stated phase at the line (no '#phase'), their phase difference is stated at the interfaces' own "
        "phases with an ESTIMATE entry (UNSTATED_PHASE_AS_INTERFACE), and has no value (PHASE_UNDEFINED) where such a pass "
        "states no optical path either (an ideal or tabulated thin lens, a Faraday rotator, a diffraction grating). Two "
        "beams' relative envelope delay, and with it their degree of coherence, is known when these agree (the same passes "
        "at the same angles, points and polarization states, in any order), and known within a bound when every pass that "
        "differs carries one (InterferenceResult.group_delay_bound_mm); otherwise the coherence metrics have no value. "
        "Omitted when there is none.")
    chief_ray_obstructed: bool = Field(
        False, description="The beam passed a central obstruction that its chief ray fell on: the beam is the annulus around it. "
        "beam.power_W is the annulus's power, the pupil rays are its geometric rays, and the chief ray is a reference axis "
        "(position, direction and path length along the axis) with no light on it in the geometric model, so beam.phase_rad is "
        "None and the Gaussian beam state is out of model scope (GAUSSIAN_CENTRAL_OBSTRUCTION).")
    power_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Surfaces on the beam's way behind which its power is out of model scope: a variable attenuator whose "
        "optical density varies across the beam by more than ND_FOOTPRINT_OD_TOLERANCE (ND_DENSITY_VARIES_ACROSS_BEAM), where "
        "beam.power_W is the chief ray's transmittance applied to the whole beam; an absorbing observation plane "
        "(observation/<id>) that part of the beam met (on its active area) while its chief ray did not cross it, so that part "
        "left the beam there while beam.power_W follows the chief ray; a physical surface that part of the beam met before "
        "this plane on the chief ray's straight leg to it, or that reflected part of the beam beyond the plane at the leg's "
        "start, so that part does not reach the plane with the beam; a physical surface that part of the beam met first on an "
        "earlier leg, beside the chief ray, where that part would have reached the surface the chief ray met inside its clear "
        "aperture (the beam still counts that light). Power-derived "
        "targets on the beam are OUT_OF_MODEL_SCOPE. Omitted when empty.")
    grating_orders: tuple[GratingOrderStep, ...] = Field(
        (), description="The diffraction grating orders on the beam's way, in order. Omitted when there is none.")
    power_upper_bound_W: float | None = Field(
        None, ge=0.0, description="Where beam.power_W is null: the most power this beam can carry (the power incident on the "
        "grating whose response does not give it). Omitted where the power is known.")
    surface_coincidence: SurfaceCoincidence | None = Field(
        None, description="The plane passes through a physical surface's chief-ray hit point: which surface, and whether this "
        "arrival is the beam arriving at it or the beam returning from it. Omitted elsewhere.")
    polarization_component: Literal[0, 1] | None = Field(
        None, description="A beam of a partially polarized source: which of its line's two mutually incoherent polarization "
        "components it is (0: the source's polarized state, (1 + P) / 2 of the line's power; 1: the orthogonal state, "
        "(1 - P) / 2; P the degree of polarization). The light on the beam's route is the sum of both components "
        "(ObservationResult.polarization_mixtures). Omitted for fully polarized light.")
    power_basis: Literal["GEOMETRIC_PUPIL_SAMPLES"] | None = Field(
        None, description="GEOMETRIC_PUPIL_SAMPLES: the beam has no valid Gaussian state and an edge met its pupil rays (a clear "
        "aperture upstream cut some or passed within a sample spacing of them, or the active area of this absorbing plane takes "
        "part of them; GEOMETRIC_POWER_BASIS), so beam.power_W is the incident power times the share of the equal-power pupil "
        "samples that passed -- and detected_power_W the share of those on the active area --, within "
        "power_sampling_resolution_W; the Gaussian beam metrics are not available, the ray results (spot, directions, paths) "
        "are. Omitted where beam.power_W is the Gaussian model's power.")
    power_sampling_resolution_W: float | None = Field(
        None, ge=0.0, description="With power_basis GEOMETRIC_PUPIL_SAMPLES: the sampling resolution of the power the plane "
        "receives from the beam (detected_power_W on an absorbing plane, else beam.power_W), the power of the pupil samples "
        "whose cells may straddle the edges (convergence evidence of the sampled value, not a proven bound). A power target is "
        "decided only where its limit lies outside the received power plus or minus it.")
    power_taken_before_at: tuple[str, ...] = Field(
        (), description="Surfaces that take a share of the beam before this plane on the chief ray's straight leg to it (a "
        "physical surface the beam meets next, an absorbing observation plane as observation/<id>, the reflecting surface the "
        "leg began at for the share it sends beyond the plane) or took one beside the chief ray on an earlier leg (a surface "
        "pupil rays or the beam's tail meet, an absorbing plane part of the beam crosses), whose share is bounded: the power crossing the plane lies "
        "between beam.power_W less power_taken_before_upper_W and beam.power_W less power_taken_before_lower_W (beam.power_W "
        "without it) (POWER_TAKEN_BEFORE_PLANE). Omitted when there is none.")
    power_taken_before_upper_W: float | None = Field(
        None, ge=0.0, description="The most power the surfaces of power_taken_before_at take from the beam before this plane "
        "together: the larger of the parallel-ray and the paraxial energy-flow bounds of each share; a share taken on an "
        "earlier leg counts with its bound in W there (later elements pass at most what arrives). Omitted with "
        "power_taken_before_at.")
    power_taken_before_lower_W: float | None = Field(
        None, ge=0.0, description="The least power those surfaces certainly take from the beam before this plane: the share "
        "of the beam's paraxial energy-flow lines that certainly meet the surface the beam meets next first, from a side "
        "that stops every light it meets (the back of an uncoated mirror, an absorber), before the plane. Omitted where none "
        "is certain.")
    reference_ray_promoted_at: tuple[str, ...] = Field(
        (), description="Surfaces where the beam's chief ray fell outside the clear aperture while some of its pupil rays "
        "passed (CLIPPED): the beam went on as those rays, and from the first of these surfaces its reference ray is the "
        "passing pupil ray nearest their centroid. uv_mm, aoi_deg, direction_uvn, beam.position, beam.direction and "
        "beam.optical_path_length_mm are that ray's; beam.power_W is the power that passed (the Gaussian power inside the "
        "opening where the beam was Gaussian there, else on the geometric pupil-sample basis, power_basis); the Gaussian beam "
        "state is not valid behind it and first_order is absent (first_order_issue). Omitted when there is none.")
    value_basis: ValueBasisField = None


class PolarizationMixture(BasisCarrier, OmitAbsentFields, StrictModel):
    """The partially polarized light of one source line on one route to an observation plane: the incoherent sum of the
    line's two polarization components (``ArrivalResult.polarization_component`` 0 and 1) that arrive along the same
    interactions (``ArrivalResult.lineage``) at the same crossing (``crossing_index``). ``path_ids`` lists the arrivals
    summed (component 0 first); a component that does not reach the plane on this route (an element stopped it, or what is
    left of it is below the trace's negligible level) adds nothing.

    ``stokes_W``: ``(S0, S1, S2, S3)`` in W, the sums of the components' Stokes parameters in the transverse basis of the
    first listed arrival (``BeamState.basis``); ``S0`` is the light's power. ``degree_of_polarization``:
    ``sqrt(S1^2 + S2^2 + S3^2) / S0``. ``polarization_state``: the normalized Stokes parameters ``S1..S3 / S0`` with the
    ellipse of the polarized part (its handedness, ellipticity and major axis) and the degree of polarization; absent where
    the polarized part is at most ``energy_conservation_tol`` of ``S0`` (unpolarized light has no ellipse).
    ``quantity_state``: ``KNOWN`` (every component's power and polarization on the route is known), ``KNOWN_ZERO`` (no
    power arrives on the route: the Stokes parameters are 0 and the degree of polarization is not defined), ``UNKNOWN``
    (``issue_codes`` and ``message`` say why; no Stokes value is given) or ``OUT_OF_MODEL_SCOPE`` (a component's power is out
    of model scope behind ``power_out_of_scope_at``: it is its chief ray's, not the light's, so no Stokes value, degree of
    polarization or ellipse is given). Two components have no interference term:
    two-beam quantities between them are not formed. ``value_basis``: what its Stokes values, degree of polarization and
    ellipse are where they are not simply the model's value (the results add a model-scope finding that reaches one of
    its components); omitted when there is none."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("stokes_W", "degree_of_polarization", "polarization_state", "issue_codes",
                                                 "message", "power_out_of_scope_at", "value_basis")

    source_id: str
    spectral_index: int
    wavelength_nm: float
    crossing_index: int = 0
    path_ids: tuple[str, ...]
    stokes_W: tuple[float, float, float, float] | None = None
    degree_of_polarization: float | None = Field(None, ge=0.0, le=1.0)
    polarization_state: PolarizationStateSummary | None = None
    quantity_state: QuantityState
    issue_codes: tuple[str, ...] = ()
    message: str | None = None
    power_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Surfaces behind which a summed component's power is out of model scope (ArrivalResult.power_out_of_scope_at): "
        "quantity_state OUT_OF_MODEL_SCOPE. Omitted when there is none.")
    value_basis: ValueBasisField = None


class ObservationPlaneFrame(StrictModel):
    """Where an observation plane is, resolved in the design frame (``origin_mm`` and the unit axes ``u``, ``v``, normal
    ``n = u x v``: the plane pose's local x, y, z). ``pose_frame`` says how the Project gave the pose: ``DESIGN`` or
    ``DEVICE_LOCAL`` (attached to ``device_id``: design pose = device pose composed with the plane pose).

    ``aperture`` is the active area in plane coordinates (u, v). ``absorbing_extent``: ``NONE`` (a transparent monitor:
    beams pass on), ``ACTIVE_AREA`` (absorbs exactly the beams that count as reaching its active area -- the chief ray is
    inside it or at least ``capture_min_power_fraction`` of the Gaussian power falls inside it; other beams pass the plane
    unchanged and are listed in ``ObservationResult.outside_active_area``. A chief ray does not cross such a plane beyond
    the scene's reach along it, ``max(laboratory_reach_mm, support)``, support being the farthest along the ray of the
    physical surfaces, source points and active areas: that beam is neither an arrival nor listed, and the leg's partial
    crossing counts any of its light on the area) or ``UNBOUNDED_PLANE`` (absorbing without an aperture: the whole infinite
    plane absorbs)."""

    origin_mm: Vector3
    u_axis: Vector3
    v_axis: Vector3
    normal: Vector3
    pose_frame: Literal["DESIGN", "DEVICE_LOCAL"]
    device_id: str | None = None
    aperture: Aperture | None = None
    absorbing: bool = False
    absorbing_extent: Literal["NONE", "ACTIVE_AREA", "UNBOUNDED_PLANE"] = "NONE"
    acceptance_half_angle_deg: float | None = None
    medium: str = "air"


class OutsideCrossing(OmitAbsentFields, StrictModel):
    """A beam that crossed the plane outside its active area (not a result of this plane; it passed on)."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("power_upper_bound_W", "power_out_of_scope_at")

    path_id: str
    crossing_index: int = 0
    uv_mm: Vector2
    miss_distance_mm: float = Field(description="Distance from the chief-ray hit to the active-area edge (positive outside).")
    power_W: float | None = Field(description="The beam's power; null: not known, at most power_upper_bound_W.")
    power_upper_bound_W: float | None = Field(None, ge=0.0, description="Where power_W is null: the most power this beam can carry. Omitted where the power is known.")
    power_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Surfaces behind which the beam's power is out of model scope (as ArrivalResult.power_out_of_scope_at): "
        "power_W is its chief ray's power, not the light's. Omitted when there is none.")


class CoherentSuperposition(BasisCarrier, OmitAbsentFields, StrictModel):
    """Coherent sum of every beam of one wavelength on an observation plane (any number of beams; see
    ``validation/superposition.py`` for the model): ``total_power_W`` = ``incoherent_power_W`` + ``interference_power_W``
    through the whole plane. Beams of different coherence groups add incoherently; within a group the mutual coherence is
    that of the Gaussian spectrum of the coherence length ``L_c`` after the two paths' relative group delay and group-delay
    dispersion (``InterferenceResult.coherence_factor``; ``exp(-(OPD / L_c)^2)`` in media without dispersion). ``None``
    values come with ``status`` / ``issue_codes`` (e.g. COHERENCE_UNDEFINED where a medium states no group index or a pulsed
    source states no coherence, OUT_OF_MODEL_SCOPE without valid Gaussian states); a line that states no coherence is taken
    as monochromatic, with a ``value_basis`` entry. Where two beams' relative group delay is known within the coatings'
    stated bounds (``InterferenceResult.group_delay_bound_mm``) and every coating whose pass differs between them states
    its phase, ``total_power_W`` is the sum at the traced delays and ``total_power_min_W`` / ``total_power_max_W`` the range
    the bounds leave. Where a differing pass states no phase or bounds no delay, ``total_power_W`` and
    ``interference_power_W`` are the sum at the trace's convention (default interface phases, no added delay;
    ``value_basis`` names the passes) and ``total_power_min_W`` / ``total_power_max_W`` the range over every phase and
    delay; behind an element that states no optical path they are None (status UNKNOWN, code PHASE_UNDEFINED). A sum, or
    an end of its range, within 1e-12 of the powers it adds (the cancellation residue of a dark port) is 0, with
    ``interference_power_W`` = -``incoherent_power_W``."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("total_power_min_W", "total_power_max_W", "phase_statement", "value_basis")

    wavelength_nm: float
    path_ids: tuple[str, ...]
    crossing_indices: tuple[int, ...] = ()
    incoherent_power_W: float
    total_power_W: float | None = None
    interference_power_W: float | None = None
    total_power_min_W: float | None = Field(
        None, description="The least total power the passes' group delays allow -- over every phase where a pass whose "
        "presence differs states no phase -- present with total_power_max_W.")
    total_power_max_W: float | None = Field(
        None, description="The largest total power the passes' group delays allow (over every phase where a pass states none).")
    phase_statement: str | None = Field(
        None, description="Where total_power_W has no value (a pass of unstated phase also states no optical path): which "
        "passes leave the phase open. Omitted otherwise.")
    status: Status
    issue_codes: tuple[str, ...] = ()
    value_basis: ValueBasisField = None


class SpectralPower(BasisCarrier, OmitAbsentFields, StrictModel):
    """The power a plane receives at one vacuum wavelength. Every wavelength a source launches is listed, whether or not
    light of it arrives."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("power_out_of_scope_at", "value_basis")

    wavelength_nm: float
    power_W: float | None = Field(
        description="Power the plane receives from the traced beams of this wavelength. 0.0 with complete true: a known zero (no "
        "light of this wavelength reaches the plane). null: no traced beam of it arrives and the plane's power at it is not "
        "known (complete false).")
    complete: bool = Field(
        True, description="True: all launched power of this wavelength was traced to a known end, none of the light the trace "
        "does not follow can reach the plane, and no omitted faint light may add, so power_W is the model's value. False: "
        "untraced_power_W of it was not followed (trace budget, interaction limit, a path ended out of model scope, a line not "
        "launched, light at the power floor, or an untraced reflection / coating transmission, light passing an aperture or an "
        "element its chief ray misses, a beam's tail beside the active area that reaches the plane, also beside its traced "
        "beams), or omitted faint light may add, so power_W counts the traced light only (null when none arrives) and the plane "
        "may receive more. False as well on a plane whose quantity_state is NOT_COMPUTED: the plane bounds did not run.")
    untraced_power_W: float | None = Field(
        0.0, ge=0.0, description="Power of this wavelength the traced value leaves out: launched power whose fate the trace does "
        "not follow, plus the power with which untraced light (followed by probe traces) arrives at the plane, also beside the "
        "traced beams of the same wavelength. null on a plane whose quantity_state is NOT_COMPUTED: the plane bounds did not "
        "run, so it was not computed.")
    power_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Surfaces behind which the power of a beam of this wavelength on the plane, or of light scattered out of "
        "one, is out of model scope (ArrivalResult.power_out_of_scope_at, SampledLightAtPlane.power_out_of_scope_at): power_W "
        "adds chief-ray powers, not the light's, and is not the model's value. Omitted when there is none.")
    value_basis: ValueBasisField = None


class ObservationResult(BasisCarrier, StrictModel):
    basis_state_field: ClassVar[str | None] = "quantity_state"
    basis_state_keys: ClassVar[tuple[str, ...]] = ("total_power_W",)

    observation_id: str
    status: Status = Field(
        description="Diagnostic severity of the beams arriving at the plane (acceptance, capture, beam-model validity) and, on "
        "a plane no traced beam reaches, of whether light may arrive. It does not say whether the plane's power is known: "
        "quantity_state says that and leads for every question about the power, so a lit plane may read PASS beside an "
        "UNKNOWN or BOUNDED power and OUT_OF_MODEL_SCOPE beside a KNOWN one. One term of the run's aggregate status; use it "
        "to sort and flag.")
    arrivals: tuple[ArrivalResult, ...] = ()
    total_power_W: float | None = Field(
        0.0, description="Power the plane receives from its arriving beams: their detected power on an absorbing plane "
        "(arrivals[].detected_power_W), the crossing beams' power on a transparent one. null: no traced beam arrives and the "
        "plane's power is not known (power_complete false).")
    message: str = ""
    plane: ObservationPlaneFrame | None = None
    power_by_wavelength: tuple[SpectralPower, ...] = Field(
        (), description="total_power_W per vacuum wavelength: every launched wavelength, 0.0 where none of its light arrives "
        "(a known zero), null where none arrives and its power is not known.")
    power_complete: bool = Field(
        True, description="True: every launched line was traced to a known end and none of the light the trace does not follow "
        "can reach the plane above its lines' negligible level, so total_power_W is the model's value (0.0 is a known zero); "
        "light left out below that level is the stated truncation of the value, and power_upper_bound_W adds the most it brings. "
        "False: light not followed may add; power_upper_bound_W records a supported bound; total_power_W counts the traced light "
        "only (null when none arrives). False as well where quantity_state is NOT_COMPUTED: the plane bounds did not run, "
        "untraced_power_W is null and the bounds are absent.")
    untraced_power_W: float | None = Field(
        0.0, ge=0.0, description="Power the traced value leaves out (all lines): launched power whose fate the trace does not "
        "follow, plus the power with which untraced light (followed by probe traces: ghost reflections, coating transmissions, "
        "light beside an element) arrives at this plane, also beside the traced beams of the same line. null where "
        "quantity_state is NOT_COMPUTED: the plane bounds did not run, so it was not computed (never 0.0 for that).")
    power_upper_bound_W: float | None = Field(
        None, ge=0, description="Supported upper bound of total received power: total_power_W plus untraced_power_W plus "
        "the most the light left out below the lines' negligible level brings (beside a KNOWN or KNOWN_ZERO power that light "
        "alone: the stated truncation of the value); on an "
        "absorbing plane at most the power its lines launch (each watt is absorbed once). Where total_power_W is a sampled "
        "value (power_sampling_resolution_W) no bound is taken from it: null, or on an absorbing plane the launched power; a "
        "power_W target there is decided only where its limit lies outside total_power_W plus or minus that resolution, which "
        "is convergence evidence, not a bound. Null: no supported finite bound. Absent in historical results. total_power_W "
        "retains traced contributions.")
    power_lower_bound_W: float | None = Field(
        None, ge=0, description="Where part of an arriving beam is taken before the plane by surfaces whose share is bounded "
        "(ArrivalResult.power_taken_before_upper_W): total_power_W less those bounds, the lower end of the plane's power, "
        "which lies between this and total_power_W (quantity_state BOUNDED). 0.0 on a plane no traced beam reaches whose only "
        "uncounted light probe traces followed to it under a supported bound: its power lies between 0 and "
        "power_upper_bound_W (quantity_state BOUNDED, total_power_W null). Omitted elsewhere.")
    power_sampling_resolution_W: float | None = Field(
        None, ge=0, description="Where the power of an arriving beam is on the geometric pupil-sample basis "
        "(arrivals[].power_basis GEOMETRIC_PUPIL_SAMPLES): the sum of those arrivals' power_sampling_resolution_W. "
        "total_power_W is then a sampled value: the model's within this resolution, which is convergence evidence and not a "
        "proven bound -- a power_W target on the plane is decided only where its limit lies outside total_power_W plus or "
        "minus it (within it: UNKNOWN), and power_upper_bound_W takes no bound from it. Omitted elsewhere.")
    absorbed_power_upper_bound_W: float | None = Field(
        None, ge=0, description="Where power_upper_bound_W is null only because light left out as negligible may cross this plane without a stated limit, and every other watt was followed "
        "to its end: the bound with that negligible light counted once, as a detector placed here would receive it. Absent otherwise, as where light of unknown fate may go on crossing the plane.")

    sampled_light: tuple[SampledLightAtPlane, ...] = Field(
        (), description="The rays of extended emitters crossing this plane, per spectral line, with the collection fraction and "
        "the crossings' columns. Their counted power is part of total_power_W and power_by_wavelength. Absent without such rays.")
    followed_light: tuple[FollowedLight, ...] = Field(
        (), description="The light the trace does not follow as beams that probe traces followed to this plane, one entry per "
        "contribution above its line's negligible level: the kind, where it left the traced beam and the most it brings. "
        "Their powers are the followed part of untraced_power_W. Absent where none reaches the plane and in results "
        "written before this field.")

    detection: DetectionResult | None = Field(
        None, description="Where the plane states a detector (Observation.detection): the photocurrent, noise currents and "
        "signal-to-noise ratio of the light it counts and, with a lock-in, the demodulated signal, its noise and SNR, each with "
        "its status (contracts/detection.py). Absent elsewhere.")

    @property
    def lit(self) -> bool:
        """Traced light is counted on this plane: an arriving beam, or a sampled ray."""

        return bool(self.arrivals) or any(line.counted_crossing_count for line in self.sampled_light)

    @property
    def power_lit(self) -> bool:
        """Traced light of a known power is counted on this plane (a beam whose power is not known is lit, and adds nothing to it)."""

        return any(a.beam.power_W is not None for a in self.arrivals) or any(line.counted_crossing_count for line in self.sampled_light)

    @model_serializer(mode="wrap")
    def _omit_historical_bound(self, handler):
        data = handler(self)
        if not self.sampled_light:
            data.pop("sampled_light", None)
        if not self.followed_light:
            data.pop("followed_light", None)
        if self.detection is None:
            data.pop("detection", None)
        if self.absorbed_power_upper_bound_W is None:
            data.pop("absorbed_power_upper_bound_W", None)
        if self.power_lower_bound_W is None:
            data.pop("power_lower_bound_W", None)
        if self.power_sampling_resolution_W is None:
            data.pop("power_sampling_resolution_W", None)
        if self.power_basis is None:
            data.pop("power_basis", None)
        if not self.polarization_mixtures:
            data.pop("polarization_mixtures", None)
        if self.value_basis is None:
            data.pop("value_basis", None)
        if "power_upper_bound_W" not in self.model_fields_set:
            data.pop("power_upper_bound_W", None)
        elif self.power_upper_bound_W is None:
            data["power_upper_bound_W"] = None
        return data

    outside_active_area: tuple[OutsideCrossing, ...] = ()
    superpositions: tuple[CoherentSuperposition, ...] = Field((), description="Coherent sum per wavelength where two or more beams arrive.")
    polarization_mixtures: tuple[PolarizationMixture, ...] = Field(
        (), description="The partially polarized light of each route of a partially polarized source on this plane: the sum "
        "of its two polarization components' Stokes parameters and its degree of polarization. Absent without such light.")
    power_basis: Literal["INCOHERENT_SUM"] | None = Field(
        None, description="INCOHERENT_SUM: total_power_W and a power_W target on this plane add the beams' powers, while "
        "superpositions states the coherent power of the same beams; the two differ by the interference term. Omitted where "
        "no coherent quantity exists on the plane.")
    quantity_state: QuantityState | None = Field(
        None,
        description="Whether the plane's power is a known scientific quantity: KNOWN (the model's value, zero or "
        "not; a sampled value within power_sampling_resolution_W where that is stated), KNOWN_ZERO (the model's value is exactly 0 and no light reaches the plane above its lines' negligible level), BOUNDED "
        "(a lit plane with a supported interval -- total_power_W up to power_upper_bound_W where light not followed may add, "
        "with an open upper end where power_upper_bound_W is null and value_basis names what leaves it open; "
        "power_lower_bound_W to total_power_W where part of a beam is taken before the "
        "plane by a bounded share) or UNKNOWN "
        "(part of the launched power was not followed, or light the trace followed may reach a plane no traced beam "
        "reaches: untraced_power_W and any supported power_upper_bound_W stand beside it and do not change this state; a "
        "power target on such a plane is graded on the interval from 0 to the bound of the followed light) or "
        "OUT_OF_MODEL_SCOPE (the power of a beam here, or of the light a surface scattered out of one, is out of model scope: "
        "arrivals[].power_out_of_scope_at, sampled_light[].power_out_of_scope_at) or NOT_COMPUTED (the evaluation did not "
        "run the plane bounds: total_power_W is the traced beams' power, and whether light the trace does not follow reaches "
        "the plane was not computed; ValidationResult.evaluation names the scope). Read it, "
        "not total_power_W alone: a missing value and a known zero are different facts. None only on a result stored "
        "before this field existed.",
    )
    value_basis: ValueBasisField = None


# ------------------------------------------------------------------------------------------- sampled light

#: pi, the projected solid angle of a hemisphere in steradians.
PI_SR = 3.141592653589793


class SampledCrossings(StrictModel):
    """Every crossing of an observation plane by the sampled rays of one source line, as columns of equal length (one row per
    crossing; a ray that crosses a plane twice has two rows). The rows hold everything an irradiance or uniformity
    calculation needs: bin ``power_W`` by ``(u_mm, v_mm)`` and divide by the bin area."""

    sample_index: tuple[int, ...] = Field(description="Which sample of the line's sequence the ray comes from (0-based; a branch keeps its sample's index).")
    u_mm: tuple[float, ...] = Field(description="Crossing point along the plane's u axis, in mm.")
    v_mm: tuple[float, ...] = Field(description="Crossing point along the plane's v axis, in mm.")
    direction_u: tuple[float, ...] = Field(description="Direction cosine of the ray on the plane's u axis.")
    direction_v: tuple[float, ...] = Field(description="Direction cosine of the ray on the plane's v axis.")
    direction_n: tuple[float, ...] = Field(description="Direction cosine of the ray on the plane's normal.")
    power_W: tuple[float, ...] = Field(description="Power the ray carries at the crossing, in W.")
    counted: tuple[bool, ...] = Field(description="The ray lands on the active area (the whole plane without one) inside the acceptance cone.")


class FollowedLight(OmitAbsentFields, StrictModel):
    """One contribution of light the trace does not follow as beams that a probe trace followed to an observation plane
    (``ObservationResult.followed_light``): a ghost reflection, a coating's transmission, the part of a clipped beam inside
    an opening, light beside an element, a traced beam's tail on the active area, part of a traced beam crossing the plane
    where its chief ray does not. Its power is part of the plane's ``untraced_power_W``, never of ``total_power_W``."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("surface_id", "path_id")

    kind: str = Field(description="What the light is: UNTRACED_REFLECTION, UNTRACED_TRANSMISSION, CLIPPED, "
                      "APERTURE_TRUNCATION (light a probe trace followed from where it left a traced beam), ACTIVE_AREA_TAIL (a "
                      "traced beam's tail on the active area it crosses beside), PARTIAL_CROSSING or DIFFRACTED_BEAM (part of a "
                      "traced beam on a leg whose chief ray does not cross the plane).")
    source_id: str
    spectral_index: int = Field(ge=0)
    surface_id: str | None = Field(None, description="The surface where the light left the traced beam (a probe's start). "
                                   "Omitted for a traced beam's own tail or partial crossing.")
    path_id: str | None = Field(None, description="The traced path the light belongs to or left.")
    power_W: float = Field(ge=0.0, description="The most this contribution brings to the plane: a probe goes on with a "
                           "beam's whole power where the share that passes an edge is not computed, and a tail or a partial "
                           "crossing is a bound of the part of the beam on the plane.")
    reach: Literal["ARRIVAL", "ACTIVE_AREA_TAIL", "PARTIAL_CROSSING"] = Field(
        description="How it meets the plane: ARRIVAL (the probe's chief ray crosses the plane and the crossing counts), "
        "ACTIVE_AREA_TAIL (it crosses beside the active area, its tail on the area), PARTIAL_CROSSING (part of the beam "
        "crosses the plane where the chief ray does not).")


class SampledLightAtPlane(BasisCarrier, OmitAbsentFields, StrictModel):
    """The light of one spectral line of an extended emitter at an observation plane, or of the light one surface scattered
    out of one beam (``scatter_surface_id``, ``parent_path_id``). Rays have no phase relation: powers add, and no coherent
    quantity exists for them. Every sampled line of the run has an entry on every plane: a line none of whose rays is counted
    there reads ``power_W`` 0.0 with ``counted_crossing_count`` 0, a sampled zero whose resolution is one ray's weight
    (``emitted_W / sample_count``, in the plane's ``power_sampling_resolution_W``). ``value_basis``: what its power and
    collection fractions are where they are not simply the model's value (the results add a model-scope finding that
    reaches the line); omitted when there is none."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("scatter_surface_id", "parent_path_id", "power_out_of_scope_at", "value_basis")

    source_id: str
    spectral_index: int
    wavelength_nm: float
    emitted_W: float = Field(description="The line's total radiant power, in W: the denominator of collection_fraction.")
    sample_count: int = Field(description="Rays the line declares.")
    samples_traced: int = Field(description="Rays launched; fewer than sample_count only when the ray-step budget was reached (MAX_PATHS). Fractions stay relative to emitted_W.")
    power_W: float = Field(description="Sum of the counted crossings' powers, in W.")
    collection_fraction: float = Field(description="power_W / emitted_W from all the rays.")
    half_sample_count: int = Field(description="Rays in the first half of the sequence: sample_count / 2.")
    collection_fraction_half_samples: float = Field(description="The same fraction from the first half of the rays alone.")
    collection_fraction_difference: float = Field(description="|collection_fraction - collection_fraction_half_samples|: how much the estimate moved when the sample count doubled.")
    collection_fraction_sigma: float = Field(description="Standard error of collection_fraction if the rays were independent random samples: the spread of each ray's contribution over sqrt(rays). The stratified rays usually do better, so it is an upper guide; the difference above may be small by chance.")
    crossing_count: int = Field(description="Rows in crossings.")
    counted_crossing_count: int = Field(description="Rows of crossings that count.")
    crossings: SampledCrossings
    scatter_surface_id: str | None = Field(
        None, description="Scattered light: the surface that scattered it (emitted_W is the power it scattered out of the beam "
        "parent_path_id). Absent for an extended emitter's light.")
    parent_path_id: str | None = Field(None, description="Scattered light: the beam path whose light the surface scattered.")
    power_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Scattered light out of a beam whose power is out of model scope behind these surfaces "
        "(ArrivalResult.power_out_of_scope_at of that beam): emitted_W and the crossings' powers follow its chief ray's power, not "
        "the light's, so the plane's power is OUT_OF_MODEL_SCOPE. Omitted when there is none.")
    value_basis: ValueBasisField = None


class SampledSourceLine(OmitAbsentFields, StrictModel):
    """One spectral line of an extended emitter or a sampled source (an intensity map, a sampled field, a speckle field:
    ``pattern``), or the light one surface scattered out of one beam (``scatter_surface_id``: a POINT source at the beam's
    hit point, of the beam's source line): what it emits and how it was sampled."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("scatter_surface_id", "parent_path_id", "scatter_side", "scattered_fraction",
                                                 "incident_power_W", "exponent", "beam_radius_mm", "pattern", "cone_half_angle_deg",
                                                 "exitance_mean_W_per_mm2", "exitance_peak_W_per_mm2", "realizations",
                                                 "power_out_of_scope_at")

    source_id: str
    spectral_index: int
    wavelength_nm: float
    emitted_W: float = Field(description="Total radiant power into the angular law's domain, in W (the declared, authoritative quantity).")
    shape: Literal["RECTANGLE", "DISC", "POINT"]
    area_mm2: float = Field(description="Emitting area, in mm^2 (0 for a POINT).")
    angular_law: Literal["LAMBERTIAN", "ISOTROPIC", "COSINE_POWER", "UNIFORM_CONE", "COLLIMATED", "FIELD_WAVEVECTOR"] = Field(
        description="LAMBERTIAN, ISOTROPIC, COSINE_POWER (intensity cos^n over the hemisphere), UNIFORM_CONE (uniform radiance "
        "within cone_half_angle_deg), COLLIMATED (along the emitter's +z), FIELD_WAVEVECTOR (a sampled or speckle field: each ray "
        "along the field's local wavevector).")
    solid_angle_measure: Literal["PROJECTED_SOLID_ANGLE", "SOLID_ANGLE"] = Field(
        description="The measure the rays are uniform in: cos(theta) dOmega over the hemisphere around the emitter's +z (LAMBERTIAN) or "
        "within its cone (UNIFORM_CONE), or dOmega over the sphere (ISOTROPIC) or the hemisphere (COSINE_POWER, weighted by its law); "
        "scattered light: SOLID_ANGLE over the hemisphere around the surface normal, the rays weighted by its law.")
    domain_sr: float | None = Field(
        description="Size of the sampling domain in that measure: pi (LAMBERTIAN), 4 pi (ISOTROPIC), 2 pi (COSINE_POWER, scattered "
        "light), pi sin^2 of the half angle (UNIFORM_CONE), 0 (COLLIMATED), in sr; null for FIELD_WAVEVECTOR (the field's phase "
        "sets each ray's direction).")
    radiance_W_per_m2_sr: float | None = Field(
        description="Derived: emitted_W / (pi area) (LAMBERTIAN) or emitted_W / (pi sin^2(half angle) area) (UNIFORM_CONE) of a "
        "uniform emitter, in W/(m^2 sr); with an intensity map the radiance follows the map (exitance_peak_W_per_mm2) and this is "
        "null; null for a POINT and for the other laws.")
    intensity_W_per_sr: float | None = Field(
        description="Derived: emitted_W / (4 pi), ISOTROPIC; emitted_W (n + 1) / (2 pi) along the axis, COSINE_POWER and scattered "
        "light (n = 1: Lambertian); in W/sr; null for the other laws.")
    sample_count: int
    samples_traced: int
    seed: int
    sequence: str = Field(description="The sampling sequence: identifies how sample_count and seed give the rays.")
    ray_weight_W: float = Field(description="Power each ray starts with: emitted_W / sample_count, in W.")
    scatter_surface_id: str | None = Field(None, description="Scattered light: the surface that scattered it.")
    parent_path_id: str | None = Field(None, description="Scattered light: the beam path whose light it is.")
    scatter_side: Literal["REFLECT", "TRANSMIT"] | None = Field(None, description="Scattered light: the side of the surface it leaves on.")
    scattered_fraction: float | None = Field(None, description="Scattered light: the surface's scattered share (emitted_W / incident_power_W).")
    incident_power_W: float | None = Field(None, description="Scattered light: the beam's power meeting the surface, in W.")
    exponent: float | None = Field(None, description="Scattered light with a COSINE_POWER law: its exponent n (intensity proportional to cos^n).")
    beam_radius_mm: float | None = Field(
        None, description="Scattered light: the beam's largest 1/e^2 radius at the surface, in mm (absent where it is not known). The "
        "scattered light leaves from the beam's hit point, a description that holds where this is small against the distances to "
        "what the light reaches.")
    pattern: Literal["INTENSITY_MAP", "SAMPLED_FIELD", "SPECKLE"] | None = Field(
        None, description="A sampled source: an intensity map (incoherent, unpolarized rays distributed as the map), a sampled "
        "complex field or a speckle field (rays distributed as |U|^2, along the field's local wavevector, with its polarization: "
        "the run adds their powers, the geometric transport of the field). Absent: uniform over the shape.")
    cone_half_angle_deg: float | None = Field(None, description="UNIFORM_CONE: the cone's half angle in degrees.")
    exitance_mean_W_per_mm2: float | None = Field(
        None, description="A sampled source: emitted_W over its emitting area (the grid's), in W/mm^2.")
    exitance_peak_W_per_mm2: float | None = Field(
        None, description="A sampled source: the largest emitted power per area at a grid point (the map's, or |U|^2 of the field), in W/mm^2.")
    realizations: int | None = Field(
        None, description="A speckle field: its independent screens; ray k leaves through realization k mod realizations.")
    power_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Scattered light out of a beam whose power is out of model scope behind these surfaces "
        "(ArrivalResult.power_out_of_scope_at of that beam): incident_power_W, emitted_W, ray_weight_W and intensity_W_per_sr "
        "follow its chief ray's power, not the light's. Omitted when there is none.")


# ---------------------------------------------------------------------------------------- interference


class InterferenceResult(BasisCarrier, OmitAbsentFields, StrictModel):
    """Analytic two-beam interference on one observation plane (plane-wave phases with paraxial Gaussian envelopes)."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("group_delay_bound_mm", "coherence_factor_min", "coherence_factor_max",
                                                 "combined_power_collinear_min_W", "combined_power_collinear_max_W", "phase_statement",
                                                 "counter_propagating", "value_basis")

    observation_id: str
    source_id_a: str
    source_id_b: str
    path_id_a: str
    path_id_b: str
    crossing_index_a: int = 0
    crossing_index_b: int = 0
    wavelength_nm: float
    status: Status
    beam_crossing_angle_deg: float | None = None
    half_angle_deg: float | None = None
    fringe_period_nm: float | None = Field(None, description="In-plane period 2*pi/|dk_parallel|.")
    fringe_spacing_3d_nm: float | None = Field(None, description="lambda/(2 n sin(alpha/2)) perpendicular to the fringe planes.")
    fringe_direction: Vector3 | None = Field(None, description="Unit in-plane direction along dk_parallel (fringe modulation direction).")
    optical_path_length_mm_a: float | None = Field(None, description="None where that beam's ray_opl_defined is false.")
    optical_path_length_mm_b: float | None = Field(None, description="None where that beam's ray_opl_defined is false.")
    optical_path_difference_mm: float | None = Field(
        None, description="None (issue PHASE_UNDEFINED) where a pass one beam makes and the other does not states no optical path: an ideal or "
        "tabulated thin lens, a Faraday rotator or a diffraction grating; passes both beams make alike cancel.")
    phase_difference_rad: float | None = Field(
        None, description="phi_a - phi_b of the two fields at the plane, wrapped to (-pi, pi] in steps of 1e-6 rad. Where a pass one beam makes and "
        "the other does not has no stated phase at this wavelength (an element without a thickness, a coating that states no "
        "phase there: ArrivalResult.group_delay_unknown_at without '#phase'), the value at the trace's default interface "
        "phases, value_basis naming the passes (UNSTATED_PHASE_AS_INTERFACE); None (issue PHASE_UNDEFINED) where such a pass "
        "also states no optical path. Passes both beams make alike cancel. Where the two polarizations are orthogonal "
        "(polarization_overlap at most energy_conservation_tol) the overlap has no phase: k0 (OPL_a - OPL_b) wrapped, "
        "value_basis ESTIMATE naming polarization_overlap.")
    coherence_length_mm: float | None = None
    group_path_difference_mm: float | None = Field(
        None, description="The difference of the two chief rays' group optical paths (sum of n_g x length from each source; "
        "the optical path difference in media without dispersion): the relative delay of the two fields' envelopes, which "
        "sets their degree of coherence. None when a group index is not available. Where the beams crossed different passes "
        "whose group delay the design does not state (ArrivalResult.group_delay_unknown_at), the traced difference without "
        "them: where their data bound it, the relative delay lies within group_delay_bound_mm of it; where not, value_basis "
        "names the passes taken as adding no delay (UNSTATED_GROUP_DELAY_ZERO), and so do the coherence quantities read from it.")
    group_delay_bound_mm: float | None = Field(
        None, description="Where the beams meet coatings differently whose data bound their group delay "
        "(CoatingModel.group_delay_bound_fs): the sum of the bounds of the passes that differ between them, as a group path "
        "length (c x delay). The beams' relative group delay lies within this of group_path_difference_mm, and the degree of "
        "coherence within coherence_factor_min / coherence_factor_max (optics/interference/coherence.py). Omitted when no "
        "such pass differs.")
    coherence_factor: float | None = Field(
        None, description="|gamma| of the Gaussian spectrum of the coherence length Lc after the relative group delay D and "
        "group-delay dispersion B of the two paths: (1 + b^2)^(-1/4) exp(-(D / Lc)^2 / (1 + b^2)) with b = 2 B / Lc^2 "
        "(exp(-(D / Lc)^2) without dispersion). With group_delay_bound_mm, the value at the traced delay; the degree of "
        "coherence lies between coherence_factor_min and coherence_factor_max, and the fringe visibilities and the peak "
        "irradiance, stated at coherence_factor, change linearly with it. Where an oblique coating of unstated phase meets a "
        "beam in a state mixing s and p, the visibilities are at the trace's default phases there (value_basis, "
        "UNSTATED_PHASE_AS_INTERFACE).")
    coherence_factor_min: float | None = Field(
        None, description="With group_delay_bound_mm: the least degree of coherence the coatings' group-delay bounds allow. "
        "Omitted where one line states no coherence length: its own length may be shorter, so no lower end above 0 is proven.")
    coherence_factor_max: float | None = Field(
        None, description="With group_delay_bound_mm: the largest degree of coherence the coatings' group-delay bounds allow. "
        "Where one line states no coherence length beside one that does: the largest degree of coherence any length of that "
        "line up to the stated one allows (1 behind bounded coating delays; above the stated length's value where the two "
        "paths' group-delay dispersions differ), stated where it exceeds the value; the values then carry value_basis "
        "ESTIMATE (FULL_COHERENCE, and UNSTATED_GROUP_DELAY_ZERO at the bounded passes).")
    coherence_margin: float | None = Field(None, description="1 - |D|/Lc, D the group path difference.")
    spatial_overlap_ratio: float | None = None
    overlap_effective_area_mm2: float | None = None
    polarization_overlap: float | None = None
    intensity_balance: float | None = None
    fringe_visibility: float | None = Field(None, description="Power-weighted (integrated) visibility.")
    fringe_visibility_peak: float | None = Field(None, description="Local visibility at the envelope-product maximum.")
    mean_irradiance_W_per_mm2: float | None = None
    peak_irradiance_W_per_mm2: float | None = None
    peak_position_uv_mm: Vector2 | None = None
    incoherent_power_sum_W: float | None = Field(None, description="P_a + P_b.")
    combined_power_collinear_W: float | None = Field(
        None, description="Coherent power of two (nearly) collinear beams: Pa + Pb + 2 sqrt(Pa Pb) eta_pol eta_sp Re[gamma exp(i phi)] "
        "for beams crossing the plane in the same direction (gamma the complex degree of coherence: complex where the paths' "
        "group-delay dispersions differ, its magnitude coherence_factor); Pa + Pb for beams crossing it in opposite directions. Pa, Pb: the "
        "power the plane receives from each beam (on an absorbing plane its detected power). Where the beams meet passes "
        "differently that state no phase, the value at the trace's default phases (value_basis) beside its range "
        "combined_power_collinear_min_W / _max_W; None where such a pass also states no optical path."
    )
    combined_power_collinear_min_W: float | None = Field(
        None, description="Where phi is open, or where one source states a coherence length and the other none while the "
        "paths' group-delay dispersions or bounded coating delays differ (the phase of gamma then moves with the unstated "
        "length): the least combined power of the two (nearly) collinear beams over every phase, with the largest degree of "
        "coherence the passes' group delays and the unstated length allow (coherence_factor_max; 1 where a pass bounds none) "
        "and the largest polarization overlap the unstated phases allow. Present with combined_power_collinear_max_W.")
    combined_power_collinear_max_W: float | None = Field(
        None, description="The largest combined power of the two (nearly) collinear beams over every phase, where "
        "combined_power_collinear_min_W is stated.")
    counter_propagating: bool | None = Field(
        None, description="True where the two beams cross the plane in opposite directions (a beam and its reflection back): "
        "the power through the plane has no cross term, and combined_power_collinear_W is Pa + Pb whatever their coherence. "
        "Omitted otherwise.")
    phase_statement: str | None = Field(
        None, description="Where phi has no value (a pass of unstated phase also states no optical path): which passes leave it "
        "open. Omitted otherwise (the passes are in value_basis).")
    issue_codes: tuple[str, ...] = ()
    value_basis: ValueBasisField = None


# ------------------------------------------------------------------------------------- grating orders


class IncoherentPairs(StrictModel):
    """The beam pairs on one observation plane whose two beams have the same wavelength and come from mutually
    incoherent sources (different coherence groups): they form no lasting fringe and their irradiances add. They are
    counted here and not listed one by one in ``interference``, except a pair a pair-metric target selects (listed and
    analysed in full) and a pair whose beams arrive in media of different index (``MEDIUM_MISMATCH``)."""

    observation_id: str
    pair_count: int = Field(ge=1, description="Same-wavelength beam pairs of mutually incoherent sources counted here.")
    coherence_group_count: int = Field(ge=2, description="Coherence groups of the beams reaching this plane.")


class GratingOrderResult(OmitAbsentFields, StrictModel):
    """One diffraction order of one beam's pass through a grating: every order followed and, where at most 129 orders
    propagate, every other order that propagates, one entry each. ``order_path_id``: the path that carries it (absent: not followed, or nonpropagating).
    ``propagation``: ``NONPROPAGATING`` where its tangential wavevector exceeds the exit medium's (it has no direction and
    is not traced).

    ``direction`` (design frame) and ``exit_angle_deg`` (signed in the plane of the grating vector and the normal, positive
    toward the grating vector) follow from the grating equation and never depend on the response. ``power_state``: ``KNOWN``
    (``power_W``, from the response), ``BOUNDED`` (``power_W`` null, at most ``power_upper_bound_W``) or ``OUT_OF_MODEL_SCOPE``
    (the incident beam's power is out of model scope behind ``power_out_of_scope_at``: ``power_W`` and the bound are null,
    ``efficiency`` stays the response's share);
    ``polarization_state``: ``KNOWN`` or ``UNKNOWN``; ``phase_state``: ``UNKNOWN``: efficiencies state no phase. A nonpropagating
    order's states are ``NOT_APPLICABLE``. ``efficiency``: the share of the incident power the response gives it;
    ``relative_response``: the value of a relative table, which is not a share of the incident power. ``response_basis``:
    ``NONE`` or the table's ``data_kind``: ``MEASURED``, ``INDEPENDENTLY_COMPUTED``, ``ASSUMED``."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("order_path_id", "direction", "exit_angle_deg", "power_W", "power_upper_bound_W", "efficiency",
                                                 "relative_response", "power_out_of_scope_at")

    surface_id: str
    element_id: str
    path_id: str
    source_id: str
    spectral_index: int
    wavelength_nm: float
    incidence_deg: float = Field(description="The chief ray's signed angle of incidence in the plane of the grating vector and the normal.")
    conical_angle_deg: float = Field(description="How far the chief ray arrives from the plane perpendicular to the grooves.")
    order: int
    propagation: Literal["PROPAGATING", "NONPROPAGATING"]
    followed: bool
    order_path_id: str | None = None
    direction: Vector3 | None = None
    exit_angle_deg: float | None = None
    power_state: QuantityState
    power_W: float | None = None
    power_upper_bound_W: float | None = None
    power_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Surfaces behind which the incident beam's power is out of model scope (ArrivalResult.power_out_of_scope_at "
        "of that beam): the beam's power is its chief ray's, not the light's, so power_state is OUT_OF_MODEL_SCOPE and no power "
        "or bound is stated. Omitted when there is none.")
    efficiency: float | None = None
    relative_response: float | None = None
    polarization_state: QuantityState
    phase_state: QuantityState
    response_basis: Literal["NONE", "MEASURED", "INDEPENDENTLY_COMPUTED", "ASSUMED"]


class NonlinearConversionResult(OmitAbsentFields, StrictModel):
    """One beam's pass into a crystal that generates its second harmonic (``SecondHarmonicGeneration``): the undepleted-pump
    conversion of the Gaussian beam that enters through ``surface_id`` (``optics/nonlinear/shg.py`` states the model).

    ``status``: ``PASS`` (the conversion is inside the model: ``harmonic_power_W`` and ``conversion_efficiency`` are its
    values) or ``OUT_OF_MODEL_SCOPE`` (``reason`` says which condition fails: the undepleted-pump estimate, the near-field
    limit, a fundamental that is not one coherent polarized continuous-wave TEM00 line, a line wider than the phase matching
    allows, a harmonic already travelling with the light, a beam without a valid Gaussian state, ...; the
    harmonic quantities are then absent, and where a harmonic is generated its power and the fundamental's power behind the
    crystal are out of model scope). ``UNKNOWN`` means bulk absorption is not established for the entering pump or for the
    conversion in the crystal. ``fundamental_power_W``: the fundamental power entering the crystal (every wave), if known.
    ``converting_power_W``: of it, the power in the waves the interaction converts, if known. ``harmonic_power_W``: the harmonic power
    generated inside the crystal, at its exit face (after the crystal's absorption, if any); ``conversion_efficiency`` = ``harmonic_power_W`` /
    ``fundamental_power_W``. ``phase_mismatch_rad`` = ``dk L / 2`` (``delta_k_per_mm`` the mismatch of the waves' normal
    components across the slab of thickness ``length_mm``, including a quasi-phase-matching grating);
    ``plane_wave_factor`` = ``sinc^2(dk L / 2)``; ``overlap_factor`` the Gaussian-beam ``H`` (``shg.py``);
    ``focusing_parameter`` ``L / b`` with ``b`` the beam's confocal parameter in the crystal (the smaller of its two axes').
    ``walkoff_mrad``: the angle between the harmonic's energy flow and the fundamental's (mean of the two waves).
    ``depletion_estimate``: the relative amount by which the undepleted result exceeds a depleted one;
    ``near_field_error``: the difference between the near-field and the focused result relative to the larger of the focused
    result and the focused result at ``dk = 0`` (near-field model only); where it is within ``nonlinear_focusing_rel_tol`` while
    the near-field result differs from the focused one by more than that fraction of the focused one (near a zero of the
    tuning curve, which the focusing fills), ``overlap_factor`` and the powers are the focused result's;
    ``quadrature_error``: the relative quadrature error of ``overlap_factor``. Acceptance bandwidths are the full widths at
    half maximum of ``sinc^2(dk L / 2)`` about the traced beam: ``angular_acceptance_mrad`` inside the crystal, in the plane of
    the wave normal and the optic axis (absent where ``dk`` changes with neither the first nor the second power of the
    angle), ``temperature_acceptance_K`` (absent without stated thermo-optic coefficients), ``spectral_acceptance_nm`` of the
    fundamental's vacuum wavelength."""

    absent_omitted: ClassVar[tuple[str, ...]] = (
        "harmonic_path_id", "fundamental_power_W", "converting_power_W", "harmonic_power_W", "conversion_efficiency", "reason",
        "d_eff_pm_per_V", "delta_k_per_mm",
        "phase_mismatch_rad", "plane_wave_factor", "overlap_factor", "focusing_parameter", "walkoff_mrad", "depletion_estimate",
        "near_field_error", "quadrature_error", "angular_acceptance_mrad", "temperature_acceptance_K", "spectral_acceptance_nm",
        "temperature_c", "qpm_period_um", "path_length_mm")

    element_id: str
    surface_id: str
    path_id: str
    source_id: str
    spectral_index: int
    wavelength_nm: float
    harmonic_spectral_index: int
    harmonic_wavelength_nm: float
    harmonic_path_id: str | None = None
    fundamental_waves: Literal["oo", "ee", "oe"]
    harmonic_wave: Literal["o", "e"]
    focusing: Literal["near_field", "boyd_kleinman"]
    status: Status
    reason: str | None = None
    fundamental_power_W: float | None = None
    converting_power_W: float | None = None
    harmonic_power_W: float | None = None
    conversion_efficiency: float | None = None
    length_mm: float
    path_length_mm: float | None = None
    d_eff_pm_per_V: float | None = None
    delta_k_per_mm: float | None = None
    phase_mismatch_rad: float | None = None
    plane_wave_factor: float | None = None
    overlap_factor: float | None = None
    focusing_parameter: float | None = None
    walkoff_mrad: float | None = None
    depletion_estimate: float | None = None
    near_field_error: float | None = None
    quadrature_error: float | None = None
    angular_acceptance_mrad: float | None = None
    temperature_acceptance_K: float | None = None
    spectral_acceptance_nm: float | None = None
    temperature_c: float | None = None
    qpm_period_um: float | None = None


# --------------------------------------------------------------------------------------- power budget


class PowerSinkEntry(StrictModel):
    """Power leaving the traced beams at one place. ``kind`` (see ``PowerBudget``); ``surface_id`` the scene surface
    (``<component>/<surface>`` or ``observation/<id>``; ``None``: free space or the source); ``element_id`` the component that
    owns it; ``detail`` the Engine event, termination or medium."""

    kind: str
    power_W: float
    source_id: str
    spectral_index: int
    wavelength_nm: float
    path_id: str | None = None
    surface_id: str | None = None
    element_id: str | None = None
    detail: str = ""


class PowerBudgetLine(OmitAbsentFields, StrictModel):
    """Power accounting of one source spectral line: ``input_power_W`` = sum of ``sinks`` + ``residual_W``. A line a crystal
    generates (a second harmonic, ``NonlinearConversionResult``) names the line it is generated from (``generated_from``,
    ``<source>#<index>``): its input is the harmonic power generated, which that line books as ``NONLINEAR_CONVERSION``."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("generated_from", "observed_out_of_scope_at")

    source_id: str
    spectral_index: int
    wavelength_nm: float
    input_power_W: float
    sinks: dict[str, float] = Field(default_factory=dict)
    observed_W: dict[str, float] = Field(default_factory=dict, description="Power counted at each observation plane "
                                         "(non-absorbing monitors observe without consuming power; not a sink). A plane where "
                                         "the power of this line's light is out of model scope is not listed here "
                                         "(observed_out_of_scope_at).")
    observed_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Observation planes where the power of a beam of this line, or of light scattered out of one, is out of "
        "model scope (ArrivalResult.power_out_of_scope_at): the power counted there would be chief-ray powers, not the light's, "
        "so observed_W states none for them. Omitted when there is none.")
    residual_W: float = 0.0
    conserved: bool = True
    generated_from: str | None = None


class ElementPowerLine(StrictModel):
    element_id: str
    sinks: dict[str, float] = Field(default_factory=dict)
    total_loss_W: float = 0.0


class PowerBudget(StrictModel):
    """Where the traced power went (chief-ray packet power model: one power per beam path, Gaussian truncation at apertures).

    Sink kinds: ``DETECTED`` (the part of a beam on an absorbing observation plane's active area), ``OUTSIDE_ACTIVE_AREA``
    (the rest of a beam an absorbing plane stops: its power beside the active area; its fate is not followed),
    ``OUTSIDE_ACCEPTANCE`` (a beam an absorbing plane stops on its active area outside its acceptance cone: not detected),
    ``ABSORBED`` (absorber), ``ESCAPED`` (left the scene), ``CLIPPED`` (chief ray on an opaque rim: the whole beam; or the
    part of a beam a central obstruction blocks), ``APERTURE_TRUNCATION`` (Gaussian power outside a clear aperture),
    ``UNTRACED_REFLECTION`` (reflection at a transmitting interface that is not followed: ghosts), ``UNTRACED_TRANSMISSION``
    (light a reflector's coating transmits), ``COATING_ABSORPTION`` (1 - R - T), ``BULK_ABSORPTION``, ``POLARIZER_REJECTION``,
    ``ELEMENT_TRANSMISSION_LOSS``, ``NONLINEAR_CONVERSION`` (power a crystal converts into a generated second harmonic, the input
    of that harmonic's own line), ``BLOCKED_BACK_SIDE``, ``TERMINATED_OUT_OF_MODEL`` (grazing incidence, medium mismatch,
    material / coating data out of range), ``PATH_LIMIT`` (power the run's trace budget left untraced) / ``INTERACTION_LIMIT`` (trace limits), ``BELOW_POWER_FLOOR`` and
    ``SOURCE_NOT_TRACED``. Every sink is power that left the modelled beams; ``UNTRACED_*`` and ``TERMINATED_OUT_OF_MODEL``
    say how much power the model does not follow. ``PATH_LIMIT``, ``INTERACTION_LIMIT``, ``SOURCE_NOT_TRACED`` and
    ``TERMINATED_OUT_OF_MODEL`` are power of unknown fate: it may still reach any plane, so a plane no traced beam reaches is
    known to be dark only when the lines in question have none (``ObservationResult.power_complete``). ``residual_W`` = input - sum of sinks (numerical; ``conserved`` within
    ``tolerance_W``). Scattering, diffraction losses and multiple ghost interference are not modelled.
    """

    lines: tuple[PowerBudgetLine, ...] = ()
    by_element: tuple[ElementPowerLine, ...] = ()
    entries: tuple[PowerSinkEntry, ...] = ()
    total_input_W: float = 0.0
    total_sinks: dict[str, float] = Field(default_factory=dict)
    residual_W: float = 0.0
    tolerance_W: float = 0.0
    conserved: bool = True


# --------------------------------------------------------------------------------------------- exposures

RatingStatus = Literal["WITHIN_RATING", "EXCEEDS_RATING", "RATING_MISSING", "CONDITIONS_MISMATCH", "CONDITIONS_NOT_VERIFIED",
                       "UNKNOWN"]
#: The conditions of a power rating a comparison records, one entry each (``RatingComparison.conditions``).
RatingConditionKind = Literal["regime", "wavelength", "pulse_duration", "repetition_rate", "beam_diameter", "pulse_count",
                              "polarization", "value_definition"]
RatingConditionState = Literal["VERIFIED", "DIFFERS", "NOT_VERIFIABLE", "NOT_STATED"]
#: How a condition that differs bounds the damage threshold of the light on the surface.
RatingConditionEffect = Literal["THRESHOLD_AT_LEAST_RATED", "THRESHOLD_AT_MOST_RATED", "NO_ESTABLISHED_RELATIONSHIP"]
#: Why a condition cannot be checked.
RatingConditionReason = Literal["DESIGN_INPUT_MISSING", "RATING_INFORMATION_MISSING", "BEAM_SIZE_UNKNOWN", "LIGHT_STATE_UNKNOWN",
                                "COHERENT_STATE_UNDETERMINED", "PULSES_PER_SITE_UNDETERMINED"]
#: What a rating says about the damage threshold of the light on the surface (``RatingComparison.applicability``).
RatingApplicability = Literal["AS_RATED", "THRESHOLD_AT_LEAST_RATED", "THRESHOLD_AT_MOST_RATED", "NOT_VERIFIED", "NOT_APPLICABLE"]


class RatingCondition(OmitAbsentFields, StrictModel):
    """One condition of a power rating against the light on a surface (``RatingComparison.conditions``).

    ``state``:

    * ``VERIFIED``: the rating states the condition and the light meets it within ``tolerance`` (relative);
    * ``DIFFERS``: the light is known to differ from it, with ``effect`` -- ``THRESHOLD_AT_LEAST_RATED`` (the damage
      threshold for this light is at least the rated value, so an exposure within the rating is within the threshold),
      ``THRESHOLD_AT_MOST_RATED`` (the threshold is at most the rated value, so an exposure above the rating is above it)
      or ``NO_ESTABLISHED_RELATIONSHIP`` (the rating says nothing about this light);
    * ``NOT_VERIFIABLE``: the condition cannot be checked, with ``reason`` -- ``DESIGN_INPUT_MISSING`` (the Project does
      not state what the condition needs, e.g. how many pulses a pulsed source sends), ``RATING_INFORMATION_MISSING`` (the
      rating does not state what deciding it needs: its pulse count, how a count above one was applied, whether its value
      is a peak value, the definition of its test beam diameter where the beam here lies between the readings, or its
      polarization in a checkable form), ``BEAM_SIZE_UNKNOWN`` (light of unknown size meets
      the surface), ``LIGHT_STATE_UNKNOWN`` (light of unknown state, wavelength or fate may reach it, or the state of a
      mixture of beams is not determined), ``COHERENT_STATE_UNDETERMINED`` (mutually coherent beams of different
      states, whose combined state depends on their phases) or ``PULSES_PER_SITE_UNDETERMINED`` (several crossings of
      pulsed light that each bring fewer pulses than the rating's count may together bring more to one site);
    * ``NOT_STATED``: the condition does not apply to the rating (a pulse duration, rate or count to a CW rating, an area
      definition to a ``W/cm`` or typical value). It is neither counted as verified nor does it keep the rating from
      applying. A test condition that applies but that the rating does not state -- its test beam diameter, a pulsed
      rating's repetition rate or pulse count, its polarization, its value's definition -- is ``NOT_VERIFIABLE``
      (``RATING_INFORMATION_MISSING``): the threshold depends on it.

    ``unit`` is the unit of the numbers: ``rated`` (the rating's value as stated, or its form: ``CW`` / ``PULSED``, ``S``,
    ``P``, ``LINEAR``, ``CIRCULAR``, ``UNPOLARIZED``), ``rated_low`` / ``rated_high`` (``beam_diameter``: the test beam's
    1/e^2 diameter -- the stated diameter for a ``1/E2`` definition, the stated FWHM / 0.5887 for ``FWHM``, from the
    stated value to 1.699 times it when the definition is not stated), ``design`` (the light's value when it has one: its
    regime, the pulses per site -- ``UNLIMITED`` for light that keeps pulsing) and ``design_low`` / ``design_high`` (the
    range of the light's values: pulses per site, wavelengths, pulse durations, repetition rates; ``beam_diameter``: the
    smallest 1/e^2 diameter across a beam and the largest 1/e^2 extent of a beam's footprint on the surface). ``message``
    says what was compared and found."""

    model_config = schema_description("One condition of a power rating (regime, wavelength, pulse duration, repetition rate, beam diameter, polarization, angle, exposure definition) against the light on a surface: whether it is verified, differs (with the effect on the rated threshold) or cannot be checked (with the reason), and the values compared. The model, its derivations and its cases: help(os_optics.contracts.RatingCondition).")

    absent_omitted: ClassVar[tuple[str, ...]] = ("effect", "reason", "rated", "rated_low", "rated_high", "design", "design_low",
                                                 "design_high", "tolerance")

    condition: RatingConditionKind
    state: RatingConditionState
    effect: RatingConditionEffect | None = None
    reason: RatingConditionReason | None = None
    unit: str = ""
    rated: int | float | str | None = None
    rated_low: float | None = None
    rated_high: float | None = None
    design: int | float | str | None = None
    design_low: float | None = None
    design_high: float | None = None
    tolerance: float | None = None
    message: str = ""


class ExposureHit(BasisCarrier, OmitAbsentFields, StrictModel):
    """One traced beam meeting one surface: a path's ``pass_index``-th arrival at it (0 = first).

    ``incident_power_W`` is the beam's power arriving at the surface (before its clear aperture). ``beam_radius_x_mm`` /
    ``_y_mm`` are the Gaussian beam's two 1/e^2 radii there (its own transverse axes), ``aoi_deg`` the chief ray's angle of
    incidence and ``footprint_semi_axes_mm`` the semi-axes of the 1/e^2 footprint on the surface (major, minor; the tangent
    plane at the chief-ray hit). Peak values of the Gaussian profile: ``peak_irradiance_W_per_cm2`` = 2 P / (pi w_x w_y) in
    the beam's transverse plane, ``peak_irradiance_surface_W_per_cm2`` = that x cos(aoi) on the surface,
    ``linear_power_density_W_per_cm`` = P / d with d = 2 min(w_x, w_y) (the 1/e^2 diameter of the narrower axis), and for a
    pulsed source ``pulse_energy_J`` = P / repetition rate and ``peak_fluence_J_per_cm2`` = 2 E / (pi w_x w_y) (transverse
    plane). For M^2 > 1 the radii are the beam's second-moment radii and the peaks those of a Gaussian of that size. "Peak"
    is the spatial peak of the profile: for a pulsed source ``peak_irradiance_*`` is the spatial peak of the *average*
    irradiance (average power / beam area), not the irradiance at the top of a pulse; the pulse is described by
    ``pulse_energy_J`` and ``peak_fluence_J_per_cm2`` (the spatial peak of one pulse's fluence).
    ``status`` ``UNKNOWN`` with ``reason`` (the Gaussian state's code) when the beam has no valid Gaussian state there, and
    ``OUT_OF_MODEL_SCOPE`` with ``METRIC_NOT_VALIDATED_FOR_ELEMENT`` when the beam crossed a response model (a tabulated thin
    lens) that is not validated for the beam size behind it: the beam-size quantities are then ``None``, never 0.

    A hit of ``SurfaceExposure.beside_hits`` is a beam whose chief ray misses the surface while its Gaussian footprint reaches
    it: ``miss_distance_mm`` is a lower bound of the distance, transverse to the beam, from the chief ray to the nearest
    point of the surface's support (in its vertex plane; a sphere's rim plane), the radii are the beam's where its chief ray
    crosses that plane, and the peak values are the Gaussian profile's at that distance -- ``2 P / (pi w_x w_y) exp(-2 d^2 /
    w_max^2)`` and the like --, an upper bound of what reaches the surface (no ``P / d``: ``None``)."""

    model_config = schema_description("One traced beam meeting one surface (a path's pass_index-th arrival): its incident power, Gaussian radii, angle of incidence and footprint there, and the peak irradiance, linear power density and, for a pulsed source, fluence and pulse energy of its Gaussian profile. A value is null where the beam's power or profile is not known there; status and reason say why. The model, its derivations and its cases: help(os_optics.contracts.ExposureHit).")

    absent_omitted: ClassVar[tuple[str, ...]] = ("miss_distance_mm", "value_basis")

    path_id: str
    source_id: str
    spectral_index: int
    wavelength_nm: float
    pass_index: int = 0
    incident_power_W: float
    aoi_deg: float
    beam_radius_x_mm: float | None = None
    beam_radius_y_mm: float | None = None
    footprint_semi_axes_mm: Vector2 | None = None
    peak_irradiance_W_per_cm2: float | None = None
    peak_irradiance_surface_W_per_cm2: float | None = None
    linear_power_density_W_per_cm: float | None = None
    pulse_energy_J: float | None = None
    peak_fluence_J_per_cm2: float | None = None
    status: Status = Status.PASS
    reason: str | None = None
    miss_distance_mm: float | None = None
    value_basis: ValueBasisField = None


class RatingComparison(BasisCarrier, OmitAbsentFields, StrictModel):
    """One power rating of the element compared with a surface's exposure.

    ``conditions`` records every condition a rating can state, one entry each (``RatingCondition``), against the light on
    the surface (traced, untraced and beside it):

    * ``regime`` (``CW`` for continuous sources, ``PULSED`` for pulsed ones), ``wavelength`` (within
      ``tolerances.wavelength_rel`` of the rating's), and for a pulsed rating ``pulse_duration`` (a rated duration read as
      a FWHM) and ``repetition_rate`` (within ``tolerances.pulse_duration_rel`` / ``repetition_rate_rel``; a rate the
      rating does not state: not verifiable): verified within the tolerance, otherwise they differ with no established
      relationship (no scaling between wavelengths, durations or rates). On a surface only light
      left out as negligible reaches, that light's lines -- each with its regime, wavelength and pulse train -- are the
      light compared: a line that meets a condition verifies it; a line that differs leaves it not verifiable when it may
      carry more than ``tolerances.unlocated_light_share`` of the most light there, and whatever its power where light
      may be stored between surfaces facing each other (nothing bounds it there; the stored light's lines count too).
    * ``beam_diameter``, the test beam on the sample (not stated: not verifiable): the test beam's 1/e^2 diameter ``D`` is
      read as the stated diameter (``1/E2``), the stated FWHM / 0.5887 (``FWHM``), or between the stated value and 1.699
      times it when the definition is not stated (the standard width definitions of a Gaussian beam lie in that range; a
      width taken at a lower level would not). For each beam on the
      surface its smallest 1/e^2 diameter across the beam and the largest 1/e^2 extent of its footprint on the surface are
      compared: verified when the definition is stated and both transverse 1/e^2 diameters are within
      ``tolerances.beam_diameter_rel`` of ``D``; a beam whose footprint is no larger than the smallest reading of ``D`` has
      a threshold at least the rated value (a smaller spot meets fewer defects, and heats less at a given fluence or
      irradiance); a beam narrower nowhere than the largest reading has a threshold at most the rated value (small test
      spots overstate the threshold of larger beams); a beam between the readings of a definition that is not stated is
      not verifiable (``RATING_INFORMATION_MISSING``); a beam whose axes lie on both sides of a stated diameter differs
      with no established relationship. Beams in opposite directions differ with no established relationship. On a metallic
      reflector (``PowerRating.surface_kind``) a pulsed rating has no established direction across beam sizes unless it is
      a single-pulse rating applied to a single pulse per site (the multi-pulse threshold of metal mirrors has been found
      to rise with the spot size); a reflecting surface of unknown kind leaves such a direction not verifiable. A ``W/cm``
      rating applied to a larger beam has no established relationship. Light whose size is not known -- a beam without a
      valid Gaussian state, light bounded by its diffraction limit -- leaves a verified or at-least direction not
      verifiable (``BEAM_SIZE_UNKNOWN``) when it may carry more than ``tolerances.unlocated_light_share`` of the most light
      the surface can receive; a beam with no known value there always does. The weakest beams that carry, with that light,
      at most this share (a faint ghost of another size) are left out of the comparison. Both shares are measured in every
      quantity that gives the size its direction: the rating's own (peak irradiance, P / d -- both readings of an unstated
      definition -- or peak fluence: a micro-spot at a low repetition rate may carry little of the irradiance and most of
      the fluence) and, for a CW rating, the power and P / d (a beam's heating scales with its P / d, a broad beam heats
      the element with its power), for a pulsed rating the peak fluence and the pulse energy; a beam whose chief ray
      misses the surface counts in each with its tail bound there, and a beam without a value in one of them is never
      left out.
    * ``pulse_count`` against the pulses a site of the surface receives -- each crossing of a source's light there brings
      that source's ``pulse.total_pulses`` (``UNLIMITED``: more than any rated count; the lines of one source along one
      path are one crossing); crossings that carry together at most ``tolerances.unlocated_light_share`` of the most light
      the surface can receive, and followed light of lines the rating is not for within that share, do not count. The
      count is compared per tested site only where that is what it means: a single-pulse rating (one pulse on each tested
      site), or a count the rating states as applied at one fluence level to each tested site (``test_protocol``
      ``S_ON_1``). Damage is irreversible -- a site damaged within ``S`` pulses at a fluence is damaged within more, and a
      site that survives ``S`` survived its first fewer --, so under such a test: every site receiving exactly ``S``,
      verified; at least ``S``, a threshold at most the rated value; at most ``S``, a threshold at least the rated value;
      crossings that may bring a site fewer or more, ``PULSES_PER_SITE_UNDETERMINED``; light whose source states no
      ``total_pulses`` beyond that share, ``DESIGN_INPUT_MISSING``. A rating that states no count, or a count above one
      without saying how it was applied (at one level, ramped, raster), leaves the condition not verifiable
      (``RATING_INFORMATION_MISSING``) whatever the sources' totals: no relation to another number of pulses is
      established. Continuous light under a pulsed rating differs with no established relationship. ``design`` is the
      pulses per site when every crossing brings the same number, ``design_low`` / ``design_high`` their range otherwise
      (``design_high`` absent: some crossing keeps pulsing).
    * ``polarization``, the test polarization (``PowerRating.polarization``; not stated: not verifiable) for the light on
      the surface as a whole, any mixture of its beams included: verified when every beam is within ``tolerances.polarization_fraction`` of one
      common state of it -- ``S`` / ``P``: at most that share of its power in the other of s and p (its plane of
      incidence at the surface); ``LINEAR``: at most that share outside one linear axis (the major axis of the strongest
      traced beam); ``CIRCULAR``: at most that share outside one circular state (the strongest traced beam's, compared in
      the laboratory frame, so a beam's own reflection travelling back along it keeps it) -- since a mixture of such beams
      is then within it too (mutually coherent beams: the part outside the state carries at most that share of the most
      they can reach together). It differs (no established relationship) only where the light at a traced beam's centre
      violates it whatever the phases of coherent beams and the amount of light there of unknown size: the smallest share
      of the mixture in the orthogonal state above the tolerance (``UNPOLARIZED``: one polarization state carrying more
      than half the power plus the tolerance; every traced beam is fully polarized). It is not verifiable for a
      polarization stated only as text or in more words than its machine-readable form, ``S`` / ``P`` at normal incidence
      (no plane of incidence; the rating states no test angle), mutually coherent beams of different states, beams in
      different states whose mixture neither holds it everywhere nor violates it for certain, and light of unknown state
      that may reach the surface (power of unknown fate, light left out as negligible beyond the tolerance). Light left out
      as negligible of a line the rating is not for that may carry more than the tolerance of the most light the surface
      can receive leaves each condition that line differs in not verifiable (its regime; else its wavelength, pulse
      duration, repetition rate). These shares of left-out light are measured in peak irradiance and in the rating's own
      quantity, the larger counting.
    * ``value_definition``, what a ``J/cm2`` or ``W/cm2`` value is (``PowerRating.value_definition``): verified when the
      rating states it (``rated``: ``PEAK`` or ``TOTAL_PER_1E2_AREA``), not verifiable (``RATING_INFORMATION_MISSING``)
      when it does not -- the value then establishes no damage threshold for the peak exposure the Engine computes.
      ``NOT_STATED`` for a ``W/cm`` value (P / d has no area definition) and a ``TYPICAL`` value (which bounds no
      threshold).

    ``applicability``, first match: ``NOT_APPLICABLE`` when a condition differs with no established relationship or two
    differ in opposite directions; ``NOT_VERIFIED`` when a condition is not verifiable; ``THRESHOLD_AT_LEAST_RATED`` /
    ``THRESHOLD_AT_MOST_RATED`` when a condition differs in that direction; ``AS_RATED`` when every stated condition is
    verified. ``conditions_match`` is true only for ``AS_RATED``. ``mismatches`` lists the messages of the conditions that
    differ, ``unverified_conditions`` those of the conditions not verifiable.

    The exposure in the rating's unit: ``W/cm2`` the peak irradiance in the transverse plane, ``W/cm`` the power per beam
    diameter P / d of the narrower axis (1/e^2: d = 2 w; FWHM: d = w sqrt(2 ln 2); a rating that does not state its
    definition: the range from the 1/e^2 to the FWHM value), ``J/cm2`` the peak fluence, with several beams combined as
    ``SurfaceExposure`` describes. ``exposure_value`` without an entry in ``value_basis`` is the exposure, known exactly.
    Otherwise the exposure lies between ``exposure_lower_bound`` (``None``: no lower bound above 0) and
    ``exposure_upper_bound`` (``None``: nothing bounds it from above), and ``exposure_value`` is the exposure of the
    followed beams alone (one number where no two of them interfere and it lies within the bounds; ``None`` otherwise)
    with the entry ``value_basis["exposure_value"]`` naming the light it does not count: kind ``MODEL`` where that is
    only light left out as negligible (``upper``: the most that light can move the exposure away from the number),
    ``ESTIMATE`` where light the trace does not follow, light of unknown fate or light that may return between surfaces
    may reach the surface too (it is coherent with the beams and may raise or lower the exposure). The bounds and the
    status never read that number. The exposure is computed for every comparison whose quantity exists for the light on
    the surface, whatever its conditions -- also for a rating of another wavelength, pulse duration or repetition rate
    (``CONDITIONS_MISMATCH``: its ratio then carries the differing condition as a validity check); a ``J/cm2`` rating has
    no exposure for continuous light.

    ``utilization`` = exposure / rated value. Without an entry it is decided: the exposure is exact, or it is the lower
    bound / rated value where that is above 1 (entry kind ``LOWER_BOUND`` where light not counted may add), for a
    comparison that applies (as stated, or with a threshold at least or at most the rated value) and a value that
    establishes a threshold. Every other comparison that reports an exposure states the same ratio with the entry
    ``value_basis["utilization"]``: the test conditions the rating does not state taken as the light's own (``assumed``:
    ``UNSTATED_CONDITION_AS_DESIGN`` at those conditions; a value of unstated definition read as a peak value, ``value``
    ``PEAK`` -- read as ``TOTAL_PER_1E2_AREA`` the ratio to the threshold is half; a typical value read as the part's own,
    ``value`` ``TYPICAL``), every condition that differs or cannot be checked as a validity check of the rated value for
    this light (``validity``: ``within`` false with the light's value and the rating's range, or null), and what the
    exposure does not count (``omitted``). Its kind is ``ESTIMATE`` unless every stated condition is verified and the
    exposure is the model's value. It is the arithmetic ratio to the rated value as stated: ``status``,
    ``applicability`` and the surface's and element's ``rating_status`` say what is decided, and none of them reads it.

    The rated value as a threshold (``rated_threshold_low`` / ``rated_threshold_high``, in the rating's unit; absent: the
    value establishes none): the exposure is a peak value (peak fluence, peak irradiance) or P / d. ``PEAK``: the threshold
    is the value. ``TOTAL_PER_1E2_AREA``: the total energy or power over pi (d/2)^2 of a Gaussian test beam of 1/e^2
    diameter d is half its peak, so the threshold is twice the value. A value whose definition is not stated establishes
    no threshold (the ``value_definition`` condition). A ``W/cm`` rating's P / d is its own threshold (its diameter
    definition has the readings above). A rating whose ``rating_status`` is ``TYPICAL`` states a typical value, not one
    each part meets: it bounds no threshold.

    ``status`` by ``applicability``: ``AS_RATED`` -- ``EXCEEDS_RATING`` when the exposure (its lower bound) is above
    ``rated_threshold_high``, ``WITHIN_RATING`` when the exposure (its upper bound) is at most ``rated_threshold_low``,
    else ``UNKNOWN``; ``THRESHOLD_AT_LEAST_RATED`` -- ``WITHIN_RATING`` when the exposure (its upper bound) is at most
    ``rated_threshold_low``, else ``UNKNOWN``; ``THRESHOLD_AT_MOST_RATED`` -- ``EXCEEDS_RATING`` when the exposure (its
    lower bound) is above ``rated_threshold_high``, else ``UNKNOWN``; a ``TYPICAL`` value: ``UNKNOWN``; ``NOT_VERIFIED`` --
    ``CONDITIONS_NOT_VERIFIED`` (also a value whose definition is not stated);
    ``NOT_APPLICABLE`` -- ``CONDITIONS_MISMATCH`` (the rating does not describe this light; not an exceedance).
    ``notes``: what the comparison used (the diameter definition of P / d, the polarization found at the beams, a typical
    value)."""

    model_config = schema_description("One power rating of an element compared with the exposure of one of its surfaces: the rating, each of its conditions against the light (conditions), the exposure in the rating's unit with its bounds, the utilization (exposure / rated value) and the status. utilization is null where the exposure has no upper bound or no rating condition relates the rating to the light; status and conditions say which. The model, its derivations and its cases: help(os_optics.contracts.RatingComparison).")

    absent_omitted: ClassVar[tuple[str, ...]] = ("exposure_upper_bound", "unverified_conditions", "rated_threshold_low",
                                                 "rated_threshold_high", "value_basis")

    rating_index: int
    regime: Literal["CW", "PULSED"]
    unit: str
    rating_value: float
    rating_wavelength_nm: float
    rating_status: str
    basis: str = ""
    origin: str = ""
    conditions_match: bool
    mismatches: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    exposure_value: float | None = None
    exposure_lower_bound: float | None = None
    exposure_upper_bound: float | None = None
    utilization: float | None = None
    rated_threshold_low: float | None = None
    rated_threshold_high: float | None = None
    status: RatingStatus
    unverified_conditions: tuple[str, ...] = ()
    applicability: RatingApplicability
    conditions: tuple[RatingCondition, ...] = ()
    value_basis: ValueBasisField = None


class CoherentExposureGroup(StrictModel):
    """Mutually coherent light meeting one surface: two or more contributions of one coherence group and one spectral line,
    from either side, between some of which the degree of coherence is not zero.

    ``members`` name the contributions as ``<light>:<path_id>:<pass_index>`` (``light`` ``traced``, ``untraced`` or
    ``beside``, as in ``SurfaceExposure``). ``incidence``: the chief rays' direction against the surface's normal --
    ``ALONG_NORMAL`` (d . n > 0) or ``AGAINST_NORMAL`` when every member arrives from that side, ``BOTH_SIDES`` when they
    arrive from both (counter-propagating beams: a standing wave). ``largest_coherence_factor``: the largest degree of
    coherence between two members anywhere on the surface where both carry light: gamma = exp(-(tau / L_c)^2) from the
    sources' declared coherence length L_c and the two fields' relative delay tau, which equals the optical path
    difference of their chief rays only along them and changes across the surface with the beams' tilt, offset and
    wavefront curvature (beams crossing at an angle have no relative delay somewhere in their overlap: about 1);
    1 where a source declares no coherence length, the path difference is not defined (a path through an ideal or
    tabulated thin lens) or a beam's Gaussian state is not known."""

    coherence_group: str
    wavelength_nm: float
    incidence: Literal["ALONG_NORMAL", "AGAINST_NORMAL", "BOTH_SIDES"]
    members: tuple[str, ...]
    largest_coherence_factor: float


class SurfaceExposure(BasisCarrier, OmitAbsentFields, StrictModel):
    """The exposure of one element surface (``surface_id`` = ``<component>/<surface>``) by the light that meets it.

    How beams combine: the exposure is the intensity of the summed fields of the light arriving at the surface from either
    side, in irradiance units. Light of different spectral lines or of different coherence groups (mutually incoherent
    sources) has no interference term in the time average, so its values add. Beams of one coherence group and line are
    mutually coherent whichever side they arrive from (beams crossing the surface in opposite directions form a standing
    wave): their fields add with phases the exposure model leaves open, so their combined value is at most
    sum_i v_i + 2 sum_{i<j} gamma_ij sqrt(v_i v_j) <= (sum_i sqrt(v_i))^2 and at least the larger of
    max(0, sqrt(v_k) - sum_{j != k} sqrt(v_j))^2 and sum_i v_i - 2 sum_{i<j} gamma_ij sqrt(v_i v_j) (v the beams' values,
    gamma_ij their degree of coherence -- the largest anywhere on the surface for the upper bound, at the point for the
    lower one; ``coherent_groups``, reason ``COHERENT_BEAMS_BOUNDED``). Beams at different places combine through their
    Gaussian profiles: a lower bound is the combination at a traced beam's centre (each beam's value there), an upper bound
    the combination of the beams' peaks. Light the trace does not follow enters a lower bound only as light that may
    cancel the traced beams; an untraced beam whose size is not known counts with the diffraction limit of its power.
    Light of unknown place -- such a beam, and the light left out as negligible (below) -- cannot be strong over the whole
    area a traced beam covers: a lower bound is also the traced beam's mean against a window over that area, less what its
    power can take away there (the window bound, ``validation/window_bound.py``, on a plane or a sphere: in peak
    irradiance and peak fluence for a beam of one Gaussian mode; in ``P / d``, that light as Gaussian beams crossing the
    surface from one side, by the strip of the window each covers).

    Light left out as negligible (in a run with power ratings): the trace and the probes leave out branches of at most
    ``negligible_power_fraction`` x the line input (or the level of a deeper pass), and paths stop at the power floor. Light
    of power ``p`` at vacuum wavelength ``lambda`` in a medium of index ``n`` cannot reach an irradiance above its
    diffraction limit ``2 pi n^2 p / lambda^2`` (the peak of a Gaussian beam of waist ``lambda / (pi n)``; the most any
    field of that power crossing a plane from one side can concentrate) -- however faint, focused it can still matter. The
    left-out light of each line (``ExposureReport.faint_light``) therefore joins the light of its line on every surface as
    a field of open phase and polarization whose amplitude ``A`` is at most ``c`` times that limit's (``n`` the larger index
    of the surface's media), ``c`` the most field its open paths' crossings bring onto a surface or across a plane:
    between two reflections it meets each lens and each refracting surface between media of different index at most once
    and travels in at most ``L`` straight legs (``L - 1`` the lenses and refracting surfaces that may turn it), each
    meeting any other surface at most as often as a straight line can (a plane once, a sphere or a conic twice, an asphere
    by the degree of its sag along a line: ``K``), so it meets the surface at most ``m`` times after each number of
    reflections (1 for a lens or a refracting surface between different media, ``K L`` for any other surface). After ``j``
    reflections it is made of pieces from different sequences of ``j`` of the ``N`` surfaces that may reflect it (no
    surface reflects it twice on an open path), mutually coherent, whose fields add: at most ``a_j`` times one crossing's,
    the smaller of the sum over those ordered sequences of the products of ``m_s sqrt(R_s)`` (``R_s`` a surface's
    reflectance bound, ``m_s`` its meetings) and the square root of their number (the products of ``m_s``) times the power
    they share (the sum of the products of ``m_s R_s``, at most 1). So ``c = sqrt(2) m sum_{j=0}^{N} a_j``, the light split
    between the two sides each time, and at least 2 (the light once from either side). It raises the comparisons' upper
    bounds to ``sum v + 2 sum gamma sqrt(v_i v_j) + 2 A sum sqrt(v_i) + A^2`` and lowers the bounds at the beams' centres by
    the same amplitude ``A``, the window bound by what its power ``(c f)^2 P*`` can take away
    (``FAINT_LIGHT_BOUNDED``, ``faint_light_peak_W_per_cm2``). A rated surface no other light reaches lies between 0 and
    that bound, never a known zero. Light that goes back and forth between two reflecting surfaces facing each other --
    the faces of a window, a lens or a cube, a window and a mirror, the mirrors of a linear cavity, with lenses or plates
    between them -- arrives at them and at every surface between them again and again: each round trip keeps at most
    ``g = r_a r_b`` of its field (the two amplitude reflectances, bounded over every angle at which the pair's light can
    meet them below the critical angle), so all its returns bring at most ``g / (1 - g)`` times one crossing more. What
    each return sends out of the pair (through its surfaces, past an aperture's edge, reflected by a surface between them)
    is at most that return's field and may reach any surface: on every surface, from either side, ``A`` is multiplied by
    ``f = 1 + sum g / (1 - g)`` over the scene's pairs (``faint_light_return_factor``, ``FAINT_LIGHT_RECIRCULATES``;
    ``validation/resonance.py``). A pair whose gain bound is 1 -- two surfaces that reflect everything, a coating met
    outside its data, an uncoated face met from inside at angles up to its critical angle (the faces of an uncoated lens)
    -- bounds nothing: no upper bound on any surface. Left-out light that may circulate through three or more surfaces
    of separate elements (a ring, a folded or coupled cavity, a periscope's windows through its folds) leaves every
    surface it may reach without an upper bound (one ``EXPOSURE_BOUND_WITHHELD`` notice of the run names the return's
    surfaces, ``closed_return_surfaces``, and the surfaces without a bound); a stack of parallel planes keeps a bound that counts all its returns where a series of
    them converges, and is treated like a ring where none does. A plane-faced solid's internal folds (and total internal
    reflection in it) add their certified closed walks to the open paths' count, or make it a closed return where they have no
    certified sum. Not counted, with their magnitudes in ``model``: light that total internal reflection holds in a curved lens
    body and light that refraction turns back onto a refracting surface it already passed, before it is reflected again. The left-out light of
    a line is left out in pieces, which are mutually coherent: ``A`` is that of their power together in phase, ``(sum of
    the square roots of their powers)^2`` (``ExposureReport.faint_light[].in_phase_power_W``), not of their total power.
    Where this light alone keeps a comparison from being decided, the run follows its lines deeper, so that the probes'
    known spots replace the bound (``faint_light[].followed_to_W``).

    ``hits``: the traced beams (lines, paths and passes). The surface values combine them as above at their largest:
    ``peak_irradiance_W_per_cm2`` / ``peak_irradiance_surface_W_per_cm2`` / ``linear_power_density_W_per_cm`` /
    ``peak_fluence_J_per_cm2`` are the beams' values summed where no two can interfere (as if their peaks coincided) and
    the largest value coherent beams can reach otherwise; ``incident_power_W`` likewise as a flux through the surface
    (beams arriving from opposite sides add their powers; coherent beams from one side at angles of incidence a_i, a_j:
    cross terms up to 2 gamma sqrt(P_i P_j) (cos a_i + cos a_j) / (2 sqrt(cos a_i cos a_j))). They are
    upper bounds of the traced light's exposure and its exact value for a single beam. ``complete`` is false when part of
    the exposure is not known (``reason_codes``: a beam without a valid Gaussian state, a beam behind a response model not
    validated for its size, power of unknown fate: ``INCOMPLETE_TRACE`` and its causes, light the trace does not follow
    that reaches the surface, coherent beams whose combination is a range); the values then cover the traced beams with a
    known size and are ``None`` when nothing is known. ``untraced_hits``: light the trace does not follow (an untraced ghost
    reflection, a reflector coating's transmission, light passing an aperture or an element its chief ray misses),
    followed through the optics by probe traces when the run has power ratings
    (``ExposureReport.untraced_light_included``), where it meets this surface (also when traced beams of the same source
    line meet it): not in the surface values, it raises the comparisons' upper bounds and, being coherent with the traced
    beams of its line, may lower their lower bounds. ``beside_hits``: in a run with power ratings, beams (traced or
    untraced) whose chief ray misses this surface while their Gaussian footprint reaches its support (within
    ``aperture_quad_extent_sigmas`` beam radii; ``INCOMPLETE_ENVELOPE``): a lower bound 0 and an upper bound from the
    profile at the nearest point -- never a known zero. ``rating_status`` from the comparisons' utilization bounds
    (exposure / damage threshold): a lower bound, the largest of the ``AS_RATED`` and ``THRESHOLD_AT_MOST_RATED``
    comparisons' lower bounds; an upper bound only when an ``AS_RATED`` or ``THRESHOLD_AT_LEAST_RATED`` comparison
    exists, then the largest upper bound of those and of the ``NOT_VERIFIED`` ones (a rating that may apply as stated
    may be the stricter one); ``NOT_APPLICABLE`` comparisons count in neither. A ``THRESHOLD_AT_MOST_RATED`` comparison
    -- and a ``NOT_VERIFIED`` one, which may have such a threshold -- leaves no upper bound (the threshold may lie below
    the exposure) unless a comparison of the same unit that is ``AS_RATED`` or ``THRESHOLD_AT_LEAST_RATED`` has a lowest
    threshold no larger than its highest under every reading (``W/cm`` values compared in P / d of the 1/e^2 diameter: a
    ``FWHM`` value stands 1 / 0.5887 above the same threshold, one of unstated definition between): the threshold then
    lies between the two. Where a ``THRESHOLD_AT_MOST_RATED`` comparison has only such comparisons whose lowest may be
    larger the ratings disagree, as where the lower bound lies above the upper bound: the lower bound stands and there
    is no upper bound. Comparisons of different units are criteria of their own. ``EXCEEDS_RATING`` when the lower bound
    is above 1, ``WITHIN_RATING`` when the upper bound is at most 1, else ``CONDITIONS_NOT_VERIFIED`` when a comparison
    is ``NOT_VERIFIED``, ``UNKNOWN`` when one applies in some direction, ``CONDITIONS_MISMATCH`` otherwise;
    ``RATING_MISSING`` when no rating of the element applies to this surface.
    ``faint_light_peak_W_per_cm2``: the most transverse peak irradiance the light left out as negligible could reach on
    this surface on its own (the sum over its lines' coherence classes of their amplitude bound squared, returns included).
    ``faint_light_return_factor``: the largest factor its field here was multiplied by for the returns between surfaces
    facing each other (omitted when the scene has no such pair). Both are proven bounds when they carry no entry in
    ``value_basis``. Where returns leave that light without a bound on this surface -- pairs whose gain bound is 1, a
    closed return whose one found trail has a round gain below 1, the folds of a solid without a certified sum -- and
    nothing else does, the two fields state the same quantities with each such pair counted at its centre-line incidence
    (one ray for the pair's light) and the trail's returns as its geometric series, with an ``ESTIMATE`` entry:
    ``assumed`` ``SINGLE_RAY_FOR_BEAM`` at the pairs' surfaces (``value``: the largest centre-line gain) and ``omitted``
    ``POSSIBLE_RETURN`` at the surfaces of the returns not bounded. The estimate enters no comparison, bound or status:
    ``exposure_upper_bound`` stays absent."""

    model_config = schema_description("The exposure of one element surface (surface_id = <component>/<surface>) by the light that meets it: the traced beams (hits), their combined peak values with the bounds that light the trace does not follow and coherent addition allow, whether the values are complete, and the comparisons with the element's power ratings. A value is null where nothing bounds it; complete, the bound fields and the issue codes say why. The model, its derivations and its cases: help(os_optics.contracts.SurfaceExposure).")

    absent_omitted: ClassVar[tuple[str, ...]] = ("untraced_hits", "beside_hits", "coherent_groups", "faint_light_peak_W_per_cm2",
                                                 "faint_light_return_factor", "value_basis")

    surface_id: str
    element_id: str
    hits: tuple[ExposureHit, ...]
    incident_power_W: float
    peak_irradiance_W_per_cm2: float | None = None
    peak_irradiance_surface_W_per_cm2: float | None = None
    linear_power_density_W_per_cm: float | None = None
    peak_fluence_J_per_cm2: float | None = None
    complete: bool = True
    reason_codes: tuple[str, ...] = ()
    rating_status: RatingStatus
    ratings: tuple[RatingComparison, ...] = ()
    untraced_hits: tuple[ExposureHit, ...] = ()
    beside_hits: tuple[ExposureHit, ...] = ()
    coherent_groups: tuple[CoherentExposureGroup, ...] = ()
    faint_light_peak_W_per_cm2: float | None = None
    faint_light_return_factor: float | None = None
    value_basis: ValueBasisField = None


#: Why a rated surface leaves its element's utilization without an upper bound (``UtilizationOpening.cause``).
OpeningCause = Literal["NO_UPPER_RATING", "EXPOSURE_UNBOUNDED", "NO_THRESHOLD", "THRESHOLD_OPEN_BELOW", "RATINGS_DISAGREE"]


class UtilizationOpening(OmitAbsentFields, StrictModel):
    """One reason an element's utilization has no upper bound (``ElementExposure.utilization_open_by``): the rated
    surface and, where one comparison is the reason, that comparison (``rating_index``, ``applicability``). ``cause``:
    ``NO_UPPER_RATING`` -- no rating on the surface applies as stated or with a threshold at least its value;
    ``EXPOSURE_UNBOUNDED`` -- the comparison's exposure has no upper bound (on a surface no followed light reaches: light
    of unknown fate may reach it); ``NO_THRESHOLD`` -- the comparison's rated value establishes no lowest threshold;
    ``THRESHOLD_OPEN_BELOW`` -- a comparison with a threshold at most its rated value, or one not verified, that no rating
    of its unit applying as stated or with a threshold at least its value bounds; ``RATINGS_DISAGREE`` -- such a
    comparison lies below a rating of its unit that applies, or the surface's lower bound lies above its upper bound."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("rating_index", "applicability")

    surface_id: str
    rating_index: int | None = None
    applicability: RatingApplicability | None = None
    cause: OpeningCause


class ElementExposure(BasisCarrier, OmitAbsentFields, StrictModel):
    """Per element: the most severe ``rating_status`` of its rated surfaces -- the surfaces at least one of its ratings
    applies to (a rating's ``surface_ids``, or every surface) -- and ``RATING_MISSING`` for an element without ratings.
    A rated element's surfaces that no rating names are listed in ``unrated_surface_ids`` and do not count. A rated surface
    no light reaches has a known zero exposure (``WITHIN_RATING``, utilization 0) only when the trace is complete and no
    light the trace does not follow, or leaves out as negligible, may reach it. ``max_utilization`` without an entry in
    ``value_basis`` = the largest exposure / damage threshold over the rated surfaces: every rated surface is decided by
    ``AS_RATED`` comparisons known exactly and no rating is ``NOT_VERIFIED``. Otherwise ``max_utilization`` is the largest
    reported exposure / damage threshold (the rated value where it establishes no threshold) over the comparisons of the
    ratings not known to be for other light (every comparison where all are), with the entry
    ``value_basis["max_utilization"]`` -- that comparison's (``RatingComparison.utilization``) and what the other rated
    surfaces leave open (a rated surface with no reported exposure: ``omitted`` at that surface, with the most its proven
    bound can add where only light left out as negligible reaches it) -- or ``None`` when no comparison reports an
    exposure; a number beside the bounds, which decides nothing. The largest utilization lies between
    ``utilization_lower_bound`` (the largest of the surfaces' lower bounds, ``SurfaceExposure``: a decided exceedance
    stands; 0 when only ratings with a threshold at least the rated value bound it) and ``utilization_upper_bound``
    (``None``: unbounded from above), for the reasons in ``reason_codes``. There is no upper bound while a rated surface
    has none -- no rating there applies as stated or with a threshold at least the rated value, a rating with a
    threshold at most its value, or one not verified, has no rating of its unit to bound it (``SurfaceExposure``), or
    its ratings disagree. ``POWER_RATING_CONDITIONS_NOT_VERIFIED`` names a rating not verified on a rated surface,
    ``POWER_RATING_CONDITIONS_MISMATCH`` a rated surface where no rating applies (every comparison
    ``NOT_APPLICABLE``). ``rating_status`` is the most severe label; the results are the comparisons
    (``SurfaceExposure.ratings``): ``rating_status_counts`` gives how many comparisons on the rated surfaces have each
    status, and ``utilization_open_by`` names what leaves the utilization without an upper bound (``UtilizationOpening``;
    empty where it has one)."""

    model_config = schema_description("The exposure of one element: its surfaces' exposures and the element's worst power-rating utilization with its status (the most severe of its ratings' comparisons). utilization is null where a surface of the element has no exposure upper bound or no rating applies. The model, its derivations and its cases: help(os_optics.contracts.ElementExposure).")

    absent_omitted: ClassVar[tuple[str, ...]] = ("utilization_lower_bound", "utilization_upper_bound", "unrated_surface_ids",
                                                 "reason_codes", "value_basis", "rating_status_counts", "utilization_open_by")

    element_id: str
    rating_status: RatingStatus
    rating_count: int = 0
    max_utilization: float | None = None
    worst_surface_id: str | None = None
    utilization_lower_bound: float | None = None
    utilization_upper_bound: float | None = None
    unrated_surface_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    value_basis: ValueBasisField = None
    rating_status_counts: dict[RatingStatus, int] | None = None
    utilization_open_by: tuple[UtilizationOpening, ...] = ()


class FaintLight(StrictModel):
    """The light of one source line the run does not follow to its end and bounds by its diffraction limit
    (``SurfaceExposure``): ``power_W`` the total power of the branches the trace and the probes left out as negligible, of
    the paths ended at the power floor and of the light of unknown fate whose power is known (the trace's or the probes'
    budget, a path ended out of model scope, a line not launched, the light a tabulated thin lens does not deliver; not
    the light at the interaction limit, which bounds nothing); ``in_phase_power_W`` the power those pieces could carry
    together where they meet in phase, ``(sum of the square roots of their powers)^2`` -- the pieces of a line are
    mutually coherent, so their fields add in amplitude, and the bounds use this, not ``power_W``; ``followed_to_W`` the
    power down to which the line's untraced light was followed (``negligible_power_fraction`` x the line input, or lower
    where a deeper pass was needed to decide a comparison)."""

    source_id: str
    spectral_index: int
    wavelength_nm: float
    power_W: float
    in_phase_power_W: float
    followed_to_W: float


class ExposureReport(OmitAbsentFields, StrictModel):
    """Per-surface beam exposure of the run and its comparison with the elements' power ratings (``exposures``).

    Every physical surface a traced beam meets is listed (never 0 for one it may not meet), and in a run with power ratings
    every surface light the trace does not follow reaches. ``trace_complete`` is false when part of the launched power has
    an unknown fate (trace budget, interaction limit, a path ended out of model scope, the light a tabulated thin lens does
    not deliver into its focused beam; ``unknown_fate_power_W``): every surface is then ``complete`` false. In a run with
    power ratings that power, where it is known, is bounded with the light left out as negligible (``faint_light``); power
    at the interaction limit (light that may be stored in a cavity) leaves every surface without an upper bound, no rating
    comparison ``WITHIN_RATING`` and the lower bounds of its lines 0 (it may be coherent with their beams). Without ratings
    any unknown fate does. ``untraced_light_included``: light the trace does not follow was followed to the surfaces
    (``SurfaceExposure.untraced_hits``) -- done when the run has power ratings; otherwise the surface values cover the
    traced beams only. ``tolerances`` are the condition-matching settings in force (the run's
    ``NumericalConfig.rating_*``, a Project's ``numerical_settings``; relative; ``beam_diameter_rel`` of a test beam
    diameter; ``polarization_fraction`` the largest share of a beam's power in the state orthogonal to a rating's test
    polarization; ``unlocated_light_share`` the largest share of the most light a surface can receive that light of
    unknown state, size or line -- and followed light the rating is not for -- may carry: in the rating's own quantity,
    and for light of unknown state or line also in peak irradiance). No scaling between conditions, no thermal or damage model: a comparison is decided
    where every condition the rating states is verified, or in the one direction a condition that differs establishes
    (``RatingComparison.applicability``). The comparisons are reported in the results; an exceedance
    raises the notice ``POWER_RATING_EXCEEDED``
    (status ``NOT_APPLICABLE``: it does not change the run's status) -- only a ``power_rating_utilization`` target does.
    ``faint_light``: per source line, the light left out as negligible or of unknown fate of known power, which bounds
    every surface's exposure by its diffraction limit and its returns (a run with power ratings)."""

    model_config = schema_description("Per-surface beam exposure of the run and its comparison with the elements' power ratings: every physical surface a traced beam meets and, in a run with power ratings, every surface that light the trace does not follow reaches; whether the trace accounts for all launched power (trace_complete, unknown_fate_power_W), the light left out as negligible (faint_light) and the model statement (model). The model, its derivations and its cases: help(os_optics.contracts.ExposureReport).")

    absent_omitted: ClassVar[tuple[str, ...]] = ("faint_light",)

    surfaces: tuple[SurfaceExposure, ...] = ()
    elements: tuple[ElementExposure, ...] = ()
    trace_complete: bool = True
    unknown_fate_power_W: float = 0.0
    untraced_light_included: bool = False
    tolerances: dict[str, float] = Field(default_factory=dict)
    model: str = ""
    faint_light: tuple[FaintLight, ...] = ()


# ------------------------------------------------------------------------------------------- layout


class LayoutIssue(StrictModel):
    kind: str
    status: Status
    device_ids: tuple[str, ...]
    clearance_mm: float | None = None
    required_clearance_mm: float | None = None
    deficit_mm: float | None = None
    contact_or_nearest_points: tuple[Vector2, ...] = ()
    rotated_bounds: dict[str, tuple[float, float, float, float]] = Field(
        default_factory=dict, description="Per device (minx, miny, maxx, maxy) of the rotated footprint."
    )
    suggested_moves: tuple[dict[str, tuple[float, float]], ...] = Field(
        (), description="Each item maps device_id -> (dx_mm, dy_mm) that removes the deficit."
    )
    message: str = ""


class LayoutResult(StrictModel):
    status: Status
    issues: tuple[LayoutIssue, ...] = ()
    min_clearance_mm: float | None = Field(None, description="Smallest footprint-to-footprint clearance among height-overlapping pairs.")
    min_clearance_pair: tuple[str, str] | None = None
    checks_run: tuple[str, ...] = ()
    rotated_bounds: dict[str, tuple[float, float, float, float]] = Field(default_factory=dict)


# ------------------------------------------------------------------------------------------ control


class SensitivityResult(StrictModel):
    actuator_refs: tuple[str, ...]
    sensor_ids: tuple[str, ...]
    jacobian: tuple[tuple[float, ...], ...] = Field(description="d sensor / d actuator (rows = sensors, columns = actuators).")
    difference_orders: tuple[int, ...] = Field((), description="1 (one-sided at a limit) or 2 (central) per actuator.")
    singular_values: tuple[float, ...] = ()
    numerical_rank: int = 0
    condition_number: float | None = None
    unobservable_actuator_refs: tuple[str, ...] = ()
    observable: bool = False
    status: Status = Status.UNKNOWN
    message: str = ""


class ControllerRun(StrictModel):
    converged: bool
    iterations: int
    initial_cost: float
    final_cost: float
    final_state: dict[str, float]
    saturated_actuators: tuple[str, ...] = ()
    lost_beam: bool = False
    capture_used: bool = False
    trajectory_cost: tuple[float, ...] = ()
    status: Status
    message: str = ""


class RecoverabilityCase(StrictModel):
    disturbance: dict[str, float]
    outcome: Literal[
        "RECOVERED", "NOT_CONVERGED", "LOST_UNRECOVERABLE", "SATURATED", "UNOBSERVABLE"
    ]
    run: ControllerRun | None = None
    message: str = ""


class ControlResult(StrictModel):
    status: Status
    sensitivity: SensitivityResult | None = None
    nominal_run: ControllerRun | None = None
    recoverability_cases: tuple[RecoverabilityCase, ...] = ()
    recoverability_fraction: float | None = None
    monte_carlo_recovered_fraction: float | None = None
    monte_carlo_sample_count: int = 0
    monte_carlo_seed: int | None = None
    monte_carlo_lost_fraction: float | None = None
    monte_carlo_saturated_fraction: float | None = None
    message: str = ""


# ------------------------------------------------------------------------------------------ overall


class ValidationSummary(StrictModel):
    target_counts: dict[str, int] = Field(default_factory=dict)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    path_count: int = 0
    observation_count: int = 0


class SurfaceProfileEvidence(OmitAbsentFields, StrictModel):
    """What a surface with a two-variable sag shape was, and what the intersection solver did on it during the physics run.
    ``aperture_reach_mm``: stated when the clear aperture reaches beyond the normalization radius of a Zernike shape (the
    stated polynomial is evaluated there, ``rho > 1``)."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("aperture_reach_mm",)

    surface_id: str
    kind: Literal["ZERNIKE", "XY_POLYNOMIAL", "BICONIC", "TOROIDAL"]
    indexing: Literal["NOLL", "FRINGE"] | None = None
    normalization: Literal["RMS_ORTHONORMAL", "PEAK_UNIT"] | None = None
    normalization_radius_mm: float | None = None
    aperture_reach_mm: float | None = Field(None, description="How far the clear aperture reaches from the surface axis (mm), "
                                            "stated when beyond the normalization radius of a Zernike shape.")
    units: Literal["mm"] = Field("mm", description="Unit of the transverse coordinates, the sag and every coefficient.")
    vertex_sag_mm: float = Field(description="Sag of the shape on the surface axis (its terms are not referenced to the vertex).")
    solved_rays: int = Field(0, description="Ray parameters the bracketed solver refined on this surface.")
    max_iterations: int = Field(0, description="The most Newton / bisection steps one call took, of NumericalConfig.sag_max_iterations.")
    max_residual_mm: float = Field(0.0, description="The largest |z - sag(a, b)| of a converged intersection.")
    non_converged_rays: int = Field(0, description="Rays whose bracket did not converge within the iteration cap; each is reported, not taken for a miss.")


class SourceSpectrum(OmitAbsentFields, StrictModel):
    """How one Gaussian-beam source's spectrum is traced (``optics/interference/spectrum.py``).

    ``DISCRETE``: the source's lines as stated (``line_count`` of them). ``SAMPLED_CONTINUOUS``: a continuous density
    (``shape`` over ``band_nm``) traced as ``line_count`` quadrature lines (``MIDPOINT_BINS``: the bins' centres, each with its
    bin's power and the bin's width ``line_width_nm`` as Gaussian linewidth). A power the run sums over the lines is the
    midpoint quadrature of the band's integral. The coherence of the source with itself follows from the spectrum:
    ``coherence_length_mm`` is the smallest path difference at which the degree of coherence ``|gamma|`` of the continuous
    spectrum falls to 1/e; ``coherence_length_lines_mm`` the same of the traced lines (each a Gaussian line) and
    ``coherence_length_doubled_lines_mm`` of twice as many lines; ``gamma_max_difference`` (``_doubled``) the largest difference
    between ``|gamma|`` of the lines (of twice as many) and of the spectrum over ``0 .. convergence_range_mm``: the
    convergence of the quadrature. ``reason`` says why a length has no value."""

    absent_omitted: ClassVar[tuple[str, ...]] = (
        "shape", "band_nm", "quadrature", "line_width_nm", "centroid_nm", "rms_width_nm", "coherence_length_mm", "coherence_length_lines_mm",
        "coherence_length_doubled_lines_mm", "gamma_max_difference", "gamma_max_difference_doubled", "convergence_range_mm", "reason")

    source_id: str
    basis: Literal["DISCRETE", "SAMPLED_CONTINUOUS"]
    line_count: int = Field(ge=1)
    shape: Literal["GAUSSIAN", "LORENTZIAN", "TOP_HAT", "TABULATED"] | None = None
    band_nm: tuple[float, float] | None = None
    quadrature: Literal["MIDPOINT_BINS"] | None = None
    line_width_nm: float | None = Field(None, description="Width of a quadrature bin: the Gaussian linewidth of each traced line.")
    centroid_nm: float | None = Field(None, description="Power-weighted mean vacuum wavelength of the spectrum within its band.")
    rms_width_nm: float | None = Field(None, description="RMS width of the spectrum within its band.")
    coherence_length_mm: float | None = Field(None, description="Smallest path difference with |gamma| = 1/e, of the continuous spectrum.")
    coherence_length_lines_mm: float | None = Field(None, description="The same, of the traced quadrature lines.")
    coherence_length_doubled_lines_mm: float | None = Field(None, description="The same, of twice as many quadrature lines.")
    gamma_max_difference: float | None = Field(None, description="Largest | |gamma_lines| - |gamma| | over 0 .. convergence_range_mm.")
    gamma_max_difference_doubled: float | None = Field(None, description="The same with twice as many lines.")
    convergence_range_mm: float | None = Field(None, description="The path differences the gamma differences are taken over (0 to this).")
    reason: str | None = None


class ValidationResult(OmitAbsentFields, StrictModel):
    """Result of ``validate``. ``status`` is the worst status over targets, physics issues and enabled stages: a summary of
    diagnostic severity, never an approval of the design, and it may read ``WARNING`` or ``UNKNOWN`` beside a target that is
    ``MET``. ``outcome`` and each target's ``assessment`` say what was decided, each quantity's own state whether it is known.

    A result with no targets and nothing evaluated is ``NOT_APPLICABLE``, never ``PASS``. ``exposures``: the per-surface beam
    exposure and power-rating comparison of the physics run (``ExposureReport``). ``surface_profiles``: one entry per surface
    with a two-variable sag shape (``SurfaceProfileEvidence``). ``evaluation``: the scope of a narrowed evaluation
    (``os_optics.evaluate``); a complete one has none.
    """

    absent_omitted: ClassVar[tuple[str, ...]] = ("exposures", "outcome", "sampled_sources", "surface_profiles", "grating_orders",
                                                 "incoherent_pairs", "source_spectra", "nonlinear_conversions", "evaluation")

    status: Status
    outcome: ResultOutcome | None = Field(
        None,
        description="The run's scientific-result contract: did the calculation complete and why (execution), and "
        "how the declared targets were assessed (requirements). Independent of status, which stays the summary of "
        "diagnostic severity. None only on a result stored before this field existed.",
    )
    summary: ValidationSummary
    targets: tuple[TargetResult, ...] = ()
    observations: tuple[ObservationResult, ...] = ()
    paths: tuple[OpticalPath, ...] = ()
    interference: tuple[InterferenceResult, ...] = ()
    incoherent_pairs: tuple[IncoherentPairs, ...] = Field(
        (), description="Per observation plane, the count of same-wavelength beam pairs from mutually incoherent sources: "
        "no fringes, their irradiances add; they are not listed in interference (except a pair a target selects). Omitted "
        "where there is none.")
    layout: LayoutResult | None = None
    control: ControlResult | None = None
    issues: tuple[ValidationIssue, ...] = ()
    power_budget: PowerBudget | None = None
    exposures: ExposureReport | None = None
    grating_orders: tuple[GratingOrderResult, ...] = Field(
        (), description="Every diffraction order of every beam's pass through a grating. Omitted when no beam meets one.")
    nonlinear_conversions: tuple[NonlinearConversionResult, ...] = Field(
        (), description="Every beam's pass into a crystal that generates its second harmonic. Omitted when no beam meets one.")
    sampled_sources: tuple[SampledSourceLine, ...] = Field(
        (), description="Every spectral line of the run's extended emitters: emitted power, derived radiance or intensity, the "
        "sampling domain and how many rays stand behind the results. Absent without extended emitters.")
    surface_profiles: tuple[SurfaceProfileEvidence, ...] = ()
    source_spectra: tuple[SourceSpectrum, ...] = Field(
        (), description="How each Gaussian-beam source's spectrum is traced (discrete lines, or a continuous spectrum's quadrature "
        "lines with the coherence derived from it and the quadrature's convergence). Present when a source states a continuous "
        "spectrum; absent: every source's lines are its discrete spectrum.")
    evaluation: EvaluationScope | None = Field(
        None, description="The scope of a narrowed evaluation: the capabilities requested and executed, those of the request's "
        "complete scope that did not run with the result sections each would have produced, and the declared targets the "
        "caller did not select. A section named there is empty because it was not computed, not because nothing was found. "
        "Absent: every capability the request enables ran and every declared target was assessed.")
    manifest: ValidationManifest | None = None
