"""Lowered scene structures: plain dataclasses over NumPy arrays, global frame, no Pydantic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..optics.attenuation import AttenuatorG
from ..optics.complex_transmission import TransmissionMapG
from ..optics.fields.source_fields import PatternG
from ..optics.materials.coatings import CoatingEvaluator
from ..optics.polarization.phases import PhaseEvaluator

if TYPE_CHECKING:  # the scene model is imported by the surface modules
    from ..optics.polarization.faraday import FaradayResponse, RetardanceTable
    from ..optics.surfaces.grating import GratingG
    from ..optics.surfaces.tabulated_lens import ThinLensTable

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ApertureG:
    """An aperture region in a surface's ``(u, v)`` frame; every question about it goes through ``optics/apertures.py``."""

    shape: Literal["none", "circle", "rectangle", "ellipse", "d_shape", "polygon"] = "none"
    p1: float = 0.0  # circle / d_shape radius, rectangle half width (u), ellipse semi-axis along ``axis``
    p2: float = 0.0  # rectangle half height (v), ellipse semi-axis across ``axis``, d_shape flat's distance along ``axis``
    offset_u: float = 0.0
    offset_v: float = 0.0
    obstruction: float = 0.0
    axis: tuple[float, float] = (1.0, 0.0)  # unit (u, v) direction: an ellipse's first axis, a d_shape flat's outward normal
    vertices: tuple[tuple[float, float], ...] = ()  # polygon corners relative to the offset, counter-clockwise
    holes: tuple["ApertureG", ...] = ()  # openings through the element (no surface there): light passes them


@dataclass(frozen=True)
class ScatterG:
    """A surface's diffuse scatter (``contracts.optics.SurfaceScatter``): the scattered share, its side, its angular law
    (``exponent``: the cosine power of the intensity, 1.0 for Lambertian), and how its rays are sampled."""

    fraction: float
    side: Literal["reflect", "transmit"]
    law: Literal["LAMBERTIAN", "COSINE_POWER"]
    exponent: float
    sample_count: int
    seed: int


@dataclass(frozen=True)
class InteractionG:
    kind: str  # refract|mirror|thin_lens|beam_splitter|polarizer|retarder|absorber|aperture_stop|neutral_density|faraday_rotator|diffraction_grating|complex_transmission|observe
    coating: CoatingEvaluator | None = None
    trace_reflection: bool = False
    reflective_side: str = "positive"
    phase_s: float = 0.0
    phase_p: float = 0.0
    focal_length: float = 0.0
    transmittance: float = 1.0
    reflection_phase: float = 0.0
    transmission_phase: float = 0.0
    axis: Array | None = None  # global unit vector (polarizer / retarder axis)
    t_par: float = 1.0
    t_perp: float = 0.0
    retardance: float = 0.0
    design_wavelength_nm: float = 0.0
    achromatic: bool = False
    observation_id: str | None = None
    absorbing: bool = False
    active_aperture: "ApertureG | None" = None  # observation planes: active area (support of the plane is unbounded)
    acceptance_half_angle_deg: float | None = None
    observation_medium: str | None = None
    # a tabulated thin lens (kind "thin_lens"): its per-wavelength response and validity domain (optics/surfaces/tabulated_lens.py)
    table: ThinLensTable | None = None
    attenuator: AttenuatorG | None = None  # neutral_density: the density law in the global frame
    faraday: FaradayResponse | None = None  # faraday_rotator: its response per wavelength (optics/polarization/faraday.py)
    grating: GratingG | None = None  # diffraction_grating: its period, axes, orders and response (optics/surfaces/grating.py)
    phases: PhaseEvaluator | None = None  # stated interface phases (refract / mirror / beam_splitter)
    # wavelength-resolved transmittance of a polarizer / retarder: (min_nm, max_nm, t (t_par), t_perp), sorted
    bands: tuple[tuple[float, float, float, float], ...] = ()
    # a retarder's retardance stated per wavelength (optics/polarization/faraday.py ``RetardanceTable``); None: retardance above
    retardance_table: RetardanceTable | None = None
    # diffuse scatter of a refracting, reflecting or absorbing surface (optics/rays/scatter.py)
    scatter: ScatterG | None = None
    # complex_transmission: the element's amplitude and phase over its plane (optics/complex_transmission.py)
    transmission_map: TransmissionMapG | None = None

    def transmittances(self, wavelength_nm: float) -> tuple[float, float]:
        """``(t, t_perp)`` at ``wavelength_nm``: the values of a band holding it -- on a boundary two bands share, the
        conservative pair: the smaller ``t`` and the larger ``t_perp`` (the larger leak of a polarizer) --, else the scalar
        values (``transmittance`` / ``t_par`` and ``t_perp``)."""

        holding = [band for band in self.bands if band[0] <= wavelength_nm <= band[1]]
        if not holding:
            return (self.t_par if self.kind == "polarizer" else self.transmittance), self.t_perp
        return min(band[2] for band in holding), max(band[3] for band in holding)


@dataclass
class SolverStats:
    """What the intersection solver did on one freeform surface during a run (``ValidationResult.surface_profiles``)."""

    solved_rays: int = 0
    max_iterations: int = 0
    max_residual_mm: float = 0.0
    non_converged_rays: int = 0


@dataclass(frozen=True)
class FreeformG:
    """A surface's two-variable sag shape, ready to evaluate (``_vendor...common.sag_shapes.SagForm``), and what the result
    states about it: its kind, indexing, normalization and radius, the sag at the vertex and the solver's record."""

    form: Any
    kind: str
    indexing: str | None = None
    normalization: str | None = None
    normalization_radius: float | None = None
    vertex_sag: float = 0.0
    stats: SolverStats = field(default_factory=SolverStats, compare=False)
    #: How far the clear aperture reaches from the axis, when beyond a Zernike shape's normalization radius (else None).
    aperture_reach: float | None = None


@dataclass(frozen=True)
class FacetG:
    """The facet surface a plane is a patch of (``optics/surfaces/facets.py``): the surface (``<owner>/<surface id>``), what
    its facets stand for (``FACETED_EXACT`` or ``TESSELLATED_FROM_CAD`` with its chord deviation), whether they bound a
    solid, and the patch's triangles (indices into the surface's ``triangles``)."""

    mesh_id: str
    approximation: str
    chord_deviation_mm: float | None
    closed: bool
    triangles: tuple[int, ...]

    @property
    def tessellated(self) -> bool:
        return self.approximation == "TESSELLATED_FROM_CAD"


@dataclass(frozen=True)
class SurfaceG:
    surface_id: str  # "<owner>/<local id>"
    owner_id: str
    geometry: Literal["plane", "sphere", "sag"]
    origin: Array  # plane point (plane) / vertex (sag) / unused (sphere)
    normal: Array  # plane normal / sag axis (unit, toward the positive side); unused for sphere
    u_axis: Array
    v_axis: Array
    center: Array  # sphere
    radius: float
    cap_axis: Array
    aperture: ApertureG
    mount_margin: float
    medium_pos: str
    medium_neg: str
    interaction: InteractionG
    aperture_radius: float = 0.0  # sphere cap transverse radius
    # sag surfaces (optics/surfaces/sag.py): vertex curvature (+: centre on the +normal side), conic constant, even-asphere
    # coefficients (A4, A6, ...) and the cylindrical flag (curved along u_axis, straight along v_axis)
    curvature: float = 0.0
    conic: float = 0.0
    coefficients: tuple[float, ...] = ()
    cylindrical: bool = False
    # axicon term L of the sag (L rho, 0 = none): a cone / roof with its apex at the vertex
    linear: float = 0.0
    # outer boundary of the mount in (u, v) when it is not the clear aperture grown by ``mount_margin`` (planes, sag surfaces)
    outline: ApertureG | None = None
    # a two-variable sag shape (optics/surfaces/freeform.py): it adds to, or replaces, the terms above
    freeform: FreeformG | None = None
    # a planar patch of a triangle-facet surface (planes only): the surface and the triangles it holds
    facet: FacetG | None = None


@dataclass(frozen=True)
class SourceG:
    source_id: str
    owner_id: str
    position: Array
    direction: Array  # +z of the source frame in global coordinates
    e1: Array  # source-frame x (transverse basis)
    e2: Array
    spectrum: tuple[tuple[float, float], ...]  # (wavelength_nm, normalized weight)
    power_W: float
    w0x: float
    w0y: float
    z_waist: float
    m2x: float
    m2y: float
    jones: NDArray[np.complex128]  # (2,)
    medium: str
    # Coherence length of each spectral line (aligned with ``spectrum``), or None when the source's coherence is undefined.
    coherence_lengths_mm: tuple[float, ...] | None
    coherence_group: str
    ring_count: int
    points_per_ring: int
    # A partially polarized source: its degree of polarization P (< 1; None: fully polarized, ``jones``). Its lines are
    # traced as two mutually incoherent components (``optics/polarization/partial.py``); a component's own SourceG carries
    # ``polarization_component`` (0: ``jones``, 1: the orthogonal state) and that component's Jones vector and coherence group.
    degree_of_polarization: float | None = None
    polarization_component: int | None = None
    # An analysis-only pupil sampling (results/imaging_trace.py): launch offsets (N, 2) in units of the source's beam radii that
    # replace the equal-power rings; a Project never sets it, so a run's trace and its hashes do not depend on it.
    pupil_layout: Array | None = None
    # A pulsed source's pulse train: (repetition rate Hz, pulse duration s, FWHM of the intensity); None: continuous. Only a
    # crystal's second-harmonic generation reads it here (its conversion of a pulse is out of model scope).
    pulse: tuple[float, float] | None = None


def line_coherence_length_mm(src: Any, index: int) -> float | None:
    """The coherence length (mm) of spectral line ``index`` of a source (``SourceG`` / ``EmitterG``; ``None``: undefined). A
    line a crystal generates (``index`` at or beyond the source's line count: the second harmonic of line ``index - count``,
    ``Tracer._second_harmonic``) has half its fundamental's: the harmonic field ``E^2`` carries twice the fundamental's phase
    fluctuations, so in the Gaussian-decay model ``|gamma_2(tau)| = |gamma_1(tau)|^4`` (the line's frequency width doubles,
    its wavelength width halves). This reads the stated line as one longitudinal mode with Gaussian phase noise; a multimode
    or chaotic line (``|gamma_2| = |gamma_1|^2``) is not modelled."""

    lengths = src.coherence_lengths_mm
    if lengths is None:
        return None
    count = len(src.spectrum)
    if index < count:
        return lengths[index]
    fundamental = line_coherence_length_mm(src, index - count)
    return None if fundamental is None else fundamental / 2.0


@dataclass(frozen=True)
class EmitterG:
    """A finite planar emitter (``ExtendedEmitter``); its samples are traced apart from the Gaussian sources
    (``optics/rays/sampled.py``). The last five fields are what a traced packet reads of its source: an emitter has no
    coherence, no pupil rings and no polarization components (its light is unpolarized as a whole)."""

    source_id: str
    owner_id: str
    position: Array
    direction: Array  # the emitter's normal: +z of its frame in global coordinates
    e1: Array  # x of its frame
    e2: Array  # y of its frame
    spectrum: tuple[tuple[float, float], ...]  # (wavelength_nm, normalized weight)
    power_W: float
    medium: str
    shape: Literal["rectangle", "disc"]  # (a disc of radius 0 is a point)
    half_x: float  # half width along x, or the disc's radius
    half_y: float  # half height along y, or the disc's radius
    law: Literal["LAMBERTIAN", "ISOTROPIC", "COSINE_POWER", "UNIFORM_CONE", "COLLIMATED"]
    sample_count: int
    seed: int
    coherence_lengths_mm: tuple[float, ...] | None = None
    coherence_group: str = ""
    ring_count: int = 0
    points_per_ring: int = 0
    polarization_component: int | None = None
    # COSINE_POWER: the exponent n of the intensity law cos^n (the scattered light of a surface, optics/rays/scatter.py)
    exponent: float = 1.0
    # UNIFORM_CONE: the cone's half angle in degrees and its sine (libm-free: math/exact_trig)
    cone_half_angle_deg: float = 0.0
    cone_sine: float = 0.0
    # a sampled pattern (optics/fields/source_fields.py): an intensity map (positions by its distribution) or a coherent field
    # (positions by |U|^2, directions along its local wavevector, its polarization); None: uniform over the shape
    pattern: PatternG | None = None


@dataclass(frozen=True)
class NonlinearG:
    """A uniaxial crystal that generates the second harmonic of light entering it (``UniaxialMaterial.nonlinear``): the
    stated interaction, the crystal frame (rows X, Y, Z in the scene frame; ``None`` unless a ``d`` tensor is stated), its two
    parallel plane faces (scene indices) and the slab's thickness between them. ``out_of_scope``: why the crystal is not
    such a slab (then no conversion is computed in it: each pass reports it out of model scope)."""

    spec: Any
    frame: Array | None
    faces: tuple[int, int]
    thickness_mm: float
    out_of_scope: str | None = None


@dataclass(frozen=True)
class GradientFaceG:
    """An end face of a gradient-index rod in the rod's local frame (transverse ``x``, ``y``, axial ``z`` from its origin): a
    plane (``kind`` ``"plane"``: through ``point`` with unit ``normal``, ``normal[2] > 0``) or a conic surface of revolution
    about the rod's axis (``"conic"``: vertex at ``point``, on the axis, vertex curvature ``curvature``, conic constant
    ``conic``); ``z_min`` / ``z_max`` its axial extent within the rod's radius and ``slope`` its largest ``|dz/dr|`` there
    (``optics/rays/gradient.py`` ``to_face``)."""

    kind: Literal["plane", "conic"]
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    z_min: float
    z_max: float
    slope: float
    curvature: float = 0.0
    conic: float = 0.0


@dataclass(frozen=True)
class GradientG:
    """A gradient-index medium in the scene (``contracts.materials.GradientIndexMaterial``, ``AxialGradientIndexMaterial``): its
    axis through ``origin`` along ``axis`` (unit) with transverse unit axes ``ex``, ``ey``, the rod's ``radius``, and its two end
    faces: the scene indices of the face at the low and at the high end along the axis and their axial positions (their
    vertices on the axis). ``faces``: both faces in the rod's frame (``GradientFaceG``) when either is not a plane across the
    axis (a slanted or curved face); ``None`` when both are planes across it."""

    medium: str
    origin: Array
    axis: Array
    ex: Array
    ey: Array
    radius: float
    face_lo: int
    face_hi: int
    z_lo: float
    z_hi: float
    faces: tuple[GradientFaceG, GradientFaceG] | None = None


@dataclass
class Scene:
    """Everything the tracer needs; built once per validation call."""

    surfaces: list[SurfaceG] = field(default_factory=list)
    sources: list[SourceG] = field(default_factory=list)
    emitters: list[EmitterG] = field(default_factory=list)
    surface_index: dict[str, int] = field(default_factory=dict)
    # the unit optic axis (scene frame) of every uniaxial crystal medium of the scene, by material id
    optic_axes: dict[str, Array] = field(default_factory=dict)
    # every radial gradient-index medium of the scene, by material id
    gradients: dict[str, GradientG] = field(default_factory=dict)
    # notices about the scene's geometry found while lowering it (``ValidationIssue``, NOT_APPLICABLE): a face fitted to
    # facets of rounded coordinates (FACET_FACE_FITTED)
    notices: list[Any] = field(default_factory=list)
    # the crystals that generate a second harmonic (``NonlinearG``), by material id
    nonlinear: dict[str, NonlinearG] = field(default_factory=dict)
