"""Build ``opensci-linear-optical-network/v1`` from an Engine ``ProjectSnapshot`` (pure Python + numpy, no simulator).

Method (one spectral line, ``wavelength_nm``):

1. For every exported source, two *probe* traces of the Engine's own tracer (``validation.pipeline.run_physics``): only that
   source, only the selected line, 1 W, unlimited coherence length, launched once lab-H and once lab-V polarized
   (``basis.lab_basis``). Each arrival on an output plane carries a flux-normalized Jones vector in its 3D basis, the
   chief-ray optical path length and a path id; its 3D field projected onto the output beam's lab basis gives one column of a
   2 x 2 path amplitude matrix.
2. Arrivals on one output plane are grouped into spatial modes by the normalized overlap of their Gaussian modes (curvature,
   tilt, offset); a pair that neither coincides nor is orthogonal is refused (``PARTIAL_MODE_OVERLAP``).
3. ``T`` = sum over paths of ``amplitudes * exp(i k0 L_ref) * exp(i k0 (OPL - L_ref))``: ``L_ref`` is the output beam's
   reference optical path length (its first path's), rounded to 1e-9 mm, a phase common to the beam's rows; each path's
   offset ``OPL - L_ref`` is rounded to 1e-12 mm, so the relative phases of the paths that interfere keep ~1e-8 rad and a
   lossless layout stays unitary. Lengths and amplitudes are quantized *before* the phases are formed and every phase is
   reduced modulo one wave with exact decimal arithmetic, so runtimes whose float results agree (CPython, Pyodide) get the
   same quantized entries.
4. Classical-limit check per source: a third trace with the source's declared polarization, and the two probes; the Engine's
   coherent superposition on every output plane (``validation/superposition.py``, a closed-form whole-plane integral) must
   equal ``sum |T J|^2`` of that plane's modes for each of the three, and the power ledger must close. It validates the
   export's mode grouping, projections and phase assembly against the Engine, not the Engine's physics (both sides come from
   the same trace).
5. Per-element 2 x 2 matrices from an observer of the probe traces (``recorder.ElementRecorder``).

Anything the network cannot describe faithfully raises ``NetworkExportError`` (structured problems); a partial network is
never returned. A finding refuses for what the matrix depends on: the surfaces on the lineage of a crossing of an exported
output plane, those planes and the exported sources; an out-of-model finding elsewhere (light no exported output receives) is
listed in ``validity.warnings``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, localcontext

import numpy as np

from .._version import ENGINE_VERSION, NUMERICAL_CONFIG_VERSION
from ..config import NumericalConfig
from ..contracts.network import (
    NETWORK_SCHEMA,
    ClassicalCheckV1,
    ColumnCheckV1,
    ElementTransferV1,
    GaussianModeV1,
    HintNoteV1,
    InputModeV1,
    LinearOpticalNetworkV1,
    NetworkExportOptionsV1,
    NetworkHintsV1,
    NetworkPathV1,
    NetworkProvenanceV1,
    NetworkValidityV1,
    OutputModeV1,
    PathPairHintV1,
    PlaneCheckV1,
    TransferMatrixV1,
    ValidityIssueV1,
)
from ..contracts.components import SpectralDensity
from ..contracts.materials import ConstantIndexMaterial
from ..contracts.polarization import JonesVector
from ..math.exact_linalg import abs2, cvdot, inverse_2x2, singular_values
from ..math.exact_trig import cis
from ..contracts.project import ProjectSnapshot
from ..errors import EngineInputError, ErrorCode, Status
from ..manifest import engine_build_id, hash_exact
from ..optics.interference.spectrum import spectrum_summary
from ..optics.materials.group import group_index
from ..optics.materials.library import BUILTIN_MATERIALS, MaterialLibrary
from ..optics.polarization.phases import coated_interface, splitter_amplitudes
from ..optics.rays.tracer import ArrivalRecord
from ..scene.lower import lower_scene
from ..scene.model import SourceG
from ..units import NM_TO_MM
from ..validation.pipeline import BEAMS_ALONE, PhysicsOutcome, run_physics
from ..validation.superposition import FULL_CAPTURE_FRACTION, superpositions
from .basis import lab_basis, near_switch, uses_fallback
from .conventions import (
    AMPLITUDE_DIGITS,
    AMPLITUDE_QUANTUM,
    CHECK_FLOOR,
    CHECK_TOLERANCE,
    COARSE_DIGITS,
    DIRECTION_DIGITS,
    ELEMENT_DIGITS,
    ENTRY_DIGITS,
    ENTRY_QUANTUM,
    FRACTION_DIGITS,
    LENGTH_DIGITS,
    LOSSLESS_TOLERANCE,
    OPL_DIGITS,
    OPL_OFFSET_DIGITS,
    OPL_OFFSET_QUANTUM,
    PROBE_COHERENCE_LENGTH_MM,
    RATIO_DIGITS,
    SIGNIFICANT_DIGITS,
    conventions,
)
from .errors import NetworkErrorCode, NetworkExportError, NetworkProblem
from .recorder import ElementEvent, ElementRecorder

#: Speed of light in mm/ps (delay hints).
C_MM_PER_PS = 0.299792458
#: Sinks that mean the trace did not follow every photon (the network would be incomplete).
INCOMPLETE_SINKS = ("PATH_LIMIT", "INTERACTION_LIMIT", "TERMINATED_OUT_OF_MODEL", "SOURCE_NOT_TRACED")
#: Power (fraction of the 1 W probe) below which paths the tracer ended at its power floor are negligible.
POWER_FLOOR_ALLOWANCE = 1e-9
#: A lineage that meets one surface this many times inside an incomplete trace is a resonator.
CAVITY_VISITS = 3
#: Relative singular-value floor below which the two probe polarizations reach an element in dependent states.
DEPENDENT_INPUT_RATIO = 1e-6
#: Decimal digits of the exact phase reduction (an OPL of 1e6 mm in units of 1e-9 mm needs 15; ample margin).
PHASE_DECIMAL_PRECISION = 60
#: Ledger entries below this (fraction of the 1 W probe) are roundoff of the output-power subtraction.
LEDGER_FLOOR = 1e-12
#: A polarization-dependent coating met closer than this to normal incidence has no physical s axis (POLARIZING_NORMAL_INCIDENCE):
#: the plane of incidence is then set by a misalignment or, at exactly normal incidence, by the Engine's internal transverse
#: axis, and a coating with constant R_s != R_p (real coatings converge to R_s = R_p there) is not a physical description.
POLARIZING_NORMAL_INCIDENCE_DEG = 1.0
#: R_s / R_p (T_s / T_p) differing by more than this make a coating polarization-dependent.
POLARIZING_COATING_TOLERANCE = 1e-9
#: An element's own 2 x 2 scattering matrix (both sides) may exceed a singular value of 1 by this much (roundoff of R, T and
#: the phase factors) before it counts as creating light.
PASSIVITY_TOLERANCE = 1e-9
#: Lineage events of coated interactions whose coating may depend on polarization.
COATED_EVENTS = ("split_reflect", "split_transmit", "reflect", "refract")
#: Power-ledger sink of the Gaussian power an aperture the chief ray passes cuts off.
APERTURE_TRUNCATION_SINK = "APERTURE_TRUNCATION"
#: Per lowered interaction kind: ``None`` where the network states the phase of light it passes or reflects, else the
#: refusal of a path that meets it (its model states no optical path length): a thin lens (ideal or tabulated, its rays'
#: ``ray_opl_defined`` false), a neutral-density attenuator (``ATTENUATOR_EVENTS`` of the lineage) and a Faraday rotator
#: (``FARADAY_EVENT`` of the lineage, ``ray_opl_defined`` false behind it too), and a diffraction grating (``GRATING_EVENTS`` of the
#: lineage, likewise). An absorber and an
#: aperture stop's plate end the light they stop; what passes an aperture stop's opening keeps its phase. An exported path
#: that meets a kind not listed here is refused (``OUT_OF_MODEL_SCOPE``): a new interaction states its phase model here first.
INTERACTION_PHASE: dict[str, NetworkErrorCode | None] = {
    "refract": None, "mirror": None, "beam_splitter": None, "polarizer": None, "retarder": None, "absorber": None,
    "aperture_stop": None, "observe": None,
    "thin_lens": NetworkErrorCode.PHASE_UNDEFINED_IDEAL_LENS, "neutral_density": NetworkErrorCode.PHASE_UNDEFINED_ATTENUATOR,
    "faraday_rotator": NetworkErrorCode.PHASE_UNDEFINED_FARADAY_ROTATOR,
    "diffraction_grating": NetworkErrorCode.PHASE_UNDEFINED_GRATING,
    "complex_transmission": NetworkErrorCode.PHASE_UNDEFINED_SAMPLE,
}
#: The input polarizations of the classical check.
CHECK_INPUTS = ("declared", "H", "V")
#: The segments gathered for a path's group optical path must reproduce its optical path length within this (mm): the check
#: that they are the segments the trace summed.
GROUP_PATH_TOLERANCE_MM = 1e-6
#: A child path starts where its parent's segment ends on the branch surface, within this distance (mm).
BRANCH_POINT_TOLERANCE_MM = 1e-6
#: Characters of an unexpected exception's text kept in a NUMERICAL_FAILURE message.
EXCEPTION_TEXT_CHARS = 200


def _q(value: float, digits: int) -> float:
    """Round to ``digits`` decimals (exact decimal rounding, identical on every runtime); -0.0 becomes 0.0."""
    return round(float(value), digits) + 0.0


def _qv(vector, digits: int = DIRECTION_DIGITS) -> tuple[float, ...]:
    return tuple(_q(x, digits) for x in vector)


def _qc(z: complex, digits: int) -> tuple[float, float]:
    return (_q(z.real, digits), _q(z.imag, digits))


def _sig(value: float) -> float:
    """``SIGNIFICANT_DIGITS`` significant digits (wavelengths, line powers, coherence lengths)."""
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _arrival_opl(arrival) -> float:
    """The chief-ray optical path length of an output crossing (the same in both probes)."""
    return arrival.opl.get("H", arrival.opl.get("V"))


def _gram_bound(t: np.ndarray, error: np.ndarray) -> float:
    """Largest change of an entry of T^H T that entry errors up to ``error`` (elementwise bound) can cause:
    |(T + E)^H (T + E) - T^H T|_jk <= sum_i (e_ij |T_ik| + |T_ij| e_ik + e_ij e_ik)."""
    if not t.size:
        return 0.0
    a = np.sqrt(abs2(t))  # |T_ij| without hypot (identical bits in every runtime)
    return float(np.max(error.T @ a + a.T @ error + error.T @ error))


def _phase_factor(opl_q: float, wavelength_nm: float) -> complex:
    """``exp(i k0 OPL)`` of an already quantized OPL, reduced modulo one wave in exact decimal arithmetic."""
    with localcontext() as ctx:
        ctx.prec = PHASE_DECIMAL_PRECISION
        waves = Decimal(repr(opl_q)) / (Decimal(repr(wavelength_nm)) * Decimal("1e-6"))
        fraction = waves - waves.to_integral_value(rounding="ROUND_FLOOR")
    return cis(2.0 * math.pi * float(fraction))  # libm-free exp(i angle): identical bits in every runtime


# ---------------------------------------------------------------------------------------------------- mode overlap


def _mode_matrix(rec: ArrivalRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(M, kappa, c): the beam on the plane as exp(-(p-c)^T M (p-c) + i kappa.(p-c)) in plane coordinates (u, v)."""
    g = rec.gstate
    k = 2.0 * math.pi * rec.n / (rec.wavelength_nm * NM_TO_MM)
    u, v = rec.plane_u, rec.plane_v
    m = np.zeros((2, 2), dtype=complex)
    for axis, q in ((g.bx, g.qx), (g.by, g.qy)):
        b = np.array([float(axis @ u), float(axis @ v)])
        m = m + (1j * k / (2.0 * q)) * np.outer(b, b)
    kappa = k * np.array([float(rec.direction @ u), float(rec.direction @ v)])
    return m, kappa, np.array(rec.uv, dtype=float)


def mode_overlap(a: ArrivalRecord, b: ArrivalRecord) -> float:
    """Normalized spatial mode overlap ``|<a|b>|^2 / (<a|a><b|b>)`` of two beams on one plane (Gaussian modes with curvature,
    tilt and offset; closed-form integral over the whole plane)."""
    ma, ka, ca = _mode_matrix(a)
    mb, kb, cb = _mode_matrix(b)
    big_a = np.conj(ma) + mb
    beta = 2.0 * (np.conj(ma) @ ca) + 2.0 * (mb @ cb) - 1j * ka + 1j * kb
    gamma = -(ca @ np.conj(ma) @ ca) - (cb @ mb @ cb) + 1j * float(ka @ ca) - 1j * float(kb @ cb)
    det = complex(np.linalg.det(big_a))
    if det == 0:
        return 0.0
    exponent = complex(beta @ np.linalg.solve(big_a, beta)) / 4.0 + complex(gamma)
    log_abs_ab = exponent.real + math.log(math.pi) - 0.5 * math.log(abs(det))
    norm_a = math.pi / math.sqrt(float(np.linalg.det(2.0 * ma.real)))
    norm_b = math.pi / math.sqrt(float(np.linalg.det(2.0 * mb.real)))
    return float(min(1.0, math.exp(2.0 * log_abs_ab - math.log(norm_a) - math.log(norm_b))))


# ---------------------------------------------------------------------------------------------------- probes


@dataclass
class _Line:
    src: SourceG
    index: int
    wavelength_nm: float
    power_W: float
    basis: tuple[np.ndarray, np.ndarray]
    coherence_length_mm: float | None


@dataclass
class _Probe:
    line: _Line
    polarization: str
    outcome: PhysicsOutcome
    events: list[ElementEvent]


@dataclass
class _Arrival:
    """One output crossing seen by the probes of one source (geometry identical in both)."""

    line: _Line
    record: ArrivalRecord  # the H probe's record when it has one
    amplitudes: dict[str, np.ndarray] = field(default_factory=dict)  # input polarization -> 3D field
    opl: dict[str, float] = field(default_factory=dict)
    group_opl: dict[str, float | None] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.record.observation_id, self.record.path_id, self.record.crossing_index)


def _probe_project(project: ProjectSnapshot, line: _Line, polarization: JonesVector, *, coherent: bool) -> ProjectSnapshot:
    """Only ``line``'s source, only that line, 1 W, the given polarization; other sources removed (their surfaces kept). A
    continuous spectrum's line is traced as that line alone (no ``spectral_density``): the probe is one frequency, as the
    network is."""
    components = []
    for comp in project.components:
        if comp.source is None:
            components.append(comp)
            continue
        if comp.instance_id != line.src.owner_id or comp.source.source_id != line.src.source_id:
            if comp.surfaces:
                components.append(comp.model_copy(update={"source": None}))
            continue
        update: dict = {"wavelength_nm": line.wavelength_nm, "spectrum": (), "spectral_density": None, "power_W": 1.0,
                        "polarization": polarization}
        if coherent:
            update.update(coherence_length_mm=PROBE_COHERENCE_LENGTH_MM, linewidth_nm=None)
        components.append(comp.model_copy(update={"source": comp.source.model_copy(update=update)}))
    return project.model_copy(update={"components": tuple(components), "targets": ()})


def lab_polarized_projects(project: ProjectSnapshot, cfg: NumericalConfig) -> list[ProjectSnapshot]:
    """``project`` with every source lab-H polarized, and with every source lab-V polarized (``basis.lab_basis`` of each
    source's direction): the field on any path is a linear combination of these two, so a published model's validity checked
    on both runs covers every input polarization (a path no polarization feeds carries no power in either)."""
    library = MaterialLibrary(project.materials, cfg)
    try:
        scene, _poses = lower_scene(project, cfg, library)
    except EngineInputError as exc:
        raise NetworkExportError([NetworkProblem(NetworkErrorCode.ENGINE_INPUT_INVALID, exc.message, exc.location)]) from None
    sources = {(s.owner_id, s.source_id): s for s in scene.sources}
    variants = []
    for which in (0, 1):
        components = []
        for comp in project.components:
            src = sources.get((comp.instance_id, comp.source.source_id)) if comp.source is not None else None
            if src is not None:
                comp = comp.model_copy(update={"source": comp.source.model_copy(
                    update={"polarization": _jones_for(lab_basis(src.direction)[which], src)})})
            components.append(comp)
        variants.append(project.model_copy(update={"components": tuple(components)}))
    return variants


def export_scope_projects(project: ProjectSnapshot, cfg: NumericalConfig, options: NetworkExportOptionsV1) -> list[ProjectSnapshot]:
    """The runs a published model's validity is checked on before a network export: only the light the export describes.

    The export traces each exported line on its own (``_Exporter.lines``: the ``source_ids`` selection, then the one line at
    ``wavelength_nm`` of each selected source). So the check runs ``project`` restricted to that selection -- each exported
    source emitting only its exported line (its line power, its declared polarization; a continuous spectrum's line alone,
    without its ``spectral_density``), every other source removed (its
    surfaces kept, as in a probe), targets dropped (they judge a run, not the optics) -- and that restricted Project with every
    exported source lab-H and lab-V polarized (``lab_polarized_projects``). A source the export does not cover (not selected, or
    without a line at the exported wavelength) cannot take the export out of a model's scope; an exported line outside a
    model's characterization still does. Output planes (``observation_ids``) do not narrow the check: the light is traced
    through every element whatever plane it ends on, and every element it meets is in the network (``elements``).

    Raises ``NetworkExportError`` for a selection the export refuses (``OPTIONS_INVALID``, ``SOURCES_MISSING``,
    ``MIXED_WAVELENGTHS``, ``WAVELENGTH_NOT_EMITTED``, ...), exactly as the export itself would."""

    lines, _wavelength, _outputs = _Exporter(project, cfg, options).lines()
    exported = {(line.src.owner_id, line.src.source_id): line for line in lines}
    components = []
    for comp in project.components:
        if comp.source is None:
            components.append(comp)
            continue
        line = exported.get((comp.instance_id, comp.source.source_id))
        if line is None:
            if comp.surfaces:
                components.append(comp.model_copy(update={"source": None}))
            continue
        components.append(comp.model_copy(update={"source": comp.source.model_copy(
            update={"wavelength_nm": line.wavelength_nm, "spectrum": (), "spectral_density": None, "power_W": line.power_W})}))
    restricted = project.model_copy(update={"components": tuple(components), "targets": ()})
    return [restricted, *lab_polarized_projects(restricted, cfg)]


def _jones_for(vector: np.ndarray, src: SourceG) -> JonesVector:
    return JonesVector(ex=(float(vector @ src.e1), 0.0), ey=(float(vector @ src.e2), 0.0))


def _source_lab_polarization(src: SourceG, basis: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    field3 = src.jones[0] * src.e1 + src.jones[1] * src.e2
    norm = math.sqrt(cvdot(field3, field3).real)
    field3 = field3 / norm if norm > 0 else field3
    return np.array([complex(basis[0] @ field3), complex(basis[1] @ field3)])


# ---------------------------------------------------------------------------------------------------- export


class _Exporter:
    def __init__(self, project: ProjectSnapshot, cfg: NumericalConfig, options: NetworkExportOptionsV1) -> None:
        self.project, self.cfg, self.options = project, cfg, options
        self.problems: list[NetworkProblem] = []
        self.material_models: dict | None = None
        self.warnings: dict[tuple[str, str | None], ValidityIssueV1] = {}
        self.notes: dict[tuple[str, str | None], HintNoteV1] = {}
        self.constant_index_media: set[str] = set()
        self.scope_issues: list = []

    # -- refusals ---------------------------------------------------------------------------------------------------
    def refuse(self, code: NetworkErrorCode, message: str, location: str | None = None, *, observation_id: str | None = None) -> None:
        problem = NetworkProblem(code, message, location, observation_id=observation_id)
        if problem not in self.problems:
            self.problems.append(problem)

    def stop_if_refused(self) -> None:
        if self.problems:
            raise NetworkExportError(self.problems)

    def note(self, code: str, message: str, location: str | None = None) -> None:
        self.notes.setdefault((code, location), HintNoteV1(code=code, message=message, location=location))

    # -- scene and lines --------------------------------------------------------------------------------------------
    def lines(self) -> tuple[list[_Line], float, list[str]]:
        library = MaterialLibrary(self.project.materials, self.cfg)
        try:
            scene, _poses = lower_scene(self.project, self.cfg, library)
        except EngineInputError as exc:
            raise NetworkExportError([NetworkProblem(NetworkErrorCode.ENGINE_INPUT_INVALID, exc.message, exc.location)]) from None
        self.scene = scene
        # surfaces the export has no phase model of: refused when an exported path meets them (``lineage_problems``)
        self.unstated = {s.surface_id: s.interaction.kind for s in scene.surfaces if s.interaction.kind not in INTERACTION_PHASE}
        self.crystal_faces = {s.surface_id for s in scene.surfaces if {s.medium_pos, s.medium_neg} & set(scene.optic_axes)}
        sources = list(scene.sources)
        known = [s.source_id for s in sources]
        if self.options.source_ids is not None:
            unknown = [s for s in self.options.source_ids if s not in known]
            if unknown:
                raise NetworkExportError([NetworkProblem(NetworkErrorCode.OPTIONS_INVALID,
                                                         f"source_ids {unknown} are not sources of the Project; sources: {known}")])
            sources = [s for s in sources if s.source_id in self.options.source_ids]
        partial = [s.source_id for s in sources if s.degree_of_polarization is not None]
        if partial:
            raise NetworkExportError([NetworkProblem(
                NetworkErrorCode.OUT_OF_MODEL_SCOPE,
                f"source {source_id} emits partially polarized light: the network holds the coherent amplitudes of one "
                "polarization state, and the two incoherent polarization components of this light have no common network; "
                "export a fully polarized source", f"source:{source_id}") for source_id in partial])
        if not sources and scene.emitters:
            raise NetworkExportError([NetworkProblem(NetworkErrorCode.OUT_OF_MODEL_SCOPE,
                                                     "the network holds coherent amplitudes of Gaussian beams; the rays of an extended "
                                                     "emitter have no phase relation, so its light has no network to export")])
        if not sources:
            raise NetworkExportError([NetworkProblem(NetworkErrorCode.SOURCES_MISSING, "the Project has no source to export")])
        observation_ids = [o.observation_id for o in self.project.observations]
        outputs = list(self.options.observation_ids) if self.options.observation_ids is not None else observation_ids
        unknown = [o for o in outputs if o not in observation_ids]
        if unknown:
            raise NetworkExportError([NetworkProblem(NetworkErrorCode.OPTIONS_INVALID,
                                                     f"observation_ids {unknown} are not observation planes of the Project; planes: {observation_ids}")])
        if not outputs:
            raise NetworkExportError([NetworkProblem(NetworkErrorCode.OUTPUTS_MISSING,
                                                     "the Project has no observation plane: place one on every output port of the network")])
        tol = self.cfg.interference_wavelength_tol_nm
        wavelengths: list[float] = []
        for src in sources:
            for wl, _weight in src.spectrum:
                if all(abs(wl - w) > tol for w in wavelengths):
                    wavelengths.append(wl)
        wavelengths.sort()
        wanted = self.options.wavelength_nm
        if wanted is None:
            if len(wavelengths) > 1:
                raise NetworkExportError([NetworkProblem(
                    NetworkErrorCode.MIXED_WAVELENGTHS,
                    f"the sources emit {len(wavelengths)} wavelengths ({', '.join(f'{w:g}' for w in wavelengths)} nm); a network describes "
                    "one wavelength: choose one with the wavelength_nm option")])
            wanted = wavelengths[0]
        densities = {(c.instance_id, c.source.source_id): c.source.spectral_density for c in self.project.components
                     if c.source is not None and c.source.spectral_density is not None}
        lines: list[_Line] = []
        for src in sources:
            matches = [(j, wl, weight) for j, (wl, weight) in enumerate(src.spectrum) if abs(wl - wanted) <= tol]
            if not matches:
                self.note("SOURCE_WITHOUT_LINE", f"source {src.source_id} emits no line at {wanted:g} nm and has no input mode",
                          f"source:{src.source_id}")
                continue
            j, wl, weight = matches[0]
            lengths = src.coherence_lengths_mm
            density = densities.get((src.owner_id, src.source_id))
            length = None if lengths is None else lengths[j]
            if density is not None:
                length = self.spectrum_coherence_length(src.source_id, density, wl)
            lines.append(_Line(src, j, wl, src.power_W * weight, lab_basis(src.direction), length))
        if not lines:
            raise NetworkExportError([NetworkProblem(NetworkErrorCode.WAVELENGTH_NOT_EMITTED,
                                                     f"no exported source emits {wanted:g} nm; the sources emit "
                                                     f"{', '.join(f'{w:g}' for w in wavelengths)} nm")])
        return lines, lines[0].wavelength_nm, outputs

    def spectrum_coherence_length(self, source_id: str, density: SpectralDensity, wavelength_nm: float) -> float | None:
        """The coherence length of a source stating a continuous spectrum: its spectral density's (``spectrum_summary``, the
        value the Engine reports in ``source_spectra`` and uses for the source's two-beam pairs), not the exported quadrature
        line's -- that line's Gaussian width is the quadrature bin, so its coherence length grows with the number of lines and
        is not a property of the light. ``None`` (with a note saying why) when ``|gamma|`` stays above 1/e over the scan."""
        summary = spectrum_summary(source_id, density)
        where = f"source:{source_id}"
        self.note("CONTINUOUS_SPECTRUM",
                  f"source {source_id} states a continuous spectrum sampled by {summary.line_count} quadrature lines of "
                  f"{summary.line_width_nm:.6g} nm: the network is that of its line at {wavelength_nm:.9g} nm, line_power_W is that "
                  "line's share of the source's power, and coherence_length_mm (input modes and path pairs) is the spectrum's: the "
                  "smallest path difference at which the degree of coherence of the declared spectral density falls to 1/e", where)
        if summary.coherence_length_mm is None:
            self.note("COHERENCE_BEYOND_SCAN",
                      f"source {source_id}: the degree of coherence of its spectral density stays above 1/e over the scanned "
                      "path differences, so its coherence length is not stated (coherence_length_mm is null)", where)
        return summary.coherence_length_mm

    def run_probe(self, line: _Line, polarization: str) -> _Probe:
        """A lab-H or lab-V trace: the amplitudes of one input column, and that column's classical-limit check (coherent, with
        the pupil rays the Engine's superposition and Gaussian checks use, like the declared-polarization check trace)."""
        vector = line.basis[0] if polarization == "H" else line.basis[1]
        project = _probe_project(self.project, line, _jones_for(vector, line.src), coherent=True)
        recorder = ElementRecorder()
        outcome = run_physics(project, self.cfg, plan=BEAMS_ALONE, recorder=recorder)
        return _Probe(line, polarization, outcome, recorder.events)

    def run_check(self, line: _Line) -> PhysicsOutcome:
        project = _probe_project(self.project, line, JonesVector(ex=tuple(map(float, (line.src.jones[0].real, line.src.jones[0].imag))),
                                                                 ey=(float(line.src.jones[1].real), float(line.src.jones[1].imag))),
                                 coherent=True)
        return run_physics(project, self.cfg, plan=BEAMS_ALONE)

    # -- trace completeness -----------------------------------------------------------------------------------------
    def check_trace(self, outcome: PhysicsOutcome, line: _Line) -> None:
        where = f"source:{line.src.source_id}"
        budget = outcome.power_budget
        sinks = budget.lines[0].sinks if budget is not None and budget.lines else {}
        incomplete = {kind: sinks.get(kind, 0.0) for kind in INCOMPLETE_SINKS if sinks.get(kind, 0.0) > 0.0}
        floor = sinks.get("BELOW_POWER_FLOOR", 0.0)
        if floor > POWER_FLOOR_ALLOWANCE:
            incomplete["BELOW_POWER_FLOOR"] = floor
        if incomplete:
            cavity = None
            for path in outcome.trace.paths:
                visits: dict[str, int] = {}
                for entry in path.lineage:
                    surface, _sep, event = entry.rpartition(":")
                    if event != "observe":
                        visits[surface] = visits.get(surface, 0) + 1
                hit = [s for s, n in visits.items() if n >= CAVITY_VISITS]
                if hit:
                    cavity = sorted(hit)[0]
                    break
            fractions = ", ".join(f"{kind} {power:.3g}" for kind, power in sorted(incomplete.items()))
            if cavity is not None:
                self.refuse(NetworkErrorCode.CAVITY_NOT_SUPPORTED,
                            f"the light returns to {cavity} again and again (a resonator): the network of a cavity is an infinite series "
                            f"the trace cuts short ({fractions} of the power not followed); export the layout without the cavity", where)
            else:
                self.refuse(NetworkErrorCode.TRACE_INCOMPLETE,
                            f"the trace did not follow every beam to its end ({fractions} of the power); the network would miss paths", where)
        if budget is not None and not budget.conserved:
            self.refuse(NetworkErrorCode.NETWORK_NOT_CLOSED, "the power ledger of the trace does not close", where)

    def check_scope(self, outcome: PhysicsOutcome, line: _Line) -> None:
        for issue in outcome.issues:
            location = issue.location
            if issue.status is Status.OUT_OF_MODEL_SCOPE:
                self.scope_issues.append(issue)  # judged against the exported paths (``scope_problems``)
            elif issue.code is ErrorCode.BULK_LOSS_NOT_ESTABLISHED:
                # The trace retains geometry and a Jones *upper envelope*, but not the field's actual amplitude.
                # Network v1 requires a single numeric T, column transmission and a PASS classical power check;
                # exporting that envelope as measured transmission would silently assume zero bulk absorption.
                self.refuse(NetworkErrorCode.OUT_OF_MODEL_SCOPE,
                            f"{issue.code.value}: {issue.message}; a single transfer matrix cannot state the bounded "
                            "output power. Supply bulk-loss data or explicitly assume a scalar bulk absorption coefficient",
                            location)
            elif issue.status is Status.WARNING:
                self.warnings.setdefault((issue.code.value, location), ValidityIssueV1(
                    code=issue.code.value, location=location, message=_WARNING_TEXT.get(issue.code.value, "the Engine reports this warning on a probe trace")))

    # -- what the exported matrix depends on ------------------------------------------------------------------------
    def exported_lineage(self, outcomes: list[PhysicsOutcome], outputs: list[str]) -> dict[str, set[str]]:
        """Per surface: the output planes whose crossings (of the probe and check traces) it lies on the lineage of -- the
        path's own and its ancestors' surfaces up to the crossing. A surface no exported path meets is not a key."""
        output_set = set(outputs)
        planes: dict[str, set[str]] = {}
        for outcome in outcomes:
            for rec in outcome.trace.arrivals:
                if rec.observation_id in output_set:
                    for entry in rec.lineage:
                        surface, _sep, event = entry.rpartition(":")
                        if event != "observe":
                            planes.setdefault(surface, set()).add(rec.observation_id)
        return planes

    def lineage_problems(self, lineage: dict[str, set[str]]) -> list[NetworkProblem]:
        """Surfaces without a phase model of the export that an exported path meets."""
        problems = [NetworkProblem(NetworkErrorCode.OUT_OF_MODEL_SCOPE,
                                   f"surface {surface_id}: the network export has no phase model of a {kind} interaction",
                                   f"surface:{surface_id}", observation_id=_one(lineage[surface_id]))
                    for surface_id, kind in sorted(self.unstated.items()) if surface_id in lineage]
        problems += [NetworkProblem(NetworkErrorCode.OUT_OF_MODEL_SCOPE,
                                    f"surface {surface_id} is a face of a uniaxial crystal: the network export has no phase model of the "
                                    "split into a crystal's ordinary and extraordinary waves", f"surface:{surface_id}",
                                    observation_id=_one(lineage[surface_id]))
                     for surface_id in sorted(self.crystal_faces) if surface_id in lineage]
        return problems

    def scope_problems(self, lineage: dict[str, set[str]], outputs: list[str], lines: list[_Line]) -> None:
        """An out-of-model finding of a probe or check trace refuses the export when it is located on a surface an exported
        path meets, on an output plane or at an exported source, or cannot be located on one of them; any other (light no
        exported output receives) is listed in ``validity.warnings``."""
        sources = {line.src.source_id for line in lines}
        for issue in self.scope_issues:
            location, code = issue.location, issue.code.value
            kind, _sep, name = (location or "").partition(":")
            planes: set[str] = set()
            if kind == "surface" and name not in lineage:
                exported = False
            elif kind == "observation" and name not in outputs:
                exported = False
            elif kind == "source" and name not in sources:
                exported = False
            else:
                exported, planes = True, lineage.get(name, set()) if kind == "surface" else set()
            if not exported:
                self.warnings.setdefault((code, location), ValidityIssueV1(
                    code=code, location=location, message="outside the Engine's model on light that reaches no exported output"))
            elif code == ErrorCode.GAUSSIAN_SEVERE_TRUNCATION.value:
                self.refuse(NetworkErrorCode.MODE_TRUNCATED, f"an aperture cuts the beam severely ({code}): the beam is no longer one "
                            "Gaussian mode; enlarge the aperture or reduce the beam", location, observation_id=_one(planes))
            elif code.startswith("GAUSSIAN_"):
                self.refuse(NetworkErrorCode.MODE_UNDEFINED, f"the beam is not a valid Gaussian mode here ({code})", location,
                            observation_id=_one(planes))
            else:
                self.refuse(NetworkErrorCode.OUT_OF_MODEL_SCOPE, f"the Engine reports {code}: outside its model", location,
                            observation_id=_one(planes))

    # -- outputs ----------------------------------------------------------------------------------------------------
    def collect(self, probes: list[_Probe], outputs: list[str]) -> list[_Arrival]:
        found: dict[tuple, _Arrival] = {}
        output_set = set(outputs)
        for probe in probes:
            paths = {p.path_id: p for p in probe.outcome.trace.paths}
            truncated = _truncating_surfaces(probe.outcome)
            crossings: dict[str, int] = {}
            for rec in probe.outcome.trace.arrivals:
                if rec.observation_id in output_set and rec.counted:
                    crossings[rec.path_id] = crossings.get(rec.path_id, 0) + 1
            for rec in probe.outcome.trace.arrivals:
                if rec.observation_id not in output_set:
                    continue
                where = f"observation:{rec.observation_id}"
                if rec.within_acceptance is False:
                    self.refuse(NetworkErrorCode.OUTSIDE_ACCEPTANCE,
                                f"beam path {rec.path_id} crosses output plane {rec.observation_id} at {rec.aoi_deg:.4g} deg, outside the "
                                "plane's acceptance cone: the plane does not count it as an output; align the plane with the beam or "
                                "widen its acceptance", where)
                    continue
                if not rec.counted:
                    continue
                # the Engine marks the Gaussian state of a beam an aperture cut above its warning fraction (a WARNING it keeps,
                # and a branch inherits); the truncating surface is one with that issue where this path or an ancestor lost power
                cut = sorted(_ancestry_surfaces(rec.path_id, paths, truncated)) if rec.gstate.status is Status.WARNING else []
                if cut:
                    self.refuse(NetworkErrorCode.MODE_TRUNCATED,
                                f"the aperture of {', '.join(cut)} cuts more than {self.cfg.gaussian_clip_warn_fraction:.3g} of the power of "
                                f"beam path {rec.path_id}, which reaches output {rec.observation_id}: the beam is no longer one Gaussian "
                                "mode (its mode-projected amplitude is not the amplitude the Engine carries); enlarge the aperture or "
                                "reduce the beam", where)
                path = paths.get(rec.path_id)
                observe = f"observation/{rec.observation_id}:observe"
                if path is not None:
                    # after its output crossing a path may only cross observation planes beside their active areas (or
                    # planes that are not outputs) and leave the layout or end on that (absorbing) output plane
                    after = list(path.lineage[len(rec.lineage):])
                    terminal = (after[:1] == [observe] and all(e.endswith(":observe") for e in after[1:])
                                and crossings[rec.path_id] == 1 and path.termination in ("ESCAPED", "DETECTED"))
                    if not terminal:
                        self.refuse(NetworkErrorCode.OUTPUT_NOT_TERMINAL,
                                    f"beam path {rec.path_id} crosses output plane {rec.observation_id} and goes on to further optics or "
                                    "another output: an output port is where light leaves the network; move the plane or leave it out of "
                                    "the observation_ids option", where)
                        continue
                if rec.detected_power_W is not None and rec.power_W > 0.0 and rec.detected_power_W < rec.power_W * FULL_CAPTURE_FRACTION:
                    self.refuse(NetworkErrorCode.MODE_TRUNCATED,
                                f"the active area of {rec.observation_id} detects only part of beam {rec.path_id}; enlarge it or remove it", where)
                gratings = _grating_crossings(rec.lineage)
                rotators = _faraday_crossings(rec.lineage)
                if gratings:
                    self.refuse(NetworkErrorCode.PHASE_UNDEFINED_GRATING,
                                f"beam path {rec.path_id} crosses the diffraction grating {', '.join(gratings)}: its model states the orders' "
                                "directions and at most their powers, not their phase; a network with its phase cannot be formed from it",
                                where)
                elif rotators:
                    self.refuse(NetworkErrorCode.PHASE_UNDEFINED_FARADAY_ROTATOR,
                                f"beam path {rec.path_id} crosses the Faraday rotator {', '.join(rotators)}: its model states a rotation and a "
                                "transmittance, not the optical path through its body; a network with its phase cannot be formed from it",
                                where)
                elif not rec.ray_opl_defined:
                    self.refuse(NetworkErrorCode.PHASE_UNDEFINED_IDEAL_LENS,
                                f"beam path {rec.path_id} crosses an ideal or tabulated thin lens, whose optical path length (phase) is not modelled; "
                                "use a lens with a real prescription", where)
                samples = sorted({e.rsplit(":", 1)[0] for e in rec.lineage if e.endswith(":complex_transmission")})
                if samples:
                    self.refuse(NetworkErrorCode.PHASE_UNDEFINED_SAMPLE,
                                f"beam path {rec.path_id} crosses the complex-transmission sample {', '.join(samples)}: its phase varies "
                                "across the beam, which is not one coefficient of a mode; a network with its phase cannot be formed from it",
                                where)
                attenuators = _attenuator_crossings(rec.lineage)
                if attenuators:
                    self.refuse(NetworkErrorCode.PHASE_UNDEFINED_ATTENUATOR,
                                f"beam path {rec.path_id} {' and '.join(attenuators)}: a neutral-density model states powers only, not "
                                "the optical path through its substrate or the phase of its reflection; model the filter as a plate "
                                "of its substrate (with a coating) to export the network", where)
                if not rec.gstate.valid:
                    self.refuse(NetworkErrorCode.MODE_UNDEFINED, f"beam path {rec.path_id} has no valid Gaussian mode at {rec.observation_id}", where)
                key = (probe.line.src.source_id, rec.observation_id, rec.path_id, rec.crossing_index)
                arrival = found.get(key)
                if arrival is None:
                    arrival = found[key] = _Arrival(probe.line, rec)
                arrival.amplitudes[probe.polarization] = rec.jones[0] * rec.e1 + rec.jones[1] * rec.e2
                arrival.opl[probe.polarization] = rec.opl_mm
                arrival.group_opl[probe.polarization] = self.group_opl(paths, rec)
        return list(found.values())

    def group_opl(self, paths: dict, rec: ArrivalRecord) -> float | None:
        """Group optical path length of the chief ray from the emission point to this output crossing: sum n_g x length over
        its segments (and those of the paths it branched from, up to each branch point); ``None`` when a medium has no group
        index here or the segments do not reproduce the crossing's optical path length."""
        if self.material_models is None:
            self.material_models = {m.material_id: m for m in BUILTIN_MATERIALS}
            self.material_models.update({m.material_id: m for m in self.project.materials})
        segments = _segments_to(paths, rec.path_id, f"observation/{rec.observation_id}", rec.opl_mm)
        if segments is None:
            return None
        total = 0.0
        for segment in segments:
            model = self.material_models.get(segment.medium_id)
            n_g = None if model is None else group_index(model, rec.wavelength_nm, self.cfg)
            if n_g is None or segment.length_mm is None:
                return None
            if isinstance(model, ConstantIndexMaterial):
                self.constant_index_media.add(segment.medium_id)
            total += n_g * segment.length_mm
        return total

    def cluster(self, arrivals: list[_Arrival], outputs: list[str], source_order: dict[str, int]) -> list[tuple[str, int, list[_Arrival]]]:
        """Distinct beams per output plane: [(observation_id, beam index, members)]."""
        tol = self.options.mode_overlap_tolerance
        out: list[tuple[str, int, list[_Arrival]]] = []
        for oid in outputs:
            members = sorted((a for a in arrivals if a.record.observation_id == oid),
                             key=lambda a: (source_order[a.line.src.source_id], a.record.path_id, a.record.crossing_index))
            groups: list[list[_Arrival]] = []
            for a in members:
                placed = False
                for group in groups:
                    head = group[0].record
                    if float(head.direction @ head.plane_normal) * float(a.record.direction @ a.record.plane_normal) <= 0.0:
                        continue  # crossing the plane in opposite directions: different modes
                    eta = mode_overlap(head, a.record)
                    if eta >= 1.0 - tol:
                        group.append(a)
                        placed = True
                        break
                    if eta > tol:
                        self.refuse(NetworkErrorCode.PARTIAL_MODE_OVERLAP,
                                    f"beams {head.path_id} and {a.record.path_id} on {oid} overlap partially (mode overlap {eta:.6f}): they are "
                                    "neither one mode nor two separate modes (a tilt, an offset or a different beam size or curvature); align "
                                    "them or separate them", f"observation:{oid}")
                        placed = True
                        break
                if not placed:
                    groups.append([a])
            # beam order on a plane: forward-crossing beams first, then by their first member
            groups.sort(key=lambda g: (float(g[0].record.direction @ g[0].record.plane_normal) < 0.0,
                                       source_order[g[0].line.src.source_id], g[0].record.path_id, g[0].record.crossing_index))
            out.extend((oid, k, g) for k, g in enumerate(groups))
        return out

    # -- main -------------------------------------------------------------------------------------------------------
    def build(self, provenance: NetworkProvenanceV1) -> LinearOpticalNetworkV1:
        lines, wavelength, outputs = self.lines()
        source_order = {line.src.source_id: i for i, line in enumerate(lines)}
        probes: list[_Probe] = []
        checks: dict[str, PhysicsOutcome] = {}
        for line in lines:
            for pol in ("H", "V"):
                probe = self.run_probe(line, pol)
                self.check_trace(probe.outcome, line)
                self.check_scope(probe.outcome, line)
                probes.append(probe)
            checks[line.src.source_id] = check = self.run_check(line)
            self.check_trace(check, line)
            self.check_scope(check, line)
        lineage = self.exported_lineage([p.outcome for p in probes] + list(checks.values()), outputs)
        unmodelled = self.lineage_problems(lineage)
        if unmodelled:
            raise NetworkExportError(unmodelled)
        self.scope_problems(lineage, outputs, lines)
        self.check_polarizing_normal_incidence(probes, wavelength)
        self.check_element_passivity(probes, wavelength)
        self.stop_if_refused()
        arrivals = self.collect(probes, outputs)
        self.stop_if_refused()
        beams = self.cluster(arrivals, outputs, source_order)
        self.stop_if_refused()

        # -- modes
        input_modes: list[InputModeV1] = []
        for line in lines:
            src = line.src
            source_pol = _source_lab_polarization(src, line.basis)
            self.basis_notes(src.direction, f"source:{src.source_id}")
            for pol, vector in zip(("H", "V"), line.basis):
                input_modes.append(InputModeV1(
                    mode_id=f"in:{src.source_id}#{line.index}:{pol}", source_id=src.source_id, spectral_index=line.index, polarization=pol,
                    position_mm=_qv(src.position, LENGTH_DIGITS), direction=_qv(src.direction), basis_h=_qv(line.basis[0]),
                    basis_v=_qv(line.basis[1]), line_power_W=_sig(line.power_W),
                    source_polarization=(_qc(source_pol[0], AMPLITUDE_DIGITS), _qc(source_pol[1], AMPLITUDE_DIGITS)),
                    coherence_group=src.coherence_group,
                    coherence_length_mm=None if line.coherence_length_mm is None else _sig(line.coherence_length_mm),
                    medium_index=_q(_probe_index(probes, line), RATIO_DIGITS)))
        # every beam's reference optical path length: its first path's (paths order); the paths' phases are taken relative to it
        ordered = sorted(arrivals, key=lambda a: (source_order[a.line.src.source_id], a.record.observation_id, a.record.path_id,
                                                  a.record.crossing_index))
        beam_of: dict[tuple, str] = {}
        for oid, k, members in beams:
            for a in members:
                beam_of[(a.line.src.source_id, *a.key)] = f"out:{oid}:{k}"
        reference: dict[str, float] = {}
        for a in ordered:
            reference.setdefault(beam_of[(a.line.src.source_id, *a.key)], _arrival_opl(a))
        output_modes: list[OutputModeV1] = []
        beam_basis: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        absorbing = {o.observation_id: o.absorbing for o in self.project.observations}
        for oid, k, members in beams:
            rec = members[0].record
            beam_id = f"out:{oid}:{k}"
            basis = lab_basis(rec.direction)
            beam_basis[beam_id] = basis
            self.basis_notes(rec.direction, f"observation:{oid}")
            g = rec.gstate
            gaussian = GaussianModeV1(
                beam_radius_x_mm=_q(g.radius("x"), LENGTH_DIGITS), beam_radius_y_mm=_q(g.radius("y"), LENGTH_DIGITS),
                waist_radius_x_mm=_q(g.waist_radius("x"), LENGTH_DIGITS), waist_radius_y_mm=_q(g.waist_radius("y"), LENGTH_DIGITS),
                waist_distance_x_mm=_q(g.waist_distance("x"), COARSE_DIGITS), waist_distance_y_mm=_q(g.waist_distance("y"), COARSE_DIGITS),
                axis_x=_qv(g.bx), axis_y=_qv(g.by), m2_x=_q(g.m2x, RATIO_DIGITS), m2_y=_q(g.m2y, RATIO_DIGITS))
            if g.m2x > 1.0 or g.m2y > 1.0:
                self.note("MULTIMODE_BEAM", "the beam's M^2 exceeds 1: the network acts on its embedded Gaussian mode and, in paraxial "
                          "optics without hard apertures, on every transverse mode of the beam alike", f"observation:{oid}")
            for pol, vector in zip(("H", "V"), basis):
                output_modes.append(OutputModeV1(
                    mode_id=f"{beam_id}:{pol}", observation_id=oid, beam=k, polarization=pol, position_mm=_qv(rec.position, LENGTH_DIGITS),
                    direction=_qv(rec.direction), basis_h=_qv(basis[0]), basis_v=_qv(basis[1]), absorbing=absorbing.get(oid, False),
                    medium_index=_q(rec.n, RATIO_DIGITS), gaussian=gaussian, reference_opl_mm=_q(reference[beam_id], OPL_DIGITS)))

        # -- paths and T
        rows = [m.mode_id for m in output_modes]
        row_observation = [m.observation_id for m in output_modes]
        cols = [m.mode_id for m in input_modes]
        row_index = {r: i for i, r in enumerate(rows)}
        col_index = {c: i for i, c in enumerate(cols)}
        t = np.zeros((len(rows), len(cols)), dtype=complex)
        # per entry, the error bound of the stated quantization: amplitude and offset rounding of each path (the reference phase is
        # common to the row and cancels in T^H T, the singular values, |T_ij|, permanents' moduli and every closed-loop phase)
        k0 = 2.0 * math.pi / (wavelength * NM_TO_MM)
        error = np.zeros((len(rows), len(cols)))
        reference_phase = {beam_id: _phase_factor(_q(opl, OPL_DIGITS), wavelength) for beam_id, opl in reference.items()}
        paths: list[NetworkPathV1] = []
        path_offset: dict[tuple[str, str], tuple[float, str, float | None]] = {}
        path_group: dict[tuple[str, str], float | None] = {}
        for a in ordered:
            src_id = a.line.src.source_id
            beam_id = beam_of[(src_id, *a.key)]
            basis = beam_basis[beam_id]
            opl = _arrival_opl(a)
            offset_q = _q(opl - reference[beam_id], OPL_OFFSET_DIGITS)
            phase = reference_phase[beam_id] * _phase_factor(offset_q, wavelength)
            amps = [[0j, 0j], [0j, 0j]]
            for c, pin in enumerate(("H", "V")):
                field3 = a.amplitudes.get(pin)
                if field3 is None:
                    continue
                for r, vector in enumerate(basis):
                    z = complex(vector @ field3)
                    amps[r][c] = complex(_q(z.real, AMPLITUDE_DIGITS), _q(z.imag, AMPLITUDE_DIGITS))
            path_id = _real_path_id(a.record.path_id, src_id, a.line.index)
            for r, pout in enumerate(("H", "V")):
                for c, pin in enumerate(("H", "V")):
                    i, j = row_index[f"{beam_id}:{pout}"], col_index[f"in:{src_id}#{a.line.index}:{pin}"]
                    t[i, j] += amps[r][c] * phase
                    error[i, j] += AMPLITUDE_QUANTUM / math.sqrt(2.0) + abs(amps[r][c]) * k0 * OPL_OFFSET_QUANTUM / 2.0
            rec = a.record
            group = a.group_opl.get("H", a.group_opl.get("V"))
            path_group[(beam_id, path_id)] = group
            paths.append(NetworkPathV1(
                path_id=path_id, source_id=src_id, output_beam=beam_id, crossing_index=rec.crossing_index, opl_mm=_q(opl, OPL_DIGITS),
                opl_offset_mm=offset_q, length_mm=_q(rec.length_mm, OPL_DIGITS),
                group_opl_mm=None if group is None else _q(group, OPL_DIGITS),
                group_delay_ps=None if group is None else _q(group / C_MM_PER_PS, OPL_DIGITS), lineage=tuple(rec.lineage),
                amplitudes=tuple(tuple((z.real + 0.0, z.imag + 0.0) for z in row) for row in amps),  # type: ignore[arg-type]
                ideal_lens=not rec.ray_opl_defined))
            path_offset[(beam_id, path_id)] = (opl - reference[beam_id], src_id, a.line.coherence_length_mm)
        t = np.vectorize(lambda z: complex(_q(z.real, ENTRY_DIGITS), _q(z.imag, ENTRY_DIGITS)))(t) if t.size else t
        error = error + ENTRY_QUANTUM / math.sqrt(2.0)
        tolerance = max(LOSSLESS_TOLERANCE, _gram_bound(t, error))
        gram = t.conj().T @ t if t.size else np.zeros((len(cols), len(cols)))
        lossless = bool(t.size) and bool(np.max(np.sqrt(abs2(gram - np.eye(len(cols))))) <= tolerance)
        singular = singular_values(t)  # one-sided Jacobi, not LAPACK: the reported values have the same bits in every runtime
        gain_allowance = max(LOSSLESS_TOLERANCE, float(np.sqrt(np.sum(error ** 2))) if t.size else 0.0)
        if singular.size and float(singular[0]) > 1.0 + gain_allowance:
            self.refuse(NetworkErrorCode.NETWORK_NOT_PASSIVE,
                        f"the largest singular value of T is {float(singular[0]):.9g}: the network would put out more light than it "
                        f"receives (beyond the {gain_allowance:.3g} the stated quantization allows); a declared phase (for example a "
                        "splitter's reflection_phase_rad other than +-pi/2 with R + T = 1, or a coating's stated reflection and "
                        "transmission phases whose difference is not +-pi/2 for a lossless splitter) breaks energy conservation")
            self.stop_if_refused()
        if not rows:
            self.note("NO_OUTPUT_REACHED", "no light of the exported sources reaches an output plane: T has no rows and every column "
                      "transmission is 0")
        transfer = TransferMatrixV1(
            rows=tuple(rows), columns=tuple(cols),
            entries=tuple(tuple((z.real + 0.0, z.imag + 0.0) for z in row) for row in t),
            column_transmission=tuple(_q(float(np.sum(abs2(t[:, j]))), FRACTION_DIGITS) for j in range(len(cols))),
            singular_values=tuple(_q(float(s), FRACTION_DIGITS) for s in singular), lossless_tolerance=_q(tolerance, FRACTION_DIGITS),
            lossless=lossless)

        # -- classical-limit check (declared, H and V input of every source)
        columns = []
        by_probe = {(p.line.src.source_id, p.polarization): p.outcome for p in probes}
        for line in lines:
            src_id = line.src.source_id
            traces = {"declared": checks[src_id], "H": by_probe[(src_id, "H")], "V": by_probe[(src_id, "V")]}
            columns.append(self.column_check(line, traces, arrivals, beams, t, row_observation, col_index, outputs))
            self.check_lossless_column(line, columns[-1], t, col_index, tolerance)
        self.stop_if_refused()
        classical = ClassicalCheckV1(tolerance=CHECK_TOLERANCE, columns=tuple(columns), status="PASS")

        # -- elements and hints
        elements = self.elements(probes, lines)
        hints = self.hints(path_offset, path_group)
        self.note("GOUY_PHASE_NOT_MODELLED", "the Gouy phase of the Gaussian beams is not part of the path phases")
        self.note("GROUP_DELAY_FIRST_ORDER", "paths carry their phase delay (opl_mm) and their group delay (group_opl_mm, group_delay_ps: "
                  "the group index of every medium at this wavelength); group-delay dispersion is not included. T is exact at this "
                  "wavelength; use group_delay_ps for the temporal overlap of wavepackets")
        if self.constant_index_media:
            media = ", ".join(sorted(self.constant_index_media))
            self.note("GROUP_INDEX_CONSTANT_MEDIA", f"media with a constant refractive index in their material model have no dispersion "
                      f"there: their group index is their index (n_g = n) in the group delays ({media})")
        stated = sorted(surf.surface_id for surf in self.scene.surfaces if surf.interaction.phases is not None)
        if stated:
            self.note("REFLECTION_PHASE_CONVENTION", f"coatings with stated phases use them (surfaces {', '.join(stated)}): r = sqrt(R) "
                      "e^{i phi_r}, t = sqrt(T) e^{i phi_t} in the (s, d_out x s) triad; every other coated or bare interface follows "
                      "the stated reflection models")
        else:
            self.note("REFLECTION_PHASE_CONVENTION", "coatings publish powers only: reflection and transmission phases follow the stated "
                      "reflection models, not measured coating phases")
        self.note("SOURCE_PORTS_ONLY", "only ports with a source are inputs: an unused input port (vacuum) has no column; place a source "
                  "on every input port that should get one")
        body = LinearOpticalNetworkV1.model_construct(
            schema_=NETWORK_SCHEMA, wavelength_nm=_sig(wavelength), provenance=provenance, conventions=conventions(),
            input_modes=tuple(input_modes), output_modes=tuple(output_modes), transfer_matrix=transfer, paths=tuple(paths),
            elements=elements, hints=NetworkHintsV1(path_pairs=hints, notes=tuple(sorted(self.notes.values(), key=lambda n: (n.code, n.location or "")))),
            classical_check=classical,
            validity=NetworkValidityV1(warnings=tuple(sorted(self.warnings.values(), key=lambda w: (w.code, w.location or "")))),
            contract_hash="")
        return finalize(body)

    def column_check(self, line: _Line, traces: dict[str, PhysicsOutcome], arrivals: list[_Arrival],
                     beams: list[tuple[str, int, list[_Arrival]]], t: np.ndarray, row_observation: list[str], col_index: dict[str, int],
                     outputs: list[str]) -> ColumnCheckV1:
        src_id = line.src.source_id
        where = f"source:{src_id}"
        col_h, col_v = col_index[f"in:{src_id}#{line.index}:H"], col_index[f"in:{src_id}#{line.index}:V"]
        inputs = {"declared": _source_lab_polarization(line.src, line.basis), "H": np.array([1.0 + 0j, 0j]), "V": np.array([0j, 1.0 + 0j])}
        k0 = 2.0 * math.pi / (line.wavelength_nm * NM_TO_MM)
        planes = []
        for name in CHECK_INPUTS:
            j_lab = inputs[name]
            outcome = traces[name]
            out_field = t[:, col_h] * j_lab[0] + t[:, col_v] * j_lab[1] if t.size else np.zeros(0)
            by_obs = {o.observation_id: o for o in outcome.observations}
            for oid in outputs:
                members = [i for i, row_oid in enumerate(row_observation) if row_oid == oid]
                predicted = float(sum(abs(out_field[i]) ** 2 for i in members))
                obs = by_obs.get(oid)
                if obs is None or not obs.arrivals:
                    engine = 0.0
                elif obs.superpositions:
                    sup = obs.superpositions[0]
                    if sup.total_power_W is None and obs.plane is not None:
                        # (the network describes one frequency: there the passes whose group delay the design does not state
                        # change no coherence, and their phases are the models' the network carries too -- the fields are
                        # summed as at one frequency)
                        single = tuple(a.model_copy(update={"group_delay_unknown_at": ()}) for a in obs.arrivals)
                        sup = superpositions(single, obs.plane, self.cfg)[0]
                    if sup.total_power_W is None:
                        self.refuse(NetworkErrorCode.CLASSICAL_LIMIT_MISMATCH,
                                    f"the Engine's coherent superposition on {oid} ({name} input) is not available "
                                    f"({', '.join(sup.issue_codes) or sup.status.value})", where)
                        continue
                    engine = float(sup.total_power_W)
                else:
                    if obs.total_power_W is None:
                        upper = obs.power_upper_bound_W
                        interval = (f"between 0 and {upper:.9g} W" if upper is not None else "not established")
                        self.refuse(NetworkErrorCode.OUT_OF_MODEL_SCOPE,
                                    f"the Engine's power on {oid} ({name} input) is {interval}; the numeric transfer matrix "
                                    "cannot be checked against an unknown coherent power", f"observation:{oid}")
                        continue
                    engine = float(obs.total_power_W)
                # T carries the declared quantization: each path's offset from its beam's reference optical path length is
                # rounded to q before the phases are formed, which shifts its phase by at most k0 q / 2, i.e. |sum T J|^2 by at most
                # A^2 k0 q (A: sum of the plane's path amplitudes); the reference phase is common to the row and cancels
                amplitude_sum = sum(_amplitude(a, j_lab) for a in arrivals if a.line is line and a.record.observation_id == oid)
                allowance = (CHECK_TOLERANCE * max(engine, CHECK_FLOOR) + amplitude_sum ** 2 * k0 * OPL_OFFSET_QUANTUM
                             + 4.0 * amplitude_sum * len(members) * ENTRY_QUANTUM
                             + _separate_mode_overlap(line, j_lab, [g for g in beams if g[0] == oid]))
                deviation = abs(predicted - engine)
                if deviation > allowance:
                    self.refuse(NetworkErrorCode.CLASSICAL_LIMIT_MISMATCH,
                                f"on {oid} the network predicts {predicted:.9g} of source {src_id}'s power ({name} input) but the Engine's "
                                f"coherent superposition of the same beams gives {engine:.9g} (deviation {deviation:.3g} > {allowance:.3g}): "
                                "the beams there are not the modes the network assumes (tilt, offset, size or curvature mismatch)",
                                f"observation:{oid}")
                planes.append(PlaneCheckV1(observation_id=oid, input=name, predicted_fraction=_q(predicted, FRACTION_DIGITS),
                                           engine_fraction=_q(engine, FRACTION_DIGITS), deviation=_q(deviation, FRACTION_DIGITS),
                                           allowance=_q(allowance, FRACTION_DIGITS)))
        # ledger of the declared-polarization trace: what reaches the outputs + the losses = 1
        check = traces["declared"]
        budget = check.power_budget
        sinks = dict(budget.lines[0].sinks) if budget is not None and budget.lines else {}
        output_set = set(outputs)
        paths = {p.path_id: p for p in check.trace.paths}
        output_power = 0.0
        for rec in check.trace.arrivals:
            if rec.observation_id in output_set and rec.counted:
                output_power += rec.power_W
                path = paths.get(rec.path_id)
                if path is not None:
                    kind = "DETECTED" if path.termination == "DETECTED" else "ESCAPED"
                    sinks[kind] = sinks.get(kind, 0.0) - path.final_power_W
        losses = {kind: power for kind, power in sinks.items() if abs(power) > LEDGER_FLOOR}
        loss = float(sum(losses.values()))
        residual = 1.0 - output_power - loss
        return ColumnCheckV1(source_id=src_id, planes=tuple(planes), output_path_power_fraction=_q(output_power, FRACTION_DIGITS),
                             lost_path_power_fraction=_q(loss, FRACTION_DIGITS),
                             lost_path_power_by_kind=tuple((kind, _q(power, FRACTION_DIGITS)) for kind, power in sorted(losses.items())),
                             ledger_residual=_q(residual, FRACTION_DIGITS), status="PASS")

    def check_polarizing_normal_incidence(self, probes: list[_Probe], wavelength_nm: float) -> None:
        """Refuse a coating whose model is polarization-dependent at normal incidence (a constant R_s != R_p, a PBS coating)
        met within POLARIZING_NORMAL_INCIDENCE_DEG of normal incidence: there the Engine's s axis is the plane of incidence of a
        misalignment or, at exactly normal incidence, its internal transverse axis (``optics/polarization/jones.s_axis``), so
        the result depends on a choice the layout does not make. A coating whose data have R_s = R_p at normal incidence
        (physical coatings: the s / p difference grows as sin^2 of the angle) is continuous there and is kept."""
        surfaces = {s.surface_id: s for s in self.scene.surfaces}
        seen: set[str] = set()
        for probe in probes:
            for event in probe.events:
                if event.event not in COATED_EVENTS or event.surface_id in seen or event.aoi_deg >= POLARIZING_NORMAL_INCIDENCE_DEG:
                    continue
                surf = surfaces.get(event.surface_id)
                coating = surf.interaction.coating if surf is not None else None
                if coating is None:
                    continue
                location = f"surface:{event.surface_id}"
                cc = coating.coefficients(wavelength_nm, np.array([0.0]), location=location)
                if cc.issue is not None:  # no data at normal incidence: judge it where the beam meets it
                    cc = coating.coefficients(wavelength_nm, np.array([event.aoi_deg]), location=location)
                    if cc.issue is not None:
                        continue
                if (abs(float(cc.rs[0]) - float(cc.rp[0])) > POLARIZING_COATING_TOLERANCE
                        or abs(float(cc.ts[0]) - float(cc.tp[0])) > POLARIZING_COATING_TOLERANCE):
                    seen.add(event.surface_id)
                    self.refuse(NetworkErrorCode.POLARIZING_NORMAL_INCIDENCE,
                                f"{event.surface_id} has a coating that is polarization-dependent at normal incidence (R_s {float(cc.rs[0]):.6g}, "
                                f"R_p {float(cc.rp[0]):.6g}, T_s {float(cc.ts[0]):.6g}, T_p {float(cc.tp[0]):.6g}) and the beam meets it at "
                                f"{event.aoi_deg:.3g} deg, within "
                                f"{POLARIZING_NORMAL_INCIDENCE_DEG:g} deg of normal incidence: which field component it treats as s is set "
                                "there by a misalignment or the Engine's internal axis, not by the layout; tilt it to its design angle of "
                                "incidence or use a non-polarizing element", f"surface:{event.surface_id}")

    def check_element_passivity(self, probes: list[_Probe], wavelength_nm: float) -> None:
        """Refuse an element whose amplitudes would create light: a beam splitter, or a refracting interface whose coating
        states phases and whose reflection is traced, met by the probes. The Engine applies such an element's stated phases (and a splitter's own reflection
        and transmission phases) to light from either side, so per polarization its two-port scattering matrix is
        ``[[r, t], [t, r]]``, with largest singular value squared ``R + T + 2 |Re(r t*)|``; an interface whose reflection keeps
        the bare Fresnel phase (its sign flips with the side) has ``[[r, t], [t, -r]]`` and ``R + T + 2 |Im(r t*)|``. Above
        ``1 + PASSIVITY_TOLERANCE`` the element is not passive whatever the arm lengths: a lossless splitter (R + T = 1)
        needs ``phi_r - phi_t = +-pi/2``. R, T at each probe's angle of incidence and the wavelength, s and p."""
        surfaces = {s.surface_id: s for s in self.scene.surfaces}
        seen: set[str] = set()
        for probe in probes:
            for event in probe.events:
                surf = surfaces.get(event.surface_id)
                if surf is None or event.surface_id in seen:
                    continue
                inter = surf.interaction
                if inter.coating is None or not (inter.kind == "beam_splitter"
                                                 or (inter.kind == "refract" and inter.phases is not None and inter.trace_reflection)):
                    continue
                location = f"surface:{event.surface_id}"
                cc = inter.coating.coefficients(wavelength_nm, np.array([event.aoi_deg]), location=location)
                if cc.issue is not None:
                    continue
                phases = None
                if inter.phases is not None:
                    phases, issue = inter.phases.at(wavelength_nm, location=location)
                    if issue is not None:
                        continue
                if inter.kind == "beam_splitter":
                    t_s, t_p, r_s, r_p = splitter_amplitudes(cc.rs[:1], cc.rp[:1], cc.ts[:1], cc.tp[:1], inter.reflection_phase,
                                                             inter.transmission_phase, phases)
                    same_reflection = True
                else:
                    bare = np.ones(1, dtype=np.complex128)  # bare Fresnel t is real and positive; a bare r's sign does not matter here
                    t_s, t_p, r_s, r_p = coated_interface(cc.rs[:1], cc.rp[:1], cc.ts[:1], cc.tp[:1], bare, bare, bare, bare, phases)
                    same_reflection = phases is not None and phases.r_s is not None
                for name, r, t in (("s", complex(r_s[0]), complex(t_s[0])), ("p", complex(r_p[0]), complex(t_p[0]))):
                    cross = r * t.conjugate()
                    largest = abs(r) ** 2 + abs(t) ** 2 + 2.0 * abs(cross.real if same_reflection else cross.imag)
                    if largest > 1.0 + PASSIVITY_TOLERANCE:
                        seen.add(event.surface_id)
                        self.refuse(NetworkErrorCode.NETWORK_NOT_PASSIVE,
                                    f"{event.surface_id} ({surf.owner_id}) at {event.aoi_deg:.6g} deg, {name}-polarized: R {abs(r) ** 2:.6g}, "
                                    f"T {abs(t) ** 2:.6g} with reflection phase {math.atan2(r.imag, r.real):.6g} rad and transmission phase "
                                    f"{math.atan2(t.imag, t.real):.6g} rad would put out up to {largest:.9g} of the light it receives from both "
                                    "sides: these phases do not belong to a passive element (a lossless splitter needs reflection minus "
                                    "transmission phase = +-pi/2); state phases that satisfy r t* + t r* = 0, or R + T + "
                                    "2 sqrt(R T) |cos(phi_r - phi_t)| <= 1 for a lossy coating", location)
                        break

    def check_lossless_column(self, line: _Line, column: ColumnCheckV1, t: np.ndarray, col_index: dict[str, int], tolerance: float) -> None:
        """A source whose power ledger loses nothing (every path reaches an output) must keep all its power in T: its declared
        input's column ``|T J|^2`` summed over the outputs equals 1 within the lossless tolerance, else the amplitudes do not
        conserve energy (NETWORK_NOT_PASSIVE), whatever the arm lengths."""
        if column.lost_path_power_fraction > LOSSLESS_TOLERANCE or not t.size:
            return
        j_lab = _source_lab_polarization(line.src, line.basis)
        col_h, col_v = col_index[f"in:{line.src.source_id}#{line.index}:H"], col_index[f"in:{line.src.source_id}#{line.index}:V"]
        out_field = t[:, col_h] * j_lab[0] + t[:, col_v] * j_lab[1]
        total = float(np.sum(np.real(out_field * np.conj(out_field))))
        if abs(total - column.output_path_power_fraction) > tolerance:
            self.refuse(NetworkErrorCode.NETWORK_NOT_PASSIVE,
                        f"source {line.src.source_id}'s paths lose no light ({column.output_path_power_fraction:.9g} of it reaches the "
                        f"outputs) but the network's amplitudes deliver {total:.9g}: the elements' phases do not conserve energy",
                        f"source:{line.src.source_id}")

    def basis_notes(self, direction, location: str) -> None:
        """Notes / warnings about the lab basis of a mode (``basis.lab_basis``)."""
        if near_switch(direction):
            self.warnings.setdefault(("POLARIZATION_BASIS_SWITCH", location), ValidityIssueV1(
                code="POLARIZATION_BASIS_SWITCH", location=location,
                message="the beam is within 1e-9 of the 45 degree switch of the polarization basis reference (design +Z to +Y): a "
                        "last-bit difference of another runtime could pick the other reference, turning this mode's H / V about the "
                        "beam (and changing the contract hash); tilt the beam away from 45 degrees to vertical"))
        elif uses_fallback(direction):
            self.note("VERTICAL_BEAM_BASIS", "a beam within 45 degrees of vertical: its V is design +Y projected perpendicular to "
                      "the beam (not +Z)", location)

    def elements(self, probes: list[_Probe], lines: list[_Line]) -> tuple[ElementTransferV1, ...]:
        owners = {s.surface_id: s.owner_id for s in self.scene.surfaces}
        by_source: dict[str, dict[str, list[ElementEvent]]] = {}
        for probe in probes:
            by_source.setdefault(probe.line.src.source_id, {})[probe.polarization] = probe.events
        seen: set[tuple] = set()
        out: list[ElementTransferV1] = []
        for line in lines:
            events = by_source.get(line.src.source_id, {})
            h_events = {e.key: e for e in events.get("H", [])}
            v_events = {e.key: e for e in events.get("V", [])}
            for key in sorted(set(h_events) | set(v_events)):
                eh, ev = h_events.get(key), v_events.get(key)
                ref = eh or ev
                identity = (ref.surface_id, ref.event, _qv(ref.d_in, 6), _qv(ref.d_out, 6))
                if identity in seen:
                    continue
                seen.add(identity)
                b_in, b_out = lab_basis(ref.d_in), lab_basis(ref.d_out)
                matrix = None
                if eh is not None and ev is not None:
                    inp = np.array([[complex(b @ e.e_in) for e in (eh, ev)] for b in b_in])
                    outp = np.array([[complex(b @ e.e_out) for e in (eh, ev)] for b in b_out])
                    sv = singular_values(inp)
                    inverse = inverse_2x2(inp)
                    if sv[0] > 0.0 and sv[-1] > DEPENDENT_INPUT_RATIO * sv[0] and inverse is not None:
                        m = outp @ inverse
                        matrix = tuple(tuple(_qc(complex(m[r, c]), ELEMENT_DIGITS) for c in range(2)) for r in range(2))
                out.append(ElementTransferV1(
                    surface_id=ref.surface_id, owner_id=owners.get(ref.surface_id), event=ref.event,
                    aoi_deg=_q(ref.aoi_deg, COARSE_DIGITS),
                    in_direction=_qv(ref.d_in), out_direction=_qv(ref.d_out), matrix=matrix,  # type: ignore[arg-type]
                    observed_on_path=_real_path_id(ref.path_id, line.src.source_id, line.index)))
        return tuple(out)

    def hints(self, path_offset: dict[tuple[str, str], tuple[float, str, float | None]],
              path_group: dict[tuple[str, str], float | None]) -> tuple[PathPairHintV1, ...]:
        """Optical path differences of the paths reaching one output beam (from their unrounded offsets to the beam's reference),
        with their group optical path and group delay differences."""
        by_beam: dict[str, list[tuple[str, float, str, float | None]]] = {}
        for (beam_id, path_id), (opl, src_id, lc) in path_offset.items():
            by_beam.setdefault(beam_id, []).append((path_id, opl, src_id, lc))
        pairs: list[PathPairHintV1] = []
        for beam_id in by_beam:
            members = by_beam[beam_id]
            for i, (pa, oa, sa, la) in enumerate(members):
                for pb, ob, sb, lb in members[i + 1:]:
                    opd = _q(oa - ob, OPL_DIGITS)
                    lengths = [x for x in (la, lb) if x is not None]
                    lc = min(lengths) if len(lengths) == 2 else None
                    ga, gb = path_group.get((beam_id, pa)), path_group.get((beam_id, pb))
                    group_opd = None if ga is None or gb is None else _q(ga - gb, OPL_DIGITS)
                    pairs.append(PathPairHintV1(
                        output_beam=beam_id, path_a=pa, path_b=pb, same_source=sa == sb, opd_mm=opd, delay_ps=_q(opd / C_MM_PER_PS, OPL_DIGITS),
                        group_opd_mm=group_opd, group_delay_ps=None if group_opd is None else _q(group_opd / C_MM_PER_PS, OPL_DIGITS),
                        coherence_length_mm=None if lc is None else _sig(lc),
                        opd_over_coherence_length=None if lc is None else _q(abs(opd) / lc, RATIO_DIGITS)))
        return tuple(pairs)


_WARNING_TEXT = {
    "RETARDER_OBLIQUE_INCIDENCE": "a waveplate or polarizer is crossed at an oblique angle: its thin-element Jones model loses accuracy",
    "GAUSSIAN_TRUNCATION": "an aperture truncates a beam slightly: the mode is no longer exactly Gaussian",
    "GRAZING_INCIDENCE": "a beam meets a surface near grazing incidence",
    "TIR_BOUNDARY": "a beam meets a surface close to the critical angle",
    "MAX_PATHS": "the trace budget was reached",
}


def _amplitude(arrival: _Arrival, j_lab: np.ndarray) -> float:
    """|field| of one output crossing for the lab input polarization ``j_lab``."""
    return float(np.linalg.norm(sum(field3 * j_lab[c] for c, field3 in enumerate(
        (arrival.amplitudes.get("H", np.zeros(3)), arrival.amplitudes.get("V", np.zeros(3)))))))


def _separate_mode_overlap(line: _Line, j_lab: np.ndarray, groups: list[tuple[str, int, list[_Arrival]]]) -> float:
    """Bound of the cross terms the network leaves out on one plane: beams it calls separate modes still overlap by
    eta <= mode_overlap_tolerance, which adds up to 2 |A_g| |A_h| sqrt(eta_gh) to the Engine's whole-plane power (A: the
    amplitude sum of a mode's paths from this source). Beams crossing the plane in opposite directions do not interfere there."""
    heads, sums = [], []
    for _oid, _k, members in groups:
        mine = [a for a in members if a.line is line]
        if mine:
            heads.append(members[0].record)
            sums.append(sum(_amplitude(a, j_lab) for a in mine))
    total = 0.0
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            a, b = heads[i], heads[j]
            if float(a.direction @ a.plane_normal) * float(b.direction @ b.plane_normal) > 0.0:
                total += 2.0 * sums[i] * sums[j] * math.sqrt(mode_overlap(a, b))
    return total


def _segments_to(paths: dict, path_id: str, end_surface: str, opl_mm: float) -> list | None:
    """The chief-ray segments from the emission point to the crossing of ``end_surface`` whose summed optical path length is
    ``opl_mm`` (within ``GROUP_PATH_TOLERANCE_MM``): the segments of the paths it branched from, each up to its branch point,
    then its own up to that crossing. ``None`` when there is no such crossing."""

    def gather(pid: str, stop) -> list | None:
        path = paths.get(pid)
        if path is None:
            return None
        gathered: list = []
        if path.parent_path_id is not None:
            if not path.segments:
                return None
            start, branch = np.array(path.segments[0].start_mm), path.branch_surface_id

            def at_branch(segment, _gathered) -> bool:
                return (segment.end_surface_id == branch and segment.end_mm is not None
                        and float(np.linalg.norm(np.array(segment.end_mm) - start)) <= BRANCH_POINT_TOLERANCE_MM)

            parent = gather(path.parent_path_id, at_branch)
            if parent is None:
                return None
            gathered = parent
        for segment in path.segments:
            gathered = [*gathered, segment]
            if stop(segment, gathered):
                return gathered
        return None

    def at_output(segment, gathered) -> bool:
        if segment.end_surface_id != end_surface:
            return False
        total = sum(s.optical_path_length_mm for s in gathered if s.optical_path_length_mm is not None)
        return abs(total - opl_mm) <= GROUP_PATH_TOLERANCE_MM

    return gather(path_id, at_output)


def _one(planes: set[str]) -> str | None:
    """The plane of a refusal that concerns exactly one output plane, else ``None``."""
    return next(iter(planes)) if len(planes) == 1 else None


def _truncating_surfaces(outcome: PhysicsOutcome) -> dict[str, set[str]]:
    """Per path id: the surfaces where it lost power at a clear aperture and where the trace reports a truncation above the
    warning fraction (``GAUSSIAN_TRUNCATION``)."""
    prefix = "surface:"
    warned = {issue.location[len(prefix):] for issue in outcome.issues
              if issue.code is ErrorCode.GAUSSIAN_TRUNCATION and issue.location is not None and issue.location.startswith(prefix)}
    out: dict[str, set[str]] = {}
    for flow in outcome.trace.power_flows:
        if flow.kind == APERTURE_TRUNCATION_SINK and flow.path_id is not None and flow.surface_id in warned and flow.power_W > 0.0:
            out.setdefault(flow.path_id, set()).add(flow.surface_id)
    return out


#: Lineage events of a neutral-density attenuator (``optics/rays/tracer.py``): its transmission and its traced reflection.
ATTENUATOR_EVENTS = {"neutral_density": "crosses the attenuator", "attenuator_reflect": "is reflected by the attenuator"}


def _attenuator_crossings(lineage) -> list[str]:
    """What a path (its lineage includes its ancestors') does at neutral-density attenuators, e.g. ``["crosses the attenuator
    at ND/attenuator"]``; empty when it meets none."""

    found: list[str] = []
    for entry in lineage:
        surface, _sep, event = entry.rpartition(":")
        if event in ATTENUATOR_EVENTS:
            text = f"{ATTENUATOR_EVENTS[event]} at {surface}"
            if text not in found:
                found.append(text)
    return found


#: Lineage event of a Faraday rotator (``optics/rays/tracer.py``).
FARADAY_EVENT = "faraday_rotator"


def _faraday_crossings(lineage) -> list[str]:
    """The Faraday rotators a path (its lineage includes its ancestors') crosses; empty when it meets none."""

    surfaces = [entry.rpartition(":")[0] for entry in lineage if entry.rpartition(":")[2] == FARADAY_EVENT]
    return list(dict.fromkeys(surfaces))


#: Lineage events of a diffraction grating's orders (``optics/rays/tracer.py``).
GRATING_EVENTS = frozenset({"grating_reflect", "grating_transmit"})


def _grating_crossings(lineage) -> list[str]:
    """The diffraction gratings a path (its lineage includes its ancestors') crosses; empty when it meets none."""

    surfaces = [entry.rpartition(":")[0] for entry in lineage if entry.rpartition(":")[2] in GRATING_EVENTS]
    return list(dict.fromkeys(surfaces))


def _ancestry_surfaces(path_id: str, paths: dict, truncated: dict[str, set[str]]) -> set[str]:
    """``truncated`` surfaces of the path and of the paths it branched from."""
    found: set[str] = set()
    seen: set[str] = set()
    current: str | None = path_id
    while current is not None and current not in seen:
        seen.add(current)
        found |= truncated.get(current, set())
        path = paths.get(current)
        current = path.parent_path_id if path is not None else None
    return found


def _probe_index(probes: list[_Probe], line: _Line) -> float:
    for probe in probes:
        if probe.line is line:
            for path in probe.outcome.trace.paths:
                if path.segments:
                    return float(path.segments[0].refractive_index)
    return 1.0


def _real_path_id(probe_path_id: str, source_id: str, index: int) -> str:
    """The probe launches the selected line as line 0: report the Project's own path id (``<source>#<line index>...``)."""
    prefix = f"{source_id}#0"
    return f"{source_id}#{index}{probe_path_id[len(prefix):]}" if probe_path_id.startswith(prefix) else probe_path_id


def finalize(body: LinearOpticalNetworkV1) -> LinearOpticalNetworkV1:
    """Validate the document and set its ``contract_hash`` (canonical JSON of the body without the hash)."""
    data = body.model_dump(mode="json", by_alias=True)
    data.pop("contract_hash", None)
    digest = hash_exact(data)
    data["contract_hash"] = digest
    return LinearOpticalNetworkV1.model_validate(data)


def contract_hash_of(document: dict) -> str:
    """Recompute the ``contract_hash`` of a network document (JSON form)."""
    data = dict(document)
    data.pop("contract_hash", None)
    return hash_exact(data)


def build_linear_optical_network(project: ProjectSnapshot, cfg: NumericalConfig | None = None, options: NetworkExportOptionsV1 | dict | None = None,
                                 *, engine_input_hash: str, project_hash: str | None = None, catalog_version: str | None = None,
                                 catalog_sha256: str | None = None, warnings: tuple[ValidityIssueV1, ...] = ()) -> LinearOpticalNetworkV1:
    """The linear optical network of ``project`` (module docstring). Raises ``NetworkExportError`` when it is not exportable.
    ``warnings``: findings of the caller's own checks on light no exported output receives (``validity.warnings``)."""
    cfg = cfg if cfg is not None else NumericalConfig()
    if options is None:
        options = NetworkExportOptionsV1()
    elif not isinstance(options, NetworkExportOptionsV1):
        options = NetworkExportOptionsV1.model_validate(options)
    provenance = NetworkProvenanceV1(
        engine_version=ENGINE_VERSION, engine_build_id=engine_build_id(), numerical_config_version=NUMERICAL_CONFIG_VERSION,
        catalog_version=catalog_version, catalog_sha256=catalog_sha256, project_hash=project_hash, engine_input_hash=engine_input_hash,
        options=options)
    exporter = _Exporter(project, cfg, options)
    for warning in warnings:
        exporter.warnings.setdefault((warning.code, warning.location), warning)
    try:
        return exporter.build(provenance)
    except NetworkExportError:
        raise
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        raise NetworkExportError([NetworkProblem(NetworkErrorCode.NUMERICAL_FAILURE,
                                                 f"the network could not be computed ({type(exc).__name__}: {str(exc)[:EXCEPTION_TEXT_CHARS]}); "
                                                 "check the layout's numbers for non-finite or extreme magnitudes")]) from None

