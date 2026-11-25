"""``opensci-linear-optical-network/v1``: a Project's optics as a linear optical network (modes, transfer matrix, paths).

The export (``os_optics.network``) describes, for ONE vacuum wavelength, how the field amplitudes of the input modes
(each source, in two lab polarizations H / V) map onto the output modes (each distinct beam crossing an output observation
plane, in its own lab H / V): ``b_out = T a_in``. Every number comes from the Engine's own trace (optical path lengths,
Jones matrices of the interactions, coating powers, Fresnel amplitudes, losses). The document states its conventions
(``conventions``) and its provenance, and carries a ``contract_hash`` over its canonical, quantized body, so the same Project
exported by the Python Engine and by the browser runtime gives the same hash.

Numbers are plain JSON: lengths in mm, wavelengths in nm, angles in rad unless named ``_deg``, powers as fractions of the
source line's power. A complex number is ``[re, im]``; a 2 x 2 complex matrix is ``[[m00, m01], [m10, m11]]`` (rows: output
H / V, columns: input H / V).
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from .base import Complex, StrictModel, Vector3

NETWORK_SCHEMA = "opensci-linear-optical-network/v1"

ComplexMatrix2 = tuple[tuple[Complex, Complex], tuple[Complex, Complex]]


class NetworkExportOptionsV1(StrictModel):
    """What to export. Every field is optional; ``None`` means the default stated per field."""

    wavelength_nm: float | None = Field(None, gt=0, description="The spectral line to export (vacuum wavelength). Required when the "
                                        "sources emit more than one wavelength; a source without a line at it has no input mode.")
    source_ids: tuple[str, ...] | None = Field(None, min_length=1, description="The sources that get input modes (default: every source).")
    observation_ids: tuple[str, ...] | None = Field(None, min_length=1, description="The observation planes that are output ports "
                                                    "(default: every observation plane).")
    mode_overlap_tolerance: float = Field(1e-6, gt=0, le=0.1, description="Two beams on one output plane are one mode when their "
                                          "normalized spatial mode overlap |<a|b>|^2 is at least 1 - tolerance, and two separate "
                                          "modes when it is at most the tolerance; anything between is refused (PARTIAL_MODE_OVERLAP).")


class NetworkProvenanceV1(StrictModel):
    engine_version: str
    engine_build_id: str
    numerical_config_version: str
    catalog_version: str | None = Field(None, description="The public optical catalog release the Project was lowered against.")
    catalog_sha256: str | None = Field(None, description="SHA-256 of that release's bytes, when this runtime holds them.")
    project_hash: str | None = Field(None, description="osc1 content hash of the Project document (None for an Engine request).")
    engine_input_hash: str = Field(description="Hash of the Engine request the export traced (the same as a run's input hash).")
    options: NetworkExportOptionsV1


class ReflectionModelV1(StrictModel):
    interaction: str
    model: str


class QuantizationV1(StrictModel):
    quantity: str
    resolution: float
    unit: str


class NetworkConventionsV1(StrictModel):
    time_dependence: str
    propagation_phase: str
    amplitude_normalization: str
    transfer_definition: str
    polarization_basis: str
    vertical_beam_fallback: str
    input_reference_plane: str
    output_reference_plane: str
    gauge: str
    group_delay: str | None = Field(None, description="How group_opl_mm / group_delay_ps of the paths and path pairs are computed.")
    output_modes: str
    input_modes: str
    reflection_models: tuple[ReflectionModelV1, ...]
    not_modelled: tuple[str, ...]
    quantization: tuple[QuantizationV1, ...]


class InputModeV1(StrictModel):
    mode_id: str = Field(description="'in:<source_id>#<spectral_index>:<H|V>'.")
    source_id: str
    spectral_index: int = Field(ge=0)
    polarization: Literal["H", "V"]
    position_mm: Vector3 = Field(description="The source's emission point (design frame): the input reference plane.")
    direction: Vector3
    basis_h: Vector3
    basis_v: Vector3
    line_power_W: float = Field(description="Power of this spectral line in the Project (the network itself is power-independent).")
    source_polarization: tuple[Complex, Complex] = Field(description="The source's declared polarization in this lab (H, V) basis.")
    coherence_group: str
    coherence_length_mm: float | None = Field(description="The source's coherence length: the declared one of the line, or for a "
                                              "continuous spectrum the spectral density's (|gamma| = 1/e). None: not stated.")
    medium_index: float


class GaussianModeV1(StrictModel):
    """The beam's Gaussian mode at the plane (embedded Gaussian of an M^2 > 1 beam)."""

    beam_radius_x_mm: float
    beam_radius_y_mm: float
    waist_radius_x_mm: float
    waist_radius_y_mm: float
    waist_distance_x_mm: float = Field(description="Plane position minus waist position along the beam (positive: waist upstream).")
    waist_distance_y_mm: float
    axis_x: Vector3
    axis_y: Vector3
    m2_x: float
    m2_y: float


class OutputModeV1(StrictModel):
    mode_id: str = Field(description="'out:<observation_id>:<beam>:<H|V>'.")
    observation_id: str
    beam: int = Field(ge=0, description="Index of the distinct beam (spatial mode) on this plane, in the plane's beam order.")
    polarization: Literal["H", "V"]
    position_mm: Vector3 = Field(description="The beam's chief-ray crossing of the plane: the output reference point.")
    direction: Vector3
    basis_h: Vector3
    basis_v: Vector3
    absorbing: bool
    medium_index: float
    gaussian: GaussianModeV1
    reference_opl_mm: float = Field(description="Optical path length of the beam's reference path (its first path in paths order), "
                                    "rounded to 1e-9 mm: exp(i k0 reference_opl_mm) is common to the beam's two rows (a choice of "
                                    "output reference plane); the paths' opl_offset_mm carry their phases relative to it.")


class TransferMatrixV1(StrictModel):
    rows: tuple[str, ...] = Field(description="Output mode ids, in output_modes order.")
    columns: tuple[str, ...] = Field(description="Input mode ids, in input_modes order.")
    entries: tuple[tuple[Complex, ...], ...] = Field(description="T[row][column] as [re, im].")
    column_transmission: tuple[float, ...] = Field(description="sum_i |T_ij|^2 per column: the fraction of an input mode's photons "
                                                   "that reach the output modes (1 - loss).")
    singular_values: tuple[float, ...]
    lossless_tolerance: float = Field(description="max(1e-6, the bound the stated quantization puts on the entries of T^H T). A "
                                      "singular value above 1 + lossless_tolerance (gain) is refused (NETWORK_NOT_PASSIVE).")
    lossless: bool = Field(description="|T^H T - I| <= lossless_tolerance entry by entry (no loss channel between the inputs and the "
                           "outputs).")


class NetworkPathV1(StrictModel):
    """One beam path from a source to an output mode: ``T`` is the sum over paths of
    ``amplitudes * exp(i k0 reference_opl_mm) * exp(i k0 opl_offset_mm)`` (``reference_opl_mm`` of the path's output beam)."""

    path_id: str
    source_id: str
    output_beam: str = Field(description="'out:<observation_id>:<beam>' (the output mode without the polarization).")
    crossing_index: int
    opl_mm: float = Field(description="Optical path length from the source's emission point to the output plane, rounded to "
                          "1e-9 mm (the phase in T uses reference_opl_mm + opl_offset_mm).")
    opl_offset_mm: float = Field(description="opl_mm minus the output beam's reference_opl_mm, from the unrounded lengths, rounded to "
                                 "1e-12 mm: paths reaching one output beam keep their relative phase to k0 x 5e-13 mm.")
    length_mm: float
    group_opl_mm: float | None = Field(None, description="Group optical path length sum n_g x geometric length from the source's emission "
                                       "point to the output plane, n_g = n - lambda dn/dlambda of each medium at this wavelength from the "
                                       "Engine's material dispersion, rounded to 1e-9 mm (None when a medium has no dispersion data here).")
    group_delay_ps: float | None = Field(None, description="group_opl_mm / c: the arrival time of a wavepacket's envelope along this path, "
                                         "rounded to 1e-9 ps.")
    lineage: tuple[str, ...]
    amplitudes: ComplexMatrix2 = Field(description="Flux-normalized Jones matrix of the path without the propagation phase: rows output "
                                       "H / V, columns input H / V.")
    ideal_lens: bool = Field(False, description="The path crosses an ideal or tabulated thin lens (its optical path length there is not modelled).")


class ElementTransferV1(StrictModel):
    """The Jones matrix of one interaction as seen in the lab bases of its input and output beam directions."""

    surface_id: str
    owner_id: str | None
    event: str
    aoi_deg: float
    in_direction: Vector3
    out_direction: Vector3
    matrix: ComplexMatrix2 | None = Field(description="Rows output H / V, columns input H / V; None when the two probe polarizations "
                                          "reach the element in the same state (an upstream polarizer), so one column is unobserved.")
    observed_on_path: str


class PathPairHintV1(StrictModel):
    output_beam: str
    path_a: str
    path_b: str
    same_source: bool
    opd_mm: float = Field(description="opl(a) - opl(b).")
    delay_ps: float = Field(description="opd_mm / c: the phase delay (relative phase of the carrier).")
    group_opd_mm: float | None = Field(None, description="group_opl_mm(a) - group_opl_mm(b).")
    group_delay_ps: float | None = Field(None, description="group_opd_mm / c: the relative delay of the two paths' wavepacket envelopes (the "
                                         "delay to use for the temporal overlap of photons or pulses).")
    coherence_length_mm: float | None = Field(description="The smaller coherence length of the two sources (InputModeV1."
                                              "coherence_length_mm).")
    opd_over_coherence_length: float | None


class HintNoteV1(StrictModel):
    code: str
    message: str
    location: str | None = None


class NetworkHintsV1(StrictModel):
    path_pairs: tuple[PathPairHintV1, ...]
    notes: tuple[HintNoteV1, ...]


class PlaneCheckV1(StrictModel):
    observation_id: str
    input: Literal["declared", "H", "V"] = Field(description="The input polarization J: the source's declared polarization, or the "
                                                 "lab H / V probe (each input column of T is checked).")
    predicted_fraction: float = Field(description="sum over the plane's output modes of |T J|^2.")
    engine_fraction: float = Field(description="The Engine's coherent superposition of the beams on the plane (1 W trace of this source "
                                   "with polarization J, unlimited coherence length).")
    deviation: float = Field(description="|predicted - engine|.")
    allowance: float = Field(description="tolerance x max(engine, 1e-9) + the bound the declared quantization puts on |T J|^2 "
                             "(A^2 k0 q_offset + 4 A n q_entry; A: sum of the plane's path amplitudes, n: its output modes, q_offset "
                             "1e-12 mm).")


class ColumnCheckV1(StrictModel):
    """One source's check, and the Engine's power ledger of its trace with the declared polarization. The ledger counts each
    beam path's own power (no interference between paths): ``output_path_power_fraction`` + ``lost_path_power_fraction`` +
    ``ledger_residual`` = 1. Interference redistributes power between paths that meet again (``transfer_matrix
    .column_transmission`` is the coherent fraction reaching the outputs)."""

    source_id: str
    planes: tuple[PlaneCheckV1, ...]
    output_path_power_fraction: float = Field(description="Sum of the powers of the paths that reach the output planes.")
    lost_path_power_fraction: float = Field(description="Sum of the powers of the paths that leave the network elsewhere, and of losses "
                                            "at interactions (lost_path_power_by_kind).")
    lost_path_power_by_kind: tuple[tuple[str, float], ...] = Field(description="Per power-ledger sink kind (ESCAPED: left the layout "
                                                                   "without crossing an output; UNTRACED_REFLECTION; ABSORBED; ...).")
    ledger_residual: float
    status: Literal["PASS"]


class ClassicalCheckV1(StrictModel):
    """What the check validates: that the export (mode grouping, projection on the lab bases, phase assembly, quantization)
    reproduces, on every output plane and for the declared, H and V input polarizations, the power of the Engine's own coherent
    superposition of the same traced beams (``validation/superposition.py``, a closed-form whole-plane integral of the Gaussian
    beams with their tilts, offsets and curvatures). Both sides come from the same trace: an error of the Engine's physics
    (a Jones model, a coating phase, an optical path length) is in both and is not detected; the tests compare the exported
    networks with independent textbook references for that."""

    tolerance: float = Field(description="Relative tolerance of the comparison with the Engine (each plane adds its quantization bound).")
    columns: tuple[ColumnCheckV1, ...]
    status: Literal["PASS"]


class ValidityIssueV1(StrictModel):
    code: str
    location: str | None = None
    message: str


class NetworkValidityV1(StrictModel):
    status: Literal["EXPORTABLE"] = "EXPORTABLE"
    warnings: tuple[ValidityIssueV1, ...] = ()


class LinearOpticalNetworkV1(StrictModel):
    """The document (module docstring). ``schema`` is serialized under that name (``by_alias=True``)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, ser_json_inf_nan="null", populate_by_name=True)

    schema_: Literal["opensci-linear-optical-network/v1"] = Field(NETWORK_SCHEMA, alias="schema")
    wavelength_nm: float
    provenance: NetworkProvenanceV1
    conventions: NetworkConventionsV1
    input_modes: tuple[InputModeV1, ...]
    output_modes: tuple[OutputModeV1, ...]
    transfer_matrix: TransferMatrixV1
    paths: tuple[NetworkPathV1, ...]
    elements: tuple[ElementTransferV1, ...]
    hints: NetworkHintsV1
    classical_check: ClassicalCheckV1
    validity: NetworkValidityV1
    contract_hash: str = Field(description="SHA-256 of the canonical JSON of this document without contract_hash (sorted keys, no "
                                           "whitespace; every number already quantized as conventions.quantization states).")
