"""The flat tables of a run's results: one column registry per export kind and the row builders.

Every table is a list of rows; a row is a ``dict`` whose keys are exactly the kind's column names, in column order.
Column names carry their unit as a suffix (``power_W``, ``beam_radius_x_mm``, ``aoi_deg``, ``opd_waves``); names without
a unit suffix are dimensionless, identifiers, flags or text (the ``unit`` of each column says which). Values are JSON
values: numbers, strings, booleans, or ``None`` where the quantity is not defined for that row (never NaN / infinity).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..contracts.results import ArrivalResult, ObservationResult, ValidationResult
from .basis import SAMPLED_LINE_FIELDS, row_basis_cell

Column = tuple[str, str, str]  # (name, unit, description)
VECTOR_AXES = ("x", "y", "z")
# First-order matrix rows (outputs) and columns (inputs) with the unit of each block: position outputs per position input
# are mm/mm, per angle input mm/rad; direction-cosine outputs per position input are 1/mm, per angle input 1/rad.
FIRST_ORDER_OUTPUTS = ("u", "v", "lu", "lv")
FIRST_ORDER_INPUTS = ("x0", "y0", "a0", "b0")


def _vector_columns(prefix: str, unit: str, what: str, frame: str = "design frame") -> list[Column]:
    suffix = f"_{unit}" if unit != "1" else ""
    return [(f"{prefix}_{axis}{suffix}", unit, f"{what}, {axis} component ({frame})") for axis in VECTOR_AXES]


def _first_order_columns() -> list[Column]:
    columns: list[Column] = []
    for out in FIRST_ORDER_OUTPUTS:
        for inp in FIRST_ORDER_INPUTS:
            position_out, angle_in = out in ("u", "v"), inp in ("a0", "b0")
            unit = {(True, False): "1", (True, True): "mm/rad", (False, False): "1/mm", (False, True): "1/rad"}[(position_out, angle_in)]
            suffix = {"1": "", "mm/rad": "_mm_per_rad", "1/mm": "_per_mm", "1/rad": "_per_rad"}[unit]
            columns.append((f"fo_d{out}_d{inp}{suffix}", unit, f"first-order ray-transfer matrix element d {out} / d {inp} "
                            "(source plane -> this crossing; null when not available)"))
    return columns


#: The ``value_basis`` column: what a number of the row is where it is not simply the model's value.
VALUE_BASIS_TEXT = ("what a number of this row is where it is not simply the model's value: the value_basis entries keyed by "
                    "column name (kind MODEL, LOWER_BOUND, UPPER_BOUND or ESTIMATE, with the parts omitted, inputs assumed and "
                    "validity checks); in CSV its JSON form; empty: every number is the model's value")

MONITOR_COLUMNS: tuple[Column, ...] = (
    ("observation_id", "", "observation plane (monitor) id"),
    ("status", "", "diagnostic severity of the arriving beams (acceptance, capture, beam-model validity), and on a plane no "
                   "traced beam reaches of whether light may arrive; whether the power is known: quantity_state"),
    ("pose_frame", "", "DESIGN or DEVICE_LOCAL (the plane pose was given relative to device_id)"),
    ("device_id", "", "component the plane is attached to (empty: none)"),
    ("absorbing", "", "true: the plane absorbs the beams that count as arriving"),
    ("absorbing_extent", "", "NONE (transparent monitor), ACTIVE_AREA or UNBOUNDED_PLANE"),
    *_vector_columns("origin", "mm", "plane origin"),
    *_vector_columns("u_axis", "1", "plane u axis (unit vector)"),
    *_vector_columns("v_axis", "1", "plane v axis (unit vector)"),
    *_vector_columns("normal", "1", "plane normal n = u x v (unit vector)"),
    ("aperture_shape", "", "circle, rectangle or empty (no active area: the whole plane)"),
    ("aperture_radius_mm", "mm", "active-area radius (circle)"),
    ("aperture_half_width_mm", "mm", "active-area half extent along u (rectangle)"),
    ("aperture_half_height_mm", "mm", "active-area half extent along v (rectangle)"),
    ("aperture_center_u_mm", "mm", "active-area centre offset along u"),
    ("aperture_center_v_mm", "mm", "active-area centre offset along v"),
    ("acceptance_half_angle_deg", "deg", "acceptance cone half angle (empty: none)"),
    ("medium", "", "medium declared at the plane"),
    ("arrival_count", "1", "beam crossings that count as arriving (inside the active area)"),
    ("outside_active_area_count", "1", "beam crossings outside the active area (they passed on)"),
    ("superposition_count", "1", "wavelengths at which two or more beams arrive (coherent sums)"),
    ("total_power_W", "W", "power the plane receives: the arriving beams' detected power on an absorbing plane, their power on a "
                           "transparent one (empty: no traced beam arrives and the power is not known)"),
    ("power_basis", "", "INCOHERENT_SUM: total_power_W adds the beams' powers, while the superpositions table states the coherent power "
                        "of the same beams; empty where no coherent quantity exists on the plane"),
    ("power_complete", "", "true: all launched power was traced to a known end, so total_power_W (0 included) is the model's value; "
                           "false: untraced_power_W was not followed and total_power_W counts the traced light only"),
    ("power_upper_bound_W", "W", "supported upper bound including omitted faint light; null when unavailable"),
    ("untraced_power_W", "W", "launched power whose fate the trace does not follow (trace budget, interaction limit, out of model scope, "
                              "power floor) plus, on a plane a line's traced beams miss, the power its first-order untraced light arrives with); "
                              "empty where quantity_state is NOT_COMPUTED"),
    ("message", "", "status message"),
    ("quantity_state", "", "whether the plane's power is a known quantity: KNOWN, KNOWN_ZERO, BOUNDED, UNKNOWN, OUT_OF_MODEL_SCOPE (the "
                           "power of a beam here, or of light scattered out of one, is out of model scope) or NOT_COMPUTED (the "
                           "evaluation did not run the plane bounds); read it, not total_power_W alone"),
    ("value_basis", "", VALUE_BASIS_TEXT),
)

ARRIVAL_COLUMNS: tuple[Column, ...] = (
    ("observation_id", "", "observation plane (monitor) id"),
    ("source_id", "", "source of the beam"),
    ("path_id", "", "beam path id (branches: '<parent>.<n>')"),
    ("parent_path_id", "", "path this branch split from (empty: a source path)"),
    ("spectral_index", "1", "index of the source's spectral line"),
    ("wavelength_nm", "nm", "vacuum wavelength"),
    ("crossing_index", "1", "0 for the path's first crossing of this plane, 1 for the second, ..."),
    *_vector_columns("position", "mm", "chief-ray hit point"),
    *_vector_columns("direction", "1", "chief-ray propagation direction (unit vector)"),
    ("u_mm", "mm", "chief-ray hit point along the plane's u axis"),
    ("v_mm", "mm", "chief-ray hit point along the plane's v axis"),
    ("direction_u", "1", "chief-ray direction cosine along u"),
    ("direction_v", "1", "chief-ray direction cosine along v"),
    ("direction_n", "1", "chief-ray direction cosine along the plane normal (negative: travelling against n)"),
    ("aoi_deg", "deg", "angle between the chief ray and the plane normal"),
    ("power_W", "W", "beam power arriving (empty: not known, a grating order without a response that gives it)"),
    ("detected_power_W", "W", "power an absorbing plane detects from the beam (its share on the active area; 0 outside the acceptance cone); "
                              "empty on a transparent plane"),
    ("coincident_surface_id", "", "the physical surface whose chief-ray hit point this plane passes through; empty elsewhere"),
    ("coincidence_role", "", "ARRIVING: the beam's power at that surface before the interaction; RETURNING: the beam that leaves it back across the plane"),
    ("coincidence_event", "", "what that surface does to the arriving beam: REFLECTIVE, TRANSMISSIVE or ABSORBING"),
    ("crossing_power_upper_bound_W", "W", "auxiliary, on an ARRIVING beam at a flat surface tilted to the plane: the most power that crosses the literal plane before the surface; no target reads it"),
    ("geometric_length_mm", "mm", "geometric path length from the source"),
    ("optical_path_length_mm", "mm", "optical path length from the source (sum of n x length); empty behind an ideal or tabulated thin lens, a Faraday "
                                     "rotator or a diffraction grating"),
    ("phase_rad", "rad", "plane-wave phase k0 OPL + arg(Ex), or + arg(Ey) where |Ey|^2 exceeds |Ex|^2 by more than rel 1e-9, "
                       "wrapped, in steps of 1e-6 rad; empty without a "
                       "phase reference and behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating"),
    ("jones_ex_re", "1", "Jones vector Ex (along basis e1), real part (|E|^2 relative to the source's)"),
    ("jones_ex_im", "1", "Jones vector Ex, imaginary part"),
    ("jones_ey_re", "1", "Jones vector Ey (along basis e2), real part"),
    ("jones_ey_im", "1", "Jones vector Ey, imaginary part"),
    *_vector_columns("basis_e1", "1", "polarization basis vector e1"),
    *_vector_columns("basis_e2", "1", "polarization basis vector e2 (e1 x e2 = direction)"),
    ("stokes_s1", "1", "normalized Stokes S1/S0 in the basis (e1, e2)"),
    ("stokes_s2", "1", "normalized Stokes S2/S0"),
    ("stokes_s3", "1", "normalized Stokes S3/S0 (-1 right circular, +1 left circular)"),
    ("polarization_ellipticity_deg", "deg", "ellipticity angle asin(s3)/2"),
    ("polarization_handedness", "", "LINEAR, RIGHT or LEFT"),
    ("polarization_azimuth_deg", "deg", "major axis projected on the plane, from u toward v, in [0, 180); empty for circular light"),
    ("gaussian_valid", "", "true when the paraxial Gaussian state is within its model scope"),
    ("gaussian_status", "", "status of the Gaussian state"),
    ("gaussian_reason", "", "why the Gaussian state is not valid (issue code)"),
    ("medium_index", "1", "refractive index of the medium at the plane (from the Gaussian state)"),
    ("beam_radius_x_mm", "mm", "1/e^2 intensity radius along the beam's Gaussian axis x (e1 of the Gaussian state)"),
    ("beam_radius_y_mm", "mm", "1/e^2 intensity radius along the beam's Gaussian axis y"),
    ("waist_radius_x_mm", "mm", "1/e^2 waist radius, x"),
    ("waist_radius_y_mm", "mm", "1/e^2 waist radius, y"),
    ("waist_distance_x_mm", "mm", "signed distance waist -> plane along the beam (positive: waist upstream), x"),
    ("waist_distance_y_mm", "mm", "signed distance waist -> plane along the beam, y"),
    ("rayleigh_range_x_mm", "mm", "Rayleigh range, x"),
    ("rayleigh_range_y_mm", "mm", "Rayleigh range, y"),
    ("far_field_half_angle_x_rad", "rad", "far-field divergence half angle, x"),
    ("far_field_half_angle_y_rad", "rad", "far-field divergence half angle, y"),
    ("curvature_radius_x_mm", "mm", "wavefront radius of curvature, x (empty: flat)"),
    ("curvature_radius_y_mm", "mm", "wavefront radius of curvature, y (empty: flat)"),
    ("q_x_re_mm", "mm", "complex beam parameter q_x, real part (= waist_distance_x_mm)"),
    ("q_x_im_mm", "mm", "complex beam parameter q_x, imaginary part (= rayleigh_range_x_mm)"),
    ("q_y_re_mm", "mm", "complex beam parameter q_y, real part"),
    ("q_y_im_mm", "mm", "complex beam parameter q_y, imaginary part"),
    ("m2_x", "1", "beam quality factor M^2, x"),
    ("m2_y", "1", "beam quality factor M^2, y"),
    *_vector_columns("gaussian_e1", "1", "Gaussian axis x"),
    *_vector_columns("gaussian_e2", "1", "Gaussian axis y"),
    ("within_acceptance", "", "chief ray inside the acceptance cone (empty: no cone declared)"),
    ("acceptance_margin_deg", "deg", "acceptance half angle minus AOI"),
    ("active_area_power_fraction", "1", "Gaussian power fraction inside the active area"),
    ("capture_margin_mm", "mm", "distance from the 1/e^2 beam ellipse to the active-area edge (negative: clipped)"),
    ("pupil_ray_count", "1", "pupil sample rays launched for this beam"),
    ("pupil_rays_arrived", "1", "pupil sample rays that reached the plane with the chief"),
    ("pupil_centroid_u_mm", "mm", "centroid of the pupil-ray hit points along u"),
    ("pupil_centroid_v_mm", "mm", "centroid of the pupil-ray hit points along v"),
    ("pupil_rms_radius_mm", "mm", "RMS radius of the pupil-ray hit points about their centroid"),
    ("ray_opl_defined", "", "per-ray optical path lengths are defined (false after an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating)"),
    ("coherence_group", "", "beams of one coherence group are mutually coherent"),
    ("coherence_length_mm", "mm", "declared coherence length of the source (empty: undeclared)"),
    *_first_order_columns(),
    ("fo_reflection_count", "1", "reflections on the path up to this plane"),
    ("fo_handedness_preserved", "", "true: an even number of reflections (image not mirrored)"),
    ("fo_image_rotation_deg", "deg", "image rotation seen along the propagation (only for a non-mirrored, regular map)"),
    ("fo_mirror_axis_deg", "deg", "in-plane mirror-line angle of a mirrored image"),
    ("fo_transverse_magnification", "1", "sqrt(|det A|) of the first-order position block"),
    ("fo_issue", "", "why the first-order transfer is not available"),
    ("lineage", "", "interactions from the source up to this plane, '<surface_id>:<event>' joined by ';'"),
    ("power_out_of_scope_at", "", "surfaces behind which the beam's power is out of model scope, joined by ';' (empty: none): power_W is "
                                  "then its chief ray's power, not the light's"),
    ("value_basis", "", VALUE_BASIS_TEXT),
)

RAY_COLUMNS: tuple[Column, ...] = (
    ("observation_id", "", "observation plane (monitor) id"),
    ("path_id", "", "beam path id"),
    ("source_id", "", "source of the beam"),
    ("wavelength_nm", "nm", "vacuum wavelength"),
    ("crossing_index", "1", "0 for the path's first crossing of this plane, 1 for the second, ..."),
    ("counted", "", "true: the beam counts as arriving at the plane's active area (else it crossed outside and passed on)"),
    ("ray_index", "1", "0 = chief ray, 1..N = pupil sample rays"),
    ("chief", "", "true for the chief ray"),
    ("ring_index", "1", "pupil ring of the ray (0 = innermost; empty for the chief)"),
    ("ring_position_index", "1", "position of the ray on its ring (empty for the chief)"),
    ("launch_offset_x", "1", "launch offset along the source frame x, in units of the source's 1/e^2 radius x"),
    ("launch_offset_y", "1", "launch offset along the source frame y, in units of the source's 1/e^2 radius y"),
    ("power_weight", "1", "share of the beam power the ray carries (equal for every pupil ray; empty for the chief)"),
    ("u_mm", "mm", "hit point along the plane's u axis"),
    ("v_mm", "mm", "hit point along the plane's v axis"),
    *_vector_columns("position", "mm", "hit point"),
    *_vector_columns("direction", "1", "propagation direction (unit vector)"),
    ("direction_u", "1", "direction cosine along u"),
    ("direction_v", "1", "direction cosine along v"),
    ("direction_n", "1", "direction cosine along the plane normal"),
    ("opl_mm", "mm", "optical path length from the source plane (every ray starts there with 0); empty where opl_defined is false"),
    ("opl_defined", "", "the ray's optical path length is defined here (false after an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating)"),
    ("ray_end", "", "how the ray ended: PATH_END, STOPPED_AT_SURFACE or NOT_FOLLOWED"),
    ("ray_end_surface_id", "", "surface where the ray stopped or left the beam"),
)

PATH_COLUMNS: tuple[Column, ...] = (
    ("path_id", "", "beam path id"),
    ("source_id", "", "source of the beam"),
    ("spectral_index", "1", "index of the source's spectral line"),
    ("wavelength_nm", "nm", "vacuum wavelength"),
    ("parent_path_id", "", "path this branch split from (empty: a source path)"),
    ("branch_surface_id", "", "surface where the branch split off"),
    ("branch_kind", "", "kind of branch (reflected / transmitted / ...)"),
    ("children", "", "paths that split from this one, joined by ';'"),
    ("depth", "1", "0 for a source path, 1 for its branches, ..."),
    ("segment_count", "1", "straight segments of the chief ray"),
    ("power_start_W", "W", "power at the start of the path, source or branch point (empty: not known)"),
    ("final_power_W", "W", "power at the end of the path (empty: not known)"),
    ("power_out_of_scope_at", "", "surfaces behind which the path's power is out of model scope, joined by ';' (empty: none): "
                                  "power_start_W / final_power_W from there on are its chief ray's, not the light's"),
    ("termination", "", "how the path ended (ESCAPED, DETECTED, ABSORBED, CLIPPED, ...)"),
    ("termination_surface_id", "", "surface where the path ended"),
    ("geometric_path_length_mm", "mm", "geometric length from the source to the path end (ancestors included)"),
    ("optical_path_length_mm", "mm", "optical path length from the source to the path end (ancestors included); empty behind an ideal or tabulated thin "
                                     "lens, a Faraday rotator or a diffraction grating"),
    ("observations", "", "observation planes the path crosses, in order, joined by ';'"),
    ("lineage", "", "interactions from the source, '<surface_id>:<event>' joined by ';'"),
)

SEGMENT_COLUMNS: tuple[Column, ...] = (
    ("path_id", "", "beam path id"),
    ("segment_index", "1", "0-based segment index along the path"),
    *_vector_columns("start", "mm", "segment start"),
    *_vector_columns("end", "mm", "segment end (empty: the beam leaves the scene)"),
    ("length_mm", "mm", "geometric length (empty: unbounded)"),
    ("medium_id", "", "medium of the segment"),
    ("refractive_index", "1", "refractive index of the medium at the path's wavelength"),
    ("optical_path_length_mm", "mm", "n x length"),
    ("power_start_W", "W", "power at the segment start"),
    ("power_end_W", "W", "power after the interaction at the segment end"),
    ("end_surface_id", "", "surface at the segment end"),
    ("end_event", "", "interaction at the segment end"),
    ("angle_of_incidence_deg", "deg", "angle of incidence at the segment end"),
    ("beam_radius_start_mm", "mm", "max(x, y) 1/e^2 beam radius at the start (empty: Gaussian state not valid)"),
    ("beam_radius_end_mm", "mm", "max(x, y) 1/e^2 beam radius at the end"),
)

POWER_BUDGET_COLUMNS: tuple[Column, ...] = (
    ("source_id", "", "source (empty on the TOTAL rows)"),
    ("spectral_index", "1", "spectral line of the source (empty on the TOTAL rows)"),
    ("wavelength_nm", "nm", "vacuum wavelength (empty on the TOTAL rows)"),
    ("category", "", "INPUT, SINK, OBSERVED (power counted at a monitor; not a sink) or RESIDUAL"),
    ("name", "", "sink kind (SINK), observation id (OBSERVED), empty otherwise"),
    ("power_W", "W", "power (empty on an OBSERVED row of a plane where the power of the line's light is out of model scope: "
                     "observed_out_of_scope_at)"),
    ("fraction_of_input", "1", "power / input power of the line"),
    ("conserved", "", "the line's ledger closes within its tolerance (INPUT and RESIDUAL rows)"),
)

SUPERPOSITION_COLUMNS: tuple[Column, ...] = (
    ("observation_id", "", "observation plane (monitor) id"),
    ("wavelength_nm", "nm", "vacuum wavelength of the superposed beams"),
    ("path_ids", "", "superposed beam paths, joined by ';'"),
    ("crossing_indices", "", "their crossing indices, joined by ';'"),
    ("incoherent_power_W", "W", "sum of the beams' powers"),
    ("total_power_W", "W", "power of the coherent sum through the whole plane (empty: it has no value)"),
    ("interference_power_W", "W", "total minus incoherent power"),
    ("total_power_min_W", "W", "the least total power the coatings' group-delay bounds allow, over every phase where total_power_W is empty"),
    ("total_power_max_W", "W", "the largest total power the coatings' group-delay bounds allow, over every phase where total_power_W is empty"),
    ("phase_statement", "", "where a coating whose pass differs states no phase: which passes leave the phase open and how to read the range"),
    ("status", "", "status of the superposition"),
    ("issue_codes", "", "issue codes, joined by ';'"),
    ("value_basis", "", VALUE_BASIS_TEXT),
)

WAVEFRONT_COLUMNS: tuple[Column, ...] = (
    ("ray_index", "1", "pupil ray index"),
    ("ring_index", "1", "pupil ring (0 = innermost)"),
    ("ring_position_index", "1", "position on the ring"),
    ("pupil_x", "1", "pupil coordinate: launch offset along the source frame x, in units of the source's 1/e^2 radius x"),
    ("pupil_y", "1", "pupil coordinate: launch offset along the source frame y, in units of the source's 1/e^2 radius y"),
    ("u_mm", "mm", "where the ray crosses the observation plane, along u"),
    ("v_mm", "mm", "where the ray crosses the observation plane, along v"),
    ("opd_waves", "waves", "optical path difference to the reference surface relative to the chief ray, piston removed, in "
                           "vacuum wavelengths (positive: longer optical path than the chief)"),
    ("opd_mm", "mm", "the same optical path difference in mm"),
)

IRRADIANCE_COLUMNS: tuple[Column, ...] = (
    ("u_mm", "mm", "grid point along the plane's u axis (a sampled map: the bin's centre)"),
    ("v_mm", "mm", "grid point along the plane's v axis (a sampled map: the bin's centre)"),
    ("irradiance_coherent_W_per_mm2", "W/mm^2", "power flux through the plane per unit area of the coherent sum (empty when "
                                                 "the mutual coherence is undefined, and for sampled rays, which have no phase relation)"),
    ("irradiance_incoherent_W_per_mm2", "W/mm^2", "sum of the beams' individual irradiances and the sampled rays' binned power per unit area (no interference)"),
    ("sample_count", "", "sampled-ray crossings in the bin (empty for a beam map)"),
)

SAMPLED_LIGHT_COLUMNS: tuple[Column, ...] = (
    ("observation_id", "", "the observation plane"),
    ("source_id", "", "the extended emitter"),
    ("spectral_index", "", "the source line (position in the emitter's lines)"),
    ("wavelength_nm", "nm", "vacuum wavelength of the line"),
    ("emitted_W", "W", "the line's total radiant power"),
    ("power_W", "W", "sum of the counted crossings' powers: on the active area (the whole plane without one) inside the acceptance cone"),
    ("collection_fraction", "1", "power_W / emitted_W"),
    ("sample_count", "", "rays the line declares"),
    ("samples_traced", "", "rays launched (fewer than sample_count only when the ray-step budget was reached)"),
    ("half_sample_count", "", "rays in the first half of the sequence"),
    ("collection_fraction_half_samples", "1", "the same fraction from the first half of the rays alone"),
    ("collection_fraction_difference", "1", "|collection_fraction - collection_fraction_half_samples|"),
    ("collection_fraction_sigma", "1", "standard error of collection_fraction if the rays were independent random samples (the stratified rays usually do better)"),
    ("crossing_count", "", "crossings of the plane by the line's rays"),
    ("counted_crossing_count", "", "of which counted"),
    ("power_out_of_scope_at", "", "light scattered out of a beam whose power is out of model scope behind these surfaces, joined by ';' "
                                  "(empty: none): its powers follow that beam's chief ray, not the light"),
    ("value_basis", "", VALUE_BASIS_TEXT),
)
SAMPLED_CROSSING_COLUMNS: tuple[Column, ...] = (
    ("observation_id", "", "the observation plane"),
    ("source_id", "", "the extended emitter"),
    ("spectral_index", "", "the source line"),
    ("wavelength_nm", "nm", "vacuum wavelength of the line"),
    ("sample_index", "", "the ray's position in its line's sequence (0-based; a branch keeps its sample's index)"),
    ("u_mm", "mm", "crossing point along the plane's u axis"),
    ("v_mm", "mm", "crossing point along the plane's v axis"),
    ("direction_u", "1", "direction cosine of the ray on the plane's u axis"),
    ("direction_v", "1", "direction cosine of the ray on the plane's v axis"),
    ("direction_n", "1", "direction cosine of the ray on the plane's normal"),
    ("power_W", "W", "power of the ray at the crossing; rays add incoherently, so a bin's irradiance is the sum of its rows' power_W over the bin's area"),
    ("counted", "", "the ray lands on the active area (the whole plane without one) inside the acceptance cone"),
    ("value_basis", "", VALUE_BASIS_TEXT + "; a crossing's power_W carries its line's power_W entry"),
)
METRIC_COLUMNS: tuple[Column, ...] = (
    ("name", "", "metric name (the metric of a target)"),
    ("unit", "", "unit of the value"),
    ("kind", "", "observation, pair, layout or control"),
    ("definition", "", "physical quantity and how it is computed"),
    ("sign_convention", "", "sign, operand order and the meaning of zero"),
    ("operands", "", "which beam is a and which b (pair metrics)"),
    ("reference_frame", "", "frame of coordinates / angles"),
    ("selectors", "", "target fields that select the beam(s), joined by ';'"),
    ("aggregation", "", "how several matching beams combine"),
    ("model_basis", "", "Engine model producing the value"),
    ("validity", "", "when the value is withheld"),
    ("version", "", "metric definition version"),
    ("result_fields", "", "where the value appears among a run's results ('<export kind>.<column>', 'monitor:<field>'), joined by ';'"),
)

UVN = ("u", "v", "n")


def _uvn_columns(prefix: str, unit: str, what: str) -> list[Column]:
    suffix = f"_{unit}" if unit != "1" else ""
    return [(f"{prefix}_{axis}{suffix}", unit, f"{what}, component along the plane's {axis} axis") for axis in UVN]


HANDOFF_COLUMNS: tuple[Column, ...] = (
    ("observation_id", "", "observation plane (monitor) id"),
    ("path_id", "", "beam path id"),
    ("source_id", "", "source of the beam"),
    ("spectral_index", "1", "index of the source's spectral line"),
    ("wavelength_nm", "nm", "vacuum wavelength"),
    ("crossing_index", "1", "0 for the path's first crossing of this plane, 1 for the second, ..."),
    ("medium_index", "1", "refractive index of the medium at the plane"),
    ("power_W", "W", "beam power at the plane (exact Gaussian power after every aperture it passed; empty: not known)"),
    ("u_mm", "mm", "chief-ray hit point along u"),
    ("v_mm", "mm", "chief-ray hit point along v"),
    *_uvn_columns("direction", "1", "chief-ray direction"),
    ("aoi_deg", "deg", "angle between the chief ray and the plane normal"),
    ("q_x_re_mm", "mm", "complex beam parameter q_x, real part: signed distance waist -> plane (positive: waist upstream)"),
    ("q_x_im_mm", "mm", "q_x, imaginary part: Rayleigh range in the medium"),
    ("q_y_re_mm", "mm", "q_y, real part"),
    ("q_y_im_mm", "mm", "q_y, imaginary part"),
    ("beam_radius_x_mm", "mm", "1/e^2 intensity radius along the Gaussian axis x"),
    ("beam_radius_y_mm", "mm", "1/e^2 intensity radius along the Gaussian axis y"),
    ("waist_radius_x_mm", "mm", "1/e^2 waist radius, x"),
    ("waist_radius_y_mm", "mm", "1/e^2 waist radius, y"),
    ("curvature_radius_x_mm", "mm", "wavefront radius of curvature, x (positive: diverging; empty: flat)"),
    ("curvature_radius_y_mm", "mm", "wavefront radius of curvature, y"),
    ("m2_x", "1", "beam quality factor M^2, x"),
    ("m2_y", "1", "beam quality factor M^2, y"),
    *_uvn_columns("gaussian_axis_x", "1", "Gaussian axis x (unit vector)"),
    *_uvn_columns("gaussian_axis_y", "1", "Gaussian axis y (unit vector)"),
    ("jones_1_re", "1", "Jones vector component along basis_e1, real part"),
    ("jones_1_im", "1", "Jones vector component along basis_e1, imaginary part"),
    ("jones_2_re", "1", "Jones vector component along basis_e2, real part"),
    ("jones_2_im", "1", "Jones vector component along basis_e2, imaginary part"),
    *_uvn_columns("basis_e1", "1", "polarization basis vector e1"),
    *_uvn_columns("basis_e2", "1", "polarization basis vector e2 (e1 x e2 = direction)"),
    ("phase_rad", "rad", "field phase at the chief ray (empty without a phase reference or when phase_defined is false)"),
    ("optical_path_length_mm", "mm", "optical path length from the source (empty when phase_defined is false)"),
    ("phase_defined", "", "false behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating: its model states no optical path, so phase and optical path "
                          "length are not defined"),
    ("beam_parameters", "", "TEM00, EMBEDDED_GAUSSIAN (M^2 > 1), TRUNCATED_GAUSSIAN (an aperture upstream cut more than the run's "
                            "gaussian_clip_warn_fraction of the beam), UNAVAILABLE or "
                            "POWER_OUT_OF_MODEL_SCOPE (the beam's power is out of model scope: power_W empty) or POWER_TAKEN_BEFORE_PLANE "
                            "(surfaces take part of the beam before the plane: power_W empty)"),
    ("tem00_field_applicable", "", "true: a TEM00 field follows from q, power and Jones vector under the stated convention"),
    ("truncated_at", "", "surfaces upstream where an aperture removed power from this beam, joined by ';'"),
    ("truncated_power_fraction", "1", "sum over those apertures of the share of the arriving power each removed (0: none; empty: a "
                                      "share is not recorded)"),
    ("field_intensity_ripple", "1", "sum of 2 sqrt(f) + f over those shares f: the largest relative change of the on-axis intensity "
                                    "behind a centred round hard edge cutting that share"),
    ("power_out_of_scope_at", "", "surfaces behind which the beam's power is out of model scope, joined by ';' (empty: none)"),
    ("power_taken_before_at", "", "surfaces that take part of the beam before the plane, joined by ';' (empty: none)"),
    ("model_scope", "", "the model-scope findings that reach this beam, as code@element joined by ';' (empty: none): a published model "
                        "used outside its validity"),
    ("value_basis", "", VALUE_BASIS_TEXT + "; keyed by the handoff document's names (power_W, chief_ray.uv_mm, gaussian.*, ...)"),
)

EXPOSURE_COLUMNS: tuple[Column, ...] = (
    ("surface_id", "", "element surface the beam meets (<component>/<surface>)"),
    ("element_id", "", "component the surface belongs to"),
    ("path_id", "", "beam path"),
    ("source_id", "", "source of the beam"),
    ("spectral_index", "1", "source spectral line index"),
    ("wavelength_nm", "nm", "vacuum wavelength of the line"),
    ("pass_index", "1", "which arrival of this path at the surface (0 = first)"),
    ("incident_power_W", "W", "power of the beam arriving at the surface (before its clear aperture)"),
    ("aoi_deg", "deg", "chief ray's angle of incidence on the local surface normal"),
    ("beam_radius_x_mm", "mm", "1/e^2 radius of the Gaussian beam along its x axis at the surface (empty: not known)"),
    ("beam_radius_y_mm", "mm", "1/e^2 radius along the beam's y axis"),
    ("footprint_major_mm", "mm", "semi-major axis of the 1/e^2 footprint on the surface"),
    ("footprint_minor_mm", "mm", "semi-minor axis of the 1/e^2 footprint on the surface"),
    ("peak_irradiance_W_per_cm2", "W/cm^2", "peak irradiance 2P / (pi w_x w_y) in the beam's transverse plane"),
    ("peak_irradiance_surface_W_per_cm2", "W/cm^2", "peak irradiance on the surface (transverse value x cos(aoi))"),
    ("linear_power_density_W_per_cm", "W/cm", "P / d with d the 1/e^2 diameter of the narrower beam axis"),
    ("pulse_energy_J", "J", "pulse energy (pulsed source: power / repetition rate)"),
    ("peak_fluence_J_per_cm2", "J/cm^2", "peak fluence 2E / (pi w_x w_y) in the transverse plane (pulsed source)"),
    ("status", "", "PASS; UNKNOWN when the beam has no valid Gaussian state there; OUT_OF_MODEL_SCOPE behind a response model "
                   "not validated for the beam size"),
    ("reason", "", "why the beam-size quantities are not known"),
    ("light", "", "traced; untraced: light the trace does not follow that reaches the surface (followed in a run with power "
                  "ratings); beside: a beam whose chief ray misses the surface while its tail reaches it (the values at the "
                  "nearest point of the support) -- the last two are not in the surface values, they bound the comparisons"),
    ("surface_incident_power_W", "W", "incident power of the traced beams on the surface: their sum, or with mutually coherent "
                                      "beams the largest power their superposition can carry"),
    ("surface_peak_irradiance_W_per_cm2", "W/cm^2", "peak irradiance of the traced beams on the surface (upper bound: peaks taken "
                                                    "as coincident, coherent beams in phase)"),
    ("surface_complete", "", "false: the surface's exposure is a range or partly unknown (see the power-ratings bounds)"),
    ("rating_status", "", "WITHIN_RATING, EXCEEDS_RATING, RATING_MISSING, CONDITIONS_MISMATCH, CONDITIONS_NOT_VERIFIED or UNKNOWN "
                          "(the surface)"),
    ("value_basis", "", VALUE_BASIS_TEXT),
)

POWER_RATING_COLUMNS: tuple[Column, ...] = (
    ("surface_id", "", "element surface (<component>/<surface>)"),
    ("element_id", "", "component the surface belongs to"),
    ("rating_index", "1", "index of the rating in the element's power ratings"),
    ("regime", "", "CW or PULSED"),
    ("unit", "", "W/cm, W/cm2 or J/cm2"),
    ("rating_value", "", "the rating, in its unit"),
    ("rating_wavelength_nm", "nm", "test wavelength of the rating"),
    ("rating_fact_status", "", "fact status of the rating (SPECIFIED, DECLARED, ...)"),
    ("origin", "", "product or model that states the rating"),
    ("applicability", "", "what the rating says about the damage threshold of the light here: AS_RATED (every stated condition "
                          "verified), THRESHOLD_AT_LEAST_RATED, THRESHOLD_AT_MOST_RATED (a condition differs in that direction: a "
                          "test beam larger / smaller than the beams here, at least as many pulses per site as the test), "
                          "NOT_VERIFIED (a condition cannot be checked) or NOT_APPLICABLE (a condition differs with no "
                          "established relationship)"),
    ("conditions", "", "each condition as condition=STATE (regime, wavelength, pulse_duration, repetition_rate, beam_diameter, "
                       "pulse_count, polarization, value_definition; DIFFERS with its effect AT_LEAST / AT_MOST / NONE, NOT_VERIFIABLE with its "
                       "reason), joined by ';'"),
    ("conditions_match", "", "every condition the rating states is verified against the light on the surface (applicability "
                             "AS_RATED)"),
    ("mismatches", "", "the conditions that differ (what differs), joined by ';'"),
    ("unverified_conditions", "", "the conditions that cannot be checked (a pulse count whose source states no total_pulses, a "
                                  "test beam diameter between the readings of an unstated definition, light of unknown size or "
                                  "state, a polarization stated as text, ...), joined by ';'"),
    ("notes", "", "what the comparison used (P / d diameter definition, polarization found), joined by ';'"),
    ("exposure_value", "", "the surface's exposure in the rating's unit: exact without a value_basis entry, else the "
                           "followed beams' exposure beside the bounds (empty: no single number)"),
    ("exposure_lower_bound", "", "a lower bound of the exposure when it is not known exactly (empty: none above 0)"),
    ("exposure_upper_bound", "", "an upper bound of the exposure when it is not known exactly (empty: unbounded)"),
    ("utilization", "1", "exposure / rated value: decided without a value_basis entry (exact exposure, or its lower bound / "
                         "rated value above 1); with one, the ratio to the rated value as stated and what it takes for granted"),
    ("status", "", "WITHIN_RATING, EXCEEDS_RATING (decided in a direction the applicability allows), UNKNOWN, "
                   "CONDITIONS_NOT_VERIFIED or CONDITIONS_MISMATCH (the rating does not apply)"),
    ("basis", "", "where the rating comes from (a manufacturer's specification, a test report, the customer's statement)"),
    ("rated_conditions", "", "the test conditions the rating states, each as condition=value unit (wavelength, pulse_duration, "
                             "repetition_rate, beam_diameter with its 1/e^2 readings, pulse_count, polarization, "
                             "value_definition), joined by ';'"),
    ("design_values", "", "the light's value for each condition the rating states (a range as low..high), joined by ';'"),
    ("rated_threshold_low", "", "the damage threshold the rated value establishes for the peak exposure, in the rating's unit "
                                "(PEAK: the value; TOTAL_PER_1E2_AREA: twice it; W/cm: the value); empty: none -- a J/cm2 or "
                                "W/cm2 value whose definition is not stated, or a typical value"),
    ("rated_threshold_high", "", "the same threshold's upper end (equal to rated_threshold_low; empty: none)"),
    ("value_basis", "", VALUE_BASIS_TEXT),
)

GRATING_ORDER_COLUMNS: tuple[Column, ...] = (
    ("element_id", "", "the grating's component"),
    ("surface_id", "", "the grating's surface"),
    ("path_id", "", "the beam path that met the grating"),
    ("source_id", "", "its source"),
    ("wavelength_nm", "nm", "vacuum wavelength"),
    ("incidence_deg", "deg", "the chief ray's signed angle of incidence in the plane of the grating vector and the normal"),
    ("conical_angle_deg", "deg", "how far the chief ray arrives from the plane perpendicular to the grooves"),
    ("order", "", "the diffraction order"),
    ("propagation", "", "PROPAGATING, or NONPROPAGATING: its tangential wavevector exceeds the exit medium's and it is not traced"),
    ("followed", "", "the order was followed as a beam path"),
    ("order_path_id", "", "the beam path that carries the order (empty: not followed)"),
    ("direction_x", "1", "the order's direction, design frame (empty: nonpropagating)"),
    ("direction_y", "1", "the order's direction, design frame"),
    ("direction_z", "1", "the order's direction, design frame"),
    ("exit_angle_deg", "deg", "the order's signed angle in the plane of the grating vector and the normal, positive toward the grating vector"),
    ("power_state", "", "KNOWN, BOUNDED (power_W empty, at most power_upper_bound_W), OUT_OF_MODEL_SCOPE (the incident beam's power is "
                        "out of model scope: power_W and the bound empty) or NOT_APPLICABLE"),
    ("power_W", "W", "the order's power as the response gives it (empty: not known)"),
    ("power_upper_bound_W", "W", "the most power it can carry where its power is not known"),
    ("power_out_of_scope_at", "", "surfaces behind which the incident beam's power is out of model scope, joined by ';' (empty: none)"),
    ("efficiency", "1", "the share of the incident power the response gives it, or the mean an unpolarized response states"),
    ("relative_response", "1", "a relative response's value, not a share of the incident power"),
    ("polarization_state", "", "KNOWN, UNKNOWN or NOT_APPLICABLE"),
    ("phase_state", "", "UNKNOWN (efficiencies state no phase) or NOT_APPLICABLE"),
    ("response_basis", "", "NONE, MEASURED, INDEPENDENTLY_COMPUTED or ASSUMED"),
)
NONLINEAR_COLUMNS: tuple[Column, ...] = (
    ("element_id", "", "the crystal (component instance id)"),
    ("surface_id", "", "the face the fundamental entered through"),
    ("path_id", "", "the fundamental's path"),
    ("source_id", "", "source"),
    ("spectral_index", "", "the fundamental's spectral line"),
    ("wavelength_nm", "nm", "the fundamental's vacuum wavelength"),
    ("harmonic_spectral_index", "", "the harmonic's spectral line (source line index + the source's line count)"),
    ("harmonic_wavelength_nm", "nm", "the harmonic's vacuum wavelength"),
    ("harmonic_path_id", "", "the path that carries the harmonic (absent: none generated)"),
    ("fundamental_waves", "", "crystal waves of the two fundamental photons: oo, ee, oe"),
    ("harmonic_wave", "", "crystal wave of the harmonic: o, e"),
    ("focusing", "", "near_field or boyd_kleinman"),
    ("status", "", "PASS, UNKNOWN (unstated bulk absorption), or OUT_OF_MODEL_SCOPE"),
    ("reason", "", "why the conversion is unknown or out of model scope"),
    ("fundamental_power_W", "W", "fundamental power entering the crystal, if known"),
    ("converting_power_W", "W", "of it, the power in the waves the interaction converts, if known"),
    ("harmonic_power_W", "W", "harmonic power generated in the crystal, at its exit face (after its absorption, if any)"),
    ("conversion_efficiency", "1", "harmonic_power_W / fundamental_power_W"),
    ("length_mm", "mm", "crystal length between its faces"),
    ("path_length_mm", "mm", "the driving wave normal's path through the crystal"),
    ("d_eff_pm_per_V", "pm/V", "effective coefficient used (quasi-phase matching: times 2/(m pi))"),
    ("delta_k_per_mm", "1/mm", "phase mismatch k_2 - k_a - k_b - K_G of the waves' normal components"),
    ("phase_mismatch_rad", "rad", "dk L / 2"),
    ("plane_wave_factor", "1", "sinc^2(dk L / 2)"),
    ("overlap_factor", "1", "Gaussian-beam overlap integral H (walk-off, focusing)"),
    ("focusing_parameter", "1", "L / b, b the beam's confocal parameter in the crystal (smaller axis)"),
    ("walkoff_mrad", "mrad", "angle between the harmonic's and the fundamental's energy flow"),
    ("depletion_estimate", "1", "relative overstatement of the harmonic by the undepleted-pump model"),
    ("near_field_error", "1", "difference between the near-field and the focused result, relative to the larger of the focused "
     "result and its value at dk = 0 (near field only)"),
    ("quadrature_error", "1", "relative quadrature error of the overlap integral"),
    ("angular_acceptance_mrad", "mrad", "FWHM of sinc^2(dk L / 2) in the internal angle, plane of wave normal and optic axis"),
    ("temperature_acceptance_K", "K", "FWHM of sinc^2(dk L / 2) in temperature (stated thermo-optic coefficients only)"),
    ("spectral_acceptance_nm", "nm", "FWHM of sinc^2(dk L / 2) in the fundamental's vacuum wavelength"),
    ("temperature_c", "degC", "crystal temperature (stated thermo-optic coefficients only)"),
    ("qpm_period_um", "um", "quasi-phase-matching period at the reference temperature"),
)
COLUMNS: dict[str, tuple[Column, ...]] = {
    "monitors": MONITOR_COLUMNS,
    "arrivals": ARRIVAL_COLUMNS,
    "rays": RAY_COLUMNS,
    "paths": PATH_COLUMNS,
    "segments": SEGMENT_COLUMNS,
    "power-budget": POWER_BUDGET_COLUMNS,
    "superpositions": SUPERPOSITION_COLUMNS,
    "wavefront": WAVEFRONT_COLUMNS,
    "irradiance": IRRADIANCE_COLUMNS,
    "sampled-light": SAMPLED_LIGHT_COLUMNS,
    "sampled-crossings": SAMPLED_CROSSING_COLUMNS,
    "metrics": METRIC_COLUMNS,
    "handoff": HANDOFF_COLUMNS,
    "exposures": EXPOSURE_COLUMNS,
    "power-ratings": POWER_RATING_COLUMNS,
    "grating-orders": GRATING_ORDER_COLUMNS,
    "nonlinear": NONLINEAR_COLUMNS,
}
LIST_SEPARATOR = ";"


def columns_of(kind: str) -> list[dict[str, str]]:
    """The documented columns of one export kind: ``[{"name", "unit", "description"}]`` in column order."""

    return [{"name": name, "unit": unit, "description": text} for name, unit, text in COLUMNS[kind]]


def _vec(prefix: str, vector: Any, unit: str) -> dict[str, Any]:
    suffix = f"_{unit}" if unit != "1" else ""
    values = (None, None, None) if vector is None else tuple(float(v) for v in vector)
    return {f"{prefix}_{axis}{suffix}": value for axis, value in zip(VECTOR_AXES, values)}


def _row(kind: str, values: dict[str, Any]) -> dict[str, Any]:
    """``values`` in the column order of ``kind`` (every column present; a missing key is a programming error)."""

    return {name: values[name] for name, _unit, _text in COLUMNS[kind]}


def monitor_row(observation: ObservationResult, quantity_state: str | None = None) -> dict[str, Any]:
    """One plane's row of the monitors export. ``quantity_state``: the plane's state as the results report it (the Local
    Validation projection's, ``OUT_OF_MODEL_SCOPE`` under a model-scope finding); without it, the Engine's own."""

    plane = observation.plane
    aperture = plane.aperture if plane is not None else None
    values: dict[str, Any] = {
        "observation_id": observation.observation_id, "status": observation.status.value,
        "pose_frame": plane.pose_frame if plane else None, "device_id": plane.device_id if plane else None,
        "absorbing": plane.absorbing if plane else None, "absorbing_extent": plane.absorbing_extent if plane else None,
        **_vec("origin", plane.origin_mm if plane else None, "mm"), **_vec("u_axis", plane.u_axis if plane else None, "1"),
        **_vec("v_axis", plane.v_axis if plane else None, "1"), **_vec("normal", plane.normal if plane else None, "1"),
        "aperture_shape": aperture.shape if aperture is not None else None,
        "aperture_radius_mm": getattr(aperture, "radius_mm", None),
        "aperture_half_width_mm": getattr(aperture, "half_width_mm", None),
        "aperture_half_height_mm": getattr(aperture, "half_height_mm", None),
        "aperture_center_u_mm": aperture.center_offset_mm[0] if aperture is not None else None,
        "aperture_center_v_mm": aperture.center_offset_mm[1] if aperture is not None else None,
        "acceptance_half_angle_deg": plane.acceptance_half_angle_deg if plane else None,
        "medium": plane.medium if plane else None,
        "arrival_count": len(observation.arrivals), "outside_active_area_count": len(observation.outside_active_area),
        "superposition_count": len(observation.superpositions), "total_power_W": observation.total_power_W,
        "power_basis": observation.power_basis,
        "power_complete": observation.power_complete, "untraced_power_W": observation.untraced_power_W,
        "power_upper_bound_W": observation.power_upper_bound_W,
        "message": observation.message,
        "quantity_state": quantity_state if quantity_state is not None
        else None if observation.quantity_state is None else observation.quantity_state.value,
        "value_basis": row_basis_cell([("total_power_W", observation, "total_power_W")]),
    }
    return _row("monitors", values)


def _first_order(arrival: ArrivalResult) -> dict[str, Any]:
    names = [name for name, _u, _t in _first_order_columns()]
    fo = arrival.first_order
    flat = [None] * len(names) if fo is None else [float(v) for row in fo.matrix for v in row]
    values = dict(zip(names, flat))
    values.update({
        "fo_reflection_count": fo.reflection_count if fo else None, "fo_handedness_preserved": fo.handedness_preserved if fo else None,
        "fo_image_rotation_deg": fo.image_rotation_deg if fo else None, "fo_mirror_axis_deg": fo.mirror_axis_deg if fo else None,
        "fo_transverse_magnification": fo.transverse_magnification if fo else None, "fo_issue": arrival.first_order_issue,
    })
    return values


#: The Gaussian columns of an arrival row whose number a ``value_basis`` entry of the beam's Gaussian state may qualify.
GAUSSIAN_VALUE_COLUMNS = ("beam_radius_x_mm", "beam_radius_y_mm", "waist_radius_x_mm", "waist_radius_y_mm", "waist_distance_x_mm",
                          "waist_distance_y_mm", "rayleigh_range_x_mm", "rayleigh_range_y_mm", "far_field_half_angle_x_rad",
                          "far_field_half_angle_y_rad", "curvature_radius_x_mm", "curvature_radius_y_mm", "m2_x", "m2_y")
#: The columns of an exposure row that are numbers of the beam's hit.
EXPOSURE_VALUE_COLUMNS = ("incident_power_W", "beam_radius_x_mm", "beam_radius_y_mm", "peak_irradiance_W_per_cm2",
                          "peak_irradiance_surface_W_per_cm2", "linear_power_density_W_per_cm", "pulse_energy_J",
                          "peak_fluence_J_per_cm2")


def _gaussian(arrival: ArrivalResult) -> dict[str, Any]:
    g = arrival.beam.gaussian
    get = (lambda name: getattr(g, name)) if g is not None else (lambda name: None)  # noqa: E731
    q_x, q_y = get("q_x_mm"), get("q_y_mm")
    return {
        "gaussian_valid": g.valid if g is not None else None, "gaussian_status": g.status.value if g is not None else None,
        "gaussian_reason": get("reason"), "medium_index": get("medium_index"),
        **{name: get(name) for name in ("beam_radius_x_mm", "beam_radius_y_mm", "waist_radius_x_mm", "waist_radius_y_mm",
                                         "waist_distance_x_mm", "waist_distance_y_mm", "rayleigh_range_x_mm", "rayleigh_range_y_mm",
                                         "far_field_half_angle_x_rad", "far_field_half_angle_y_rad", "curvature_radius_x_mm",
                                         "curvature_radius_y_mm")},
        "q_x_re_mm": q_x[0] if q_x else None, "q_x_im_mm": q_x[1] if q_x else None,
        "q_y_re_mm": q_y[0] if q_y else None, "q_y_im_mm": q_y[1] if q_y else None,
        "m2_x": get("m2_x"), "m2_y": get("m2_y"), **_vec("gaussian_e1", get("e1"), "1"), **_vec("gaussian_e2", get("e2"), "1"),
    }


#: The arrivals table's columns whose numbers a model-scope finding may qualify beside the power (``basis.scoped_observation``:
#: the arrival's ``value_basis`` key each column reads; the Gaussian columns read the Gaussian state's own entries).
ARRIVAL_BASIS_COLUMNS: tuple[tuple[str, str], ...] = (
    *((f"position_{axis}_mm", "beam.position_mm") for axis in VECTOR_AXES), *((f"direction_{axis}", "beam.direction") for axis in VECTOR_AXES),
    ("wavelength_nm", "beam.wavelength_nm"), ("u_mm", "uv_mm"), ("v_mm", "uv_mm"), ("direction_u", "direction_uvn"),
    ("direction_v", "direction_uvn"), ("direction_n", "direction_uvn"), ("aoi_deg", "aoi_deg"),
    ("geometric_length_mm", "beam.geometric_length_mm"), ("optical_path_length_mm", "beam.optical_path_length_mm"),
    ("phase_rad", "beam.phase_rad"), ("stokes_s1", "polarization_state.s1"), ("stokes_s2", "polarization_state.s2"),
    ("stokes_s3", "polarization_state.s3"), ("polarization_ellipticity_deg", "polarization_state.ellipticity_deg"),
    ("polarization_azimuth_deg", "polarization_state.azimuth_in_plane_deg"), ("acceptance_margin_deg", "acceptance_margin_deg"),
    ("active_area_power_fraction", "active_area_power_fraction"), ("capture_margin_mm", "capture_margin_mm"),
    ("pupil_centroid_u_mm", "pupil_centroid_uv_mm"), ("pupil_centroid_v_mm", "pupil_centroid_uv_mm"),
    ("pupil_rms_radius_mm", "pupil_rms_radius_mm"), ("fo_image_rotation_deg", "first_order.image_rotation_deg"),
    ("fo_transverse_magnification", "first_order.transverse_magnification"),
)


def arrival_row(observation_id: str, arrival: ArrivalResult) -> dict[str, Any]:
    beam, pol, coincidence = arrival.beam, arrival.polarization_state, arrival.surface_coincidence
    duvn = arrival.direction_uvn or (None, None, None)
    values: dict[str, Any] = {
        "observation_id": observation_id, "source_id": beam.source_id, "path_id": beam.path_id, "parent_path_id": arrival.parent_path_id,
        "spectral_index": beam.spectral_index, "wavelength_nm": beam.wavelength_nm, "crossing_index": arrival.crossing_index,
        **_vec("position", beam.position_mm, "mm"), **_vec("direction", beam.direction, "1"),
        "u_mm": arrival.uv_mm[0], "v_mm": arrival.uv_mm[1], "direction_u": duvn[0], "direction_v": duvn[1], "direction_n": duvn[2],
        "aoi_deg": arrival.aoi_deg, "power_W": beam.power_W, "detected_power_W": arrival.detected_power_W,
        "coincident_surface_id": None if coincidence is None else coincidence.surface_id,
        "coincidence_role": None if coincidence is None else coincidence.role,
        "coincidence_event": None if coincidence is None else coincidence.surface_event,
        "crossing_power_upper_bound_W": None if coincidence is None else coincidence.crossing_power_upper_bound_W,
        "geometric_length_mm": beam.geometric_length_mm,
        "optical_path_length_mm": beam.optical_path_length_mm, "phase_rad": beam.phase_rad,
        "jones_ex_re": None if beam.polarization is None else beam.polarization.ex[0], "jones_ex_im": None if beam.polarization is None else beam.polarization.ex[1],
        "jones_ey_re": None if beam.polarization is None else beam.polarization.ey[0], "jones_ey_im": None if beam.polarization is None else beam.polarization.ey[1],
        **_vec("basis_e1", beam.basis.e1, "1"), **_vec("basis_e2", beam.basis.e2, "1"),
        "stokes_s1": pol.s1 if pol else None, "stokes_s2": pol.s2 if pol else None, "stokes_s3": pol.s3 if pol else None,
        "polarization_ellipticity_deg": pol.ellipticity_deg if pol else None, "polarization_handedness": pol.handedness if pol else None,
        "polarization_azimuth_deg": pol.azimuth_in_plane_deg if pol else None,
        **_gaussian(arrival),
        "within_acceptance": arrival.within_acceptance, "acceptance_margin_deg": arrival.acceptance_margin_deg,
        "active_area_power_fraction": arrival.active_area_power_fraction, "capture_margin_mm": arrival.capture_margin_mm,
        "pupil_ray_count": arrival.pupil_ray_count, "pupil_rays_arrived": arrival.pupil_rays_arrived,
        "pupil_centroid_u_mm": arrival.pupil_centroid_uv_mm[0] if arrival.pupil_centroid_uv_mm else None,
        "pupil_centroid_v_mm": arrival.pupil_centroid_uv_mm[1] if arrival.pupil_centroid_uv_mm else None,
        "pupil_rms_radius_mm": arrival.pupil_rms_radius_mm, "ray_opl_defined": arrival.ray_opl_defined,
        "coherence_group": arrival.coherence_group, "coherence_length_mm": arrival.coherence_length_mm,
        **_first_order(arrival), "lineage": list(arrival.lineage), "power_out_of_scope_at": list(arrival.power_out_of_scope_at),
        "value_basis": row_basis_cell([("power_W", arrival, "beam.power_W"), ("detected_power_W", arrival, "detected_power_W"),
                                       *((column, arrival, field) for column, field in ARRIVAL_BASIS_COLUMNS
                                         if field in (arrival.value_basis or {})),
                                       *((name, arrival.beam.gaussian, name) for name in GAUSSIAN_VALUE_COLUMNS
                                         if arrival.beam.gaussian is not None)]),
    }
    return _row("arrivals", values)


def path_rows(result: ValidationResult, crossings: dict[str, list[str]]) -> list[dict[str, Any]]:
    """One row per beam path; ``crossings`` maps a path id to the observation ids it crosses, in order."""

    by_id = {p.path_id: p for p in result.paths}
    children: dict[str, list[str]] = {}
    for p in result.paths:
        if p.parent_path_id is not None:
            children.setdefault(p.parent_path_id, []).append(p.path_id)
    rows = []
    for p in result.paths:
        depth, parent = 0, p.parent_path_id
        while parent is not None and parent in by_id:
            depth, parent = depth + 1, by_id[parent].parent_path_id
        rows.append(_row("paths", {
            "path_id": p.path_id, "source_id": p.source_id, "spectral_index": p.spectral_index, "wavelength_nm": p.wavelength_nm,
            "parent_path_id": p.parent_path_id, "branch_surface_id": p.branch_surface_id, "branch_kind": p.branch_kind,
            "children": sorted(children.get(p.path_id, [])), "depth": depth, "segment_count": len(p.segments),
            "power_start_W": p.segments[0].power_start_W if p.segments else None, "final_power_W": p.final_power_W,
            "power_out_of_scope_at": list(p.power_out_of_scope_at),
            "termination": p.termination, "termination_surface_id": p.termination_surface_id,
            "geometric_path_length_mm": p.geometric_path_length_mm, "optical_path_length_mm": p.optical_path_length_mm,
            "observations": crossings.get(p.path_id, []), "lineage": list(p.lineage),
        }))
    return rows


def segment_rows(result: ValidationResult, path_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for p in result.paths:
        if path_id is not None and p.path_id != path_id:
            continue
        for index, s in enumerate(p.segments):
            rows.append(_row("segments", {
                "path_id": p.path_id, "segment_index": index, **_vec("start", s.start_mm, "mm"), **_vec("end", s.end_mm, "mm"),
                "length_mm": s.length_mm, "medium_id": s.medium_id, "refractive_index": s.refractive_index,
                "optical_path_length_mm": s.optical_path_length_mm, "power_start_W": s.power_start_W, "power_end_W": s.power_end_W,
                "end_surface_id": s.end_surface_id, "end_event": s.end_event, "angle_of_incidence_deg": s.angle_of_incidence_deg,
                "beam_radius_start_mm": s.beam_radius_start_mm, "beam_radius_end_mm": s.beam_radius_end_mm,
            }))
    return rows


def _fraction(power: float, total: float) -> float | None:
    return power / total if total > 0.0 else None


def power_budget_rows(result: ValidationResult) -> list[dict[str, Any]]:
    budget = result.power_budget
    if budget is None:
        return []
    rows = []

    def add(line: Any, category: str, name: str | None, power: float | None, total: float, conserved: bool | None) -> None:
        rows.append(_row("power-budget", {
            "source_id": line.source_id if line else None, "spectral_index": line.spectral_index if line else None,
            "wavelength_nm": line.wavelength_nm if line else None, "category": category, "name": name, "power_W": power,
            "fraction_of_input": None if power is None else _fraction(power, total), "conserved": conserved,
        }))

    for line in budget.lines:
        total = line.input_power_W
        add(line, "INPUT", None, total, total, line.conserved)
        # (by name: the rows of a result held in memory are those of the same result read from its run directory, whose
        # JSON holds the keys sorted)
        for kind, power in sorted(line.sinks.items()):
            add(line, "SINK", kind, power, total, None)
        for observation_id, power in sorted(line.observed_W.items()):
            add(line, "OBSERVED", observation_id, power, total, None)
        for observation_id in line.observed_out_of_scope_at:  # (no power stated: chief-ray powers, not the light's)
            add(line, "OBSERVED", observation_id, None, total, None)
        add(line, "RESIDUAL", None, line.residual_W, total, line.conserved)
    add(None, "INPUT", "TOTAL", budget.total_input_W, budget.total_input_W, budget.conserved)
    for kind, power in sorted(budget.total_sinks.items()):
        add(None, "SINK", kind, power, budget.total_input_W, None)
    add(None, "RESIDUAL", "TOTAL", budget.residual_W, budget.total_input_W, budget.conserved)
    return rows


def sampled_light_rows(observations: Iterable[ObservationResult], observation_id: str | None = None) -> list[dict[str, Any]]:
    """One row per observation plane and sampled source line (``ObservationResult.sampled_light``) of ``observations`` (the
    planes as the results report them, ``RunResults._scoped_observations``), with the ``value_basis`` of its power and
    collection fractions."""

    return [_row("sampled-light", {"observation_id": o.observation_id,
                                   **{name: list(value) if isinstance(value := getattr(line, name), tuple) else value
                                      for name, _u, _t in SAMPLED_LIGHT_COLUMNS[1:-1]},
                                   "value_basis": row_basis_cell([(name, line, name) for name in SAMPLED_LINE_FIELDS])})
            for o in observations if observation_id in (None, o.observation_id) for line in o.sampled_light]


def sampled_crossing_rows(observations: Iterable[ObservationResult], observation_id: str | None = None) -> list[dict[str, Any]]:
    """One row per crossing of a plane by a sampled ray (``SampledCrossings``) of ``observations`` (as ``sampled_light_rows``),
    in the order the rays were traced; a crossing's power carries its line's ``power_W`` entry."""

    rows = []
    for o in observations:
        if observation_id not in (None, o.observation_id):
            continue
        for line in o.sampled_light:
            c = line.crossings
            basis = row_basis_cell([("power_W", line, "power_W")])
            rows.extend(_row("sampled-crossings", {
                "observation_id": o.observation_id, "source_id": line.source_id, "spectral_index": line.spectral_index,
                "wavelength_nm": line.wavelength_nm, "sample_index": c.sample_index[k], "u_mm": c.u_mm[k], "v_mm": c.v_mm[k],
                "direction_u": c.direction_u[k], "direction_v": c.direction_v[k], "direction_n": c.direction_n[k],
                "power_W": c.power_W[k], "counted": c.counted[k], "value_basis": basis}) for k in range(len(c.power_W)))
    return rows


def superposition_rows(observations: Iterable[ObservationResult], observation_id: str | None = None) -> list[dict[str, Any]]:
    """The coherent sums of ``observations`` (the planes as the results report them, ``RunResults._scoped_observations``)."""

    rows = []
    for observation in observations:
        if observation_id is not None and observation.observation_id != observation_id:
            continue
        for s in observation.superpositions:
            rows.append(_row("superpositions", {
                "observation_id": observation.observation_id, "wavelength_nm": s.wavelength_nm, "path_ids": list(s.path_ids),
                "crossing_indices": list(s.crossing_indices), "incoherent_power_W": s.incoherent_power_W,
                "total_power_W": s.total_power_W, "interference_power_W": s.interference_power_W,
                "total_power_min_W": s.total_power_min_W, "total_power_max_W": s.total_power_max_W, "phase_statement": s.phase_statement,
                "status": s.status.value,
                "issue_codes": list(s.issue_codes),
                "value_basis": row_basis_cell([(name, s, name) for name in ("incoherent_power_W", "total_power_W",
                                                                          "interference_power_W", "total_power_min_W",
                                                                          "total_power_max_W")]),
            }))
    return rows


def grating_order_rows(result: ValidationResult, element_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for order in result.grating_orders:
        if element_id is not None and order.element_id != element_id:
            continue
        direction = order.direction or (None, None, None)
        rows.append(_row("grating-orders", {**dict.fromkeys(name for name, _unit, _text in GRATING_ORDER_COLUMNS), **order.model_dump(mode="json", exclude={"direction"}),
                                            "direction_x": direction[0], "direction_y": direction[1], "direction_z": direction[2],
                                            "power_out_of_scope_at": list(order.power_out_of_scope_at)}))
    return rows


def nonlinear_rows(result: ValidationResult, element_id: str | None = None) -> list[dict[str, Any]]:
    """One row per beam's pass into a crystal that generates its second harmonic (``NonlinearConversionResult``)."""

    return [_row("nonlinear", {**dict.fromkeys(name for name, _unit, _text in NONLINEAR_COLUMNS), **row.model_dump(mode="json")})
            for row in result.nonlinear_conversions if element_id is None or row.element_id == element_id]


def exposure_rows(result: ValidationResult, element_id: str | None = None) -> list[dict[str, Any]]:
    report = result.exposures
    rows: list[dict[str, Any]] = []
    for surface in report.surfaces if report is not None else ():
        if element_id is not None and surface.element_id != element_id:
            continue
        for light, hit in ([("traced", h) for h in surface.hits] + [("untraced", h) for h in surface.untraced_hits]
                           + [("beside", h) for h in surface.beside_hits]):
            footprint = hit.footprint_semi_axes_mm or (None, None)
            rows.append(_row("exposures", {
                **hit.model_dump(mode="json", exclude={"footprint_semi_axes_mm"}), "light": light, "surface_id": surface.surface_id,
                "element_id": surface.element_id, "footprint_major_mm": footprint[0], "footprint_minor_mm": footprint[1],
                "surface_incident_power_W": surface.incident_power_W,
                "surface_peak_irradiance_W_per_cm2": surface.peak_irradiance_W_per_cm2, "surface_complete": surface.complete,
                "rating_status": surface.rating_status,
                "value_basis": row_basis_cell([(name, hit, name) for name in EXPOSURE_VALUE_COLUMNS]),
            }))
    return rows


def power_rating_rows(result: ValidationResult, element_id: str | None = None) -> list[dict[str, Any]]:
    report = result.exposures
    rows: list[dict[str, Any]] = []
    for surface in report.surfaces if report is not None else ():
        if element_id is not None and surface.element_id != element_id:
            continue
        for c in surface.ratings:
            rows.append(_row("power-ratings", {
                **c.model_dump(mode="json"), "exposure_upper_bound": c.exposure_upper_bound,
                # (absent from the dump when the rated value establishes no threshold: an empty cell, never a number)
                "rated_threshold_low": c.rated_threshold_low, "rated_threshold_high": c.rated_threshold_high,
                "surface_id": surface.surface_id,
                "element_id": surface.element_id,
                "rating_fact_status": c.rating_status, "mismatches": list(c.mismatches), "notes": list(c.notes),
                "unverified_conditions": list(c.unverified_conditions), "conditions": [_condition_text(x) for x in c.conditions],
                "rated_conditions": [t for t in (_rated_text(x) for x in c.conditions) if t],
                "design_values": [t for t in (_design_text(x) for x in c.conditions if x.state != "NOT_STATED") if t],
                "value_basis": row_basis_cell([(name, c, name) for name in ("exposure_value", "exposure_lower_bound",
                                                                          "exposure_upper_bound", "utilization")]),
            }))
    return rows


def _value(value: Any, unit: str) -> str:
    text = f"{value:.6g}" if isinstance(value, float) else str(value)
    if unit == "pulses" and value == 1:  # (one pulse)
        unit = "pulse"
    return f"{text} {unit}".strip() if unit and not isinstance(value, str) else text


def _rated_text(condition: Any) -> str:
    """``condition=value unit`` of a condition the rating states (``beam_diameter`` with its 1/e^2 readings); empty for one
    it does not state."""
    if condition.state == "NOT_STATED" or condition.rated is None:
        return ""
    text = f"{condition.condition}={_value(condition.rated, condition.unit)}"
    if condition.rated_low is not None and (condition.rated_low, condition.rated_high) != (condition.rated, condition.rated):
        text += f" (1/e^2 {_value(condition.rated_low, '')}..{_value(condition.rated_high, condition.unit)})"
    return text


def _design_text(condition: Any) -> str:
    """``condition=value unit`` of the light's value for a condition (a range as ``low..high``); empty when it has none."""
    if condition.design is not None:
        return f"{condition.condition}={_value(condition.design, condition.unit)}"
    if condition.design_low is None:
        return ""
    if condition.design_high == condition.design_low:
        return f"{condition.condition}={_value(condition.design_low, condition.unit)}"
    high = "" if condition.design_high is None else _value(condition.design_high, condition.unit)
    return f"{condition.condition}={_value(condition.design_low, '')}..{high or 'unbounded'}"


#: Short forms of a condition's effect in the ``power-ratings`` export.
_EFFECTS = {"THRESHOLD_AT_LEAST_RATED": "AT_LEAST", "THRESHOLD_AT_MOST_RATED": "AT_MOST", "NO_ESTABLISHED_RELATIONSHIP": "NONE"}


def _condition_text(condition: Any) -> str:
    """``condition=STATE`` of a rating condition, with its effect or reason (``beam_diameter=DIFFERS(AT_MOST)``)."""

    detail = _EFFECTS.get(condition.effect or "", condition.reason or "")
    return f"{condition.condition}={condition.state}" + (f"({detail})" if detail else "")


def metric_rows(descriptors: Any) -> list[dict[str, Any]]:
    return [_row("metrics", {**d.model_dump(mode="json"), "selectors": list(d.selectors), "result_fields": list(d.result_fields)})
            for d in descriptors]


def handoff_row(observation_id: str, beam: dict[str, Any]) -> dict[str, Any]:
    """One beam of a handoff document (``results.handoff``) as a flat row."""

    g = beam["gaussian"] or {}
    q_x, q_y = g.get("q_x_mm") or (None, None), g.get("q_y_mm") or (None, None)
    pol, ray, validity = beam["polarization"], beam["chief_ray"], beam["validity"]

    def uvn(prefix: str, vector: Any) -> dict[str, Any]:
        values = (None, None, None) if vector is None else vector
        return {f"{prefix}_{axis}": value for axis, value in zip(UVN, values)}

    values: dict[str, Any] = {
        "observation_id": observation_id, "path_id": beam["path_id"], "source_id": beam["source_id"],
        "spectral_index": beam["spectral_index"], "wavelength_nm": beam["wavelength_nm"], "crossing_index": beam["crossing_index"],
        "medium_index": beam["medium_index"], "power_W": beam["power_W"], "u_mm": ray["uv_mm"][0], "v_mm": ray["uv_mm"][1],
        **uvn("direction", ray["direction_uvn"]), "aoi_deg": ray["aoi_deg"],
        "q_x_re_mm": q_x[0], "q_x_im_mm": q_x[1], "q_y_re_mm": q_y[0], "q_y_im_mm": q_y[1],
        **{name: g.get(name) for name in ("beam_radius_x_mm", "beam_radius_y_mm", "waist_radius_x_mm", "waist_radius_y_mm",
                                          "curvature_radius_x_mm", "curvature_radius_y_mm", "m2_x", "m2_y")},
        **uvn("gaussian_axis_x", g.get("axis_x_uvn")), **uvn("gaussian_axis_y", g.get("axis_y_uvn")),
        **(dict.fromkeys(("jones_1_re", "jones_1_im", "jones_2_re", "jones_2_im")) if pol["jones"] is None else {
            "jones_1_re": pol["jones"][0][0], "jones_1_im": pol["jones"][0][1], "jones_2_re": pol["jones"][1][0], "jones_2_im": pol["jones"][1][1]}),
        **uvn("basis_e1", pol["basis_e1_uvn"]), **uvn("basis_e2", pol["basis_e2_uvn"]),
        "phase_rad": beam["phase_rad"], "optical_path_length_mm": beam["optical_path_length_mm"],
        "phase_defined": validity["phase_defined"], "beam_parameters": validity["beam_parameters"], "tem00_field_applicable": validity["tem00_field_applicable"],
        "truncated_at": sorted({t["surface_id"] for t in validity["truncated_upstream"] if t["surface_id"]}),
        "truncated_power_fraction": validity.get("truncated_power_fraction"), "field_intensity_ripple": validity.get("field_intensity_ripple"),
        "power_out_of_scope_at": list(validity.get("power_out_of_scope_at", [])),
        "power_taken_before_at": list(validity.get("power_taken_before_at", [])),
        "model_scope": sorted({f"{f['code']}@{f['instance_id']}" for f in validity.get("model_scope", [])}),
        "value_basis": {**(beam.get("value_basis") or {}), **{f"gaussian.{key}": entry for key, entry in (g.get("value_basis") or {}).items()}}
        or None,
    }
    return _row("handoff", values)
