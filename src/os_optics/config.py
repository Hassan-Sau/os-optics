"""Centralized numerical settings.

Every epsilon / tolerance / threshold / step / iteration limit used by the engine
lives here, with the reason for its default. Physics code receives a
``NumericalConfig`` explicitly; there are no module-level numeric tolerances.
Changing a default (or adding a setting) after a published release bumps
``NUMERICAL_CONFIG_VERSION``, once per release (``_version.py``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ._version import NUMERICAL_CONFIG_VERSION

# The effective budgets are serialized through both Python and JavaScript. Keep integer identity exact.
MAX_EXACT_JSON_INTEGER = 2**53 - 1


class NumericalConfig(BaseModel):
    """All numerical tolerances of the engine (frozen, hashed into the manifest)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    config_version: str = Field(NUMERICAL_CONFIG_VERSION, description="Version of the defaults set.")

    # ---- ray geometry -----------------------------------------------------------------------
    ray_min_advance_mm: float = Field(
        1e-7, gt=0, description="Minimum ray parameter t for a valid hit; prevents re-intersecting the surface just left."
    )
    parallel_dot_tol: float = Field(
        1e-12, gt=0, description="|n.d| below this is 'parallel': no intersection is reported."
    )
    laboratory_reach_mm: float = Field(
        1000.0, ge=1000.0, le=1000.0,
        description="A declared numerical choice, the reach of a laboratory scene: on a leg that leaves the scene, a plane without "
        "an active area counts part of a beam up to max(this, the scene's support along the ray) from the leg's start. It is "
        "recorded with the run and fixed at this value: no other value is accepted."
    )
    grazing_warn_aoi_deg: float = Field(85.0, gt=0, lt=90, description="AOI above which GRAZING_INCIDENCE warns.")
    grazing_fail_aoi_deg: float = Field(89.9, gt=0, lt=90, description="AOI above which the interaction is rejected.")
    antiparallel_tol: float = Field(
        1e-12, gt=0, description="1+cos(angle) below which two directions are treated as antiparallel (basis transport)."
    )
    unit_vector_tol: float = Field(1e-9, gt=0, description="Deviation |1-|v|| up to which a direction's components are used unchanged; beyond it the direction is normalized.")
    zero_vector_tol: float = Field(1e-12, gt=0, description="Vectors with norm below this are zero-length.")
    quaternion_norm_tol: float = Field(
        1e-6, gt=0,
        description="Deviation |1-|q|| beyond which a stated quaternion is refused: the shared Project contract rejects it and the Engine input builder refuses it (POSE_INVALID). The Engine itself takes any non-zero, finite quaternion as the rotation of q / |q|, whatever its length; a zero one is refused by the request contract (INVALID_INPUT).",
    )
    aperture_edge_tol_mm: float = Field(
        1e-9, ge=0, description="Inclusive aperture boundary tolerance (floating-point edge)."
    )
    tir_warn_margin: float = Field(
        1e-3, gt=0, description="Relative distance to the critical angle that triggers TIR_BOUNDARY."
    )
    power_epsilon_W: float = Field(1e-15, ge=0, description="Power at or below this terminates a path (ZERO_POWER); 0: no absolute floor.")
    energy_conservation_tol: float = Field(1e-9, ge=0, description="Tolerance for R+T<=1 and unit-norm Jones checks.")
    negligible_power_fraction: float = Field(
        1e-9, ge=0, lt=1, description="Share of a source line's power below which light is not followed, probed or listed: the "
        "level of the light a result leaves out by statement (a plane reads 'nothing left out' relative to it). A caller setting."
    )
    basis_orthonormality_tol: float = Field(
        1e-9, gt=0, description="Allowed deviation of a transported transverse basis before a warning."
    )
    max_interactions_per_path: int = Field(
        64, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER, description="Guard against endless cavity loops: physical surface interactions per path (observation planes are "
        "virtual and do not count). A path that would exceed it stops (MAX_INTERACTIONS, UNKNOWN); its remaining power is the ledger's "
        "INTERACTION_LIMIT, so a result that light could still change is UNKNOWN."
    )
    max_paths: int = Field(
        512, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER, description="Trace budget: beam paths per run (source spectral lines and every branch a splitting surface creates), "
        "the strongest first. Beyond it the weakest source lines and branches are not traced; one MAX_PATHS issue gives their count and "
        "power, and a target no traced beam reaches is UNKNOWN (MAX_PATHS), not 0 W. Each traced path also carries its pupil rays into the ray "
        "data (about 70 kB of run files per trace step)."
    )
    max_trace_steps: int = Field(
        2048, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER, description="Trace budget: chief-ray steps over all paths of a run (a path takes one step to each surface it reaches "
        "and one to leave the scene; a source line through N surfaces takes N + 1: 512 lines through one lens to a monitor plane). Once "
        "spent, the path being traced stops there and the weaker queued branches are not traced (one MAX_PATHS issue); with max_paths "
        "it bounds a run's time, memory and run files (about 150 MB) in a laptop process or a browser tab."
    )
    retarder_warn_aoi_deg: float = Field(
        20.0, gt=0, description="Retarder/polarizer AOI above which RETARDER_OBLIQUE_INCIDENCE warns."
    )
    birefringent_interface_warn_aoi_deg: float = Field(
        2.0, gt=0, lt=90, description="Angle of incidence (deg, the incident wave normal on a crystal face) above which the face's "
        "power split between the crystal's ordinary and extraordinary waves warns BIREFRINGENT_INTERFACE_APPROXIMATE: it uses the "
        "isotropic Fresnel (or coating) transmittance at each wave's index, exact at normal incidence for any optic-axis direction."
    )
    birefringent_gaussian_rel_tol: float = Field(
        1e-2, gt=0, description="A Gaussian beam crossing a uniaxial crystal as an extraordinary wave is carried with one diffraction "
        "index while its two axes diffract with two: once the accumulated difference of its reduced q-parameter (the uncertainty of its "
        "waist position, mm of reduced path) exceeds this fraction of its reduced Rayleigh range the Gaussian state is out of model "
        "scope (GAUSSIAN_ANISOTROPIC_MEDIUM)."
    )
    nonlinear_depletion_rel_tol: float = Field(
        2e-2, gt=0, lt=1, description="Second-harmonic generation is computed with an undepleted pump: once the estimated relative "
        "amount by which that overstates the harmonic power (8/9 of the conversion efficiency for one fundamental wave, more for two "
        "unequal ones) exceeds this, the conversion is out of model scope (NONLINEAR_CONVERSION_OUT_OF_SCOPE)."
    )
    nonlinear_focusing_rel_tol: float = Field(
        2e-2, gt=0, lt=1, description="A crystal whose second-harmonic generation takes the beam radius as constant along it (near "
        "field) is out of model scope when that result differs from the one that follows the beam's focusing through the crystal by "
        "more than this fraction of the larger of that focused result and the focused result at phase matching (dk = 0) "
        "(NONLINEAR_CONVERSION_OUT_OF_SCOPE). Where it is within this fraction of the focused result at dk = 0 but not of the "
        "focused result itself (near a zero of the tuning curve, which the focusing fills), the focused result is reported."
    )
    nonlinear_quadrature_rel_tol: float = Field(
        1e-7, gt=0, lt=1e-2, description="Relative agreement of two successive node counts at which the Gaussian-beam overlap integral "
        "of second-harmonic generation stops refining; a result whose last two values still differ by more than "
        "nonlinear_focusing_rel_tol is out of model scope."
    )
    nonlinear_harmonic_gaussian_rel_tol: float = Field(
        2e-2, gt=0, lt=1, description="Walk-off spreads the harmonic generated along a crystal over a strip: once the relative growth "
        "of its rms width along the walk-off, (s L / w0)^2 / 3, exceeds this, the harmonic is traced without a Gaussian state "
        "(HARMONIC_BEAM_NOT_GAUSSIAN); its power is unaffected."
    )
    nonlinear_beam_quality_tol: float = Field(
        5e-2, gt=0, lt=1, description="Second-harmonic generation is computed for a TEM00 fundamental: a beam whose M^2 exceeds 1 by "
        "more than this on either axis is out of model scope (NONLINEAR_CONVERSION_OUT_OF_SCOPE)."
    )
    nonlinear_linewidth_rel_tol: float = Field(
        2e-2, gt=0, lt=1, description="Second-harmonic generation is computed for one frequency: for a line with a stated linewidth "
        "(or coherence length) the phase-matching factor sinc^2(dk L / 2) averaged over its Gaussian spectrum may differ from the "
        "line centre's by at most this fraction of the larger of the two (the centre's value is reported); beyond, the "
        "conversion is out of model scope (NONLINEAR_CONVERSION_OUT_OF_SCOPE)."
    )

    sag_bracket_samples: int = Field(
        65, ge=2, description="Samples of F(t) = z(t) - sag(t) along a ray's crossing of a sag surface's bounding box, used to "
        "bracket the nearest intersection when the surface has even-asphere, axicon or two-variable terms (conic-only surfaces are solved "
        "in closed form); on an even asphere two intersections closer than one sample spacing (a ray grazing a local bump) are not "
        "separated, on a two-variable shape they are, at the closest approach between the samples."
    )
    sag_max_iterations: int = Field(
        64, ge=1, description="Iteration cap of the bracketed Newton / bisection refinement of a sag-surface intersection (bisection "
        "alone halves the bracket each step, so the cap is never reached for brackets below 1e6 mm)."
    )
    sag_intersection_tol_mm: float = Field(
        1e-12, gt=0, description="A sag-surface intersection is converged when the last Newton / bisection step changed the ray "
        "parameter by at most this (mm); a ray whose bracket does not converge continues as missing the surface, and on a two-variable "
        "shape the run reports it (SAG_INTERSECTION_NOT_CONVERGED)."
    )

    sellmeier_pole_rel_tol: float = Field(
        1e-9, gt=0, description="Relative |l^2 - C_i| / l^2 below which a Sellmeier pole is considered hit."
    )
    tabulated_range_tol_nm: float = Field(
        1e-9, ge=0, description="Slack (nm) on the inclusive wavelength validity bounds of materials and coatings."
    )
    tabulated_aoi_tol_deg: float = Field(1e-9, ge=0, description="Slack (deg) on the AOI bounds of tabulated coatings.")

    # ---- Gaussian beam ------------------------------------------------------------------------
    gaussian_warn_divergence_rad: float = Field(0.10, gt=0, description="Half-angle divergence (NA~0.1) warning threshold.")
    gaussian_scope_divergence_rad: float = Field(
        0.30, gt=0, description="Half-angle divergence above which the paraxial Gaussian model is out of scope."
    )
    gaussian_max_w_over_radius: float = Field(
        0.2, gt=0, description="Beam radius / |surface radius of curvature| limit for ABCD validity."
    )
    gaussian_max_aoi_deg: float = Field(60.0, gt=0, lt=90, description="Tilted-surface ABCD validity limit on AOI.")
    gaussian_max_ray_spread_ratio: float = Field(
        2.0, gt=1, description="Pupil-ray RMS spread / the largest spread an aberration-free beam with the arrival's Gaussian envelope "
        "can have (1 for a full pupil sample, at most sqrt(2) for any subset of it), on the observation plane and on the Gaussian waist "
        "plane, above which the beam is aberration dominated (GAUSSIAN_ABERRATION_DOMINATED, OUT_OF_MODEL_SCOPE): the Gaussian radius "
        "there is then about this factor too small or worse."
    )
    gaussian_clip_warn_fraction: float = Field(1e-3, ge=0, description="Truncation power loss that warns.")
    gaussian_clip_scope_fraction: float = Field(
        0.10, gt=0, description="Truncation power loss above which the Gaussian state is invalidated."
    )
    gaussian_astig_align_tol: float = Field(
        1e-9, gt=0, description="|sin(phi)cos(phi)| below which beam axes are aligned with (t,s)."
    )
    gaussian_round_tol: float = Field(
        1e-9, gt=0, description="Relative difference below which the two transverse axes are 'round'."
    )
    gaussian_umbilic_rel_tol: float = Field(
        1e-9, gt=0, description="Relative difference of a sag surface's two principal curvatures at the chief-ray hit below which "
        "the surface is treated as umbilic there (no preferred axes): its principal directions then need not align with (t, s)."
    )
    gaussian_plane_axial_spread_fraction: float = Field(
        0.1, gt=0, description="Axial spread across a tilted plane footprint vs z_R that triggers a warning."
    )
    aperture_quad_angular: int = Field(
        256, ge=8, description="Angular trapezoid nodes for circular-aperture Gaussian truncation (periodic, spectral)."
    )
    aperture_quad_sector_nodes: int = Field(
        48, ge=4, le=2**53 - 1, description="Gauss-Legendre nodes per panel/sector for off-axis circular and rectangular truncation "
        "(building the rule grows with the square of the count; a count whose rule the machine cannot hold is refused by name)."
    )
    aperture_quad_extent_sigmas: float = Field(
        6.0, gt=0, description="Gaussian integrals are truncated at this many beam radii from the beam centre (exp(-72) ~ 0)."
    )

    # ---- pupil sampling -----------------------------------------------------------------------
    pupil_ring_count: int = Field(8, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER, description="Equal-power rings in the default pupil sampling.")
    pupil_points_per_ring: int = Field(12, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER, description="Points on each ring of the default pupil sampling.")

    # ---- first-order (differential) rays -----------------------------------------------------------
    first_order_position_step_mm: float = Field(
        1e-2, gt=0, description="Source-plane displacement of the differential rays (central differences of the first-order ray-transfer "
        "matrix): small against apertures and focal lengths (truncation error ~(step/f)^2 < 1e-7), large against coordinate roundoff."
    )
    first_order_angle_step_rad: float = Field(
        1e-4, gt=0, description="Source-plane tilt of the differential rays (central differences; truncation error ~step^2 = 1e-8)."
    )
    first_order_unitless_resolution: float = Field(
        1e-6, gt=0, description="Reported resolution of the dimensionless A and D blocks of the first-order matrix (entries are rounded to "
        "it: finite-difference roundoff ~1e-12, so the result is identical on every runtime)."
    )
    first_order_b_resolution_mm: float = Field(1e-4, gt=0, description="Reported resolution of the B block (mm per rad).")
    first_order_c_resolution_per_mm: float = Field(1e-8, gt=0, description="Reported resolution of the C block (per mm).")
    power_ledger_fraction_resolution: float = Field(
        1e-12, gt=0, description="Resolution, as a fraction of the beam power, of ledger entries computed as a small difference of two powers "
        "(aperture truncation): rounded so the ledger is identical on every runtime."
    )

    # ---- interference -------------------------------------------------------------------------
    interference_min_crossing_deg: float = Field(
        1e-6, gt=0, description="Crossing angle below which the fringe period is undefined."
    )
    interference_wavelength_tol_nm: float = Field(
        1e-9, gt=0, description="Two arrivals must agree in wavelength within this to interfere."
    )
    interference_min_overlap: float = Field(
        1e-3, gt=0, description="Spatial overlap ratio below which beams are considered not overlapping (visibility < 0.1 %)."
    )
    interference_collinear_period_ratio: float = Field(
        10.0, gt=1, description="Fringe period / beam diameter above which two beams are treated as collinear (single-fringe detector)."
    )
    max_interference_pairs: int = Field(256, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER,
                                        description="Cap on automatically analysed beam pairs per observation.")
    max_return_analysis_work: int = Field(
        4_000_000, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER, description="Work budget of the return-light analysis of the "
        "whole run, in work units: one per cone test of the screen for closed returns of three or more surfaces, 1000 per "
        "pair of reflecting surfaces screened, and the fold-solid analysis' tests (about 4 microseconds per unit on a "
        "desktop CPU). The analysis runs only for a wavelength whose light left out of the trace a power or exposure upper "
        "bound reads; the wavelengths share the budget in the order the run reads them (each record's work_before). An "
        "analysis that reaches the budget stops: the upper bounds that depend on it are withheld (RETURN_ANALYSIS_BUDGET, a "
        "notice; the execution stays COMPLETED and the budget is listed as exhausted in phase return_analysis), never "
        "replaced by a value the analysis did not finish."
    )
    interference_det_tol: float = Field(1e-30, gt=0, description="Determinant floor for the summed envelope matrix.")
    max_scatter_origins: int = Field(
        64, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER, description="Work budget: scattering origins whose scattered light the "
        "run traces as rays, the strongest first; the light of the others stays booked as scattered at its surface "
        "(SCATTERED_LIGHT_NOT_FOLLOWED on the planes it may reach)."
    )
    max_mesh_patches: int = Field(
        1024, strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER, description="Work budget: traced planar patches one faceted surface "
        "may have; a surface with more is refused."
    )

    # ---- power ratings (caller settings: the windows within which a rating's test condition counts as matched) -------
    rating_wavelength_tol_rel: float = Field(
        0.02, ge=0, le=1, description="Relative distance from a rating's test wavelength within which the light's matches it."
    )
    rating_pulse_duration_tol_rel: float = Field(
        0.10, ge=0, le=1, description="Relative distance from a rating's test pulse duration within which the light's matches it."
    )
    rating_repetition_rate_tol_rel: float = Field(
        0.10, ge=0, le=1, description="Relative distance from a rating's test repetition rate within which the light's matches it."
    )
    rating_beam_diameter_tol_rel: float = Field(
        0.05, ge=0, le=1, description="Relative distance from a rating's test beam diameter within which the beam's matches it."
    )
    rating_polarization_fraction_tol: float = Field(
        0.01, ge=0, le=1, description="Largest share of a beam's power outside a rating's stated polarization for it to match."
    )
    rating_unlocated_light_share: float = Field(
        0.01, ge=0, le=1, description="Largest share of the light a surface can receive that light of unknown size, place or "
        "state may carry for a rating condition to be verified."
    )

    # ---- published design points (caller settings: how far a single stated point of model data is used as stated) ---
    design_point_wavelength_rel: float = Field(
        0.005, ge=0, lt=1, description="Relative distance from a single stated design wavelength of a coating or zero-order "
        "retarder within which its published values are used."
    )
    design_point_retarder_wavelength_rel: float = Field(
        0.0005, ge=0, lt=1, description="The same for a multi-order retarder (its retardance moves m + 1 times faster)."
    )
    design_point_angle_metallic_reflector_deg: float = Field(
        5.0, ge=0, le=90, description="Distance (deg) from a metallic reflector's single design angle within which its values are used."
    )
    design_point_angle_dielectric_high_reflector_deg: float = Field(
        3.0, ge=0, le=90, description="Distance (deg) from a dielectric high reflector's single design angle within which its values are used."
    )
    design_point_angle_anti_reflection_deg: float = Field(
        3.0, ge=0, le=90, description="Distance (deg) from an anti-reflection coating's single design angle within which its values are used."
    )
    design_point_angle_beamsplitter_deg: float = Field(
        2.0, ge=0, le=90, description="Distance (deg) from a beamsplitter coating's single design angle within which its values are used."
    )
    design_point_interference_shift_fraction: float = Field(
        0.25, gt=0, le=1, description="Share of the narrowest spectral width an interference filter's bands leave that the "
        "angle shift of its spectrum may reach within its single design angle's tolerance."
    )
    design_point_interference_angle_max_deg: float = Field(
        3.0, ge=0, le=90, description="Largest angle tolerance (deg) an interference filter's single design angle gets."
    )

    # ---- control ------------------------------------------------------------------------------
    fd_step_translation_mm: float = Field(1e-3, gt=0, description="Central-difference step for translation DOFs.")
    fd_step_rotation_rad: float = Field(1e-5, gt=0, description="Central-difference step for rotation DOFs.")
    lm_initial_damping: float = Field(1e-3, gt=0, description="Initial Levenberg-Marquardt damping.")
    lm_damping_up: float = Field(10.0, gt=1, description="Damping multiplier after a rejected step.")
    lm_damping_down: float = Field(0.1, gt=0, lt=1, description="Damping multiplier after an accepted step.")
    lm_max_damping: float = Field(1e12, gt=0, description="Damping above which LM gives up.")
    lm_max_iterations: int = Field(100, ge=1, description="Iteration cap for LM.")
    lm_cost_tol: float = Field(1e-18, ge=0, description="Converged when the cost falls below this.")
    lm_step_tol: float = Field(1e-12, ge=0, description="Converged when the scaled step norm falls below this.")
    lm_residual_tol: float = Field(
        1e-6, gt=0, description="Default weighted-residual norm regarded as converged when the spec gives none (sensor units, e.g. mm)."
    )
    lm_scaling_floor: float = Field(
        1e-12, gt=0, description="Floor on diag(J^T J) in Marquardt scaling so unobserved DOFs stay bounded."
    )
    lm_gain_accept: float = Field(0.0, ge=0, description="Minimum actual/predicted reduction ratio to accept a step.")
    jacobian_rcond: float = Field(1e-10, gt=0, description="Singular values below rcond*sigma_max are rank-deficient.")
    observability_participation_min: float = Field(
        0.25, gt=0, le=1, description="An actuator is reported unobservable when at least this fraction of its unit motion "
        "lies in the null space of the sensor Jacobian (0.5 would sit exactly on the symmetric two-DOF case)."
    )
    jacobian_condition_warn: float = Field(1e6, gt=1, description="Jacobian condition number that warns.")
    capture_min_power_fraction: float = Field(
        1e-3, gt=0, description="Detected/available power below this means the beam is lost."
    )
    capture_scan_max_points: int = Field(4096, ge=1, description="Maximum points of the coarse-capture raster scan.")

    # ---- layout -------------------------------------------------------------------------------
    layout_geometry_tol_mm: float = Field(1e-6, ge=0, description="Overlaps/clearance deficits below this are ignored.")
    layout_arc_resolution: int = Field(32, ge=4, description="Shapely buffer segments per quarter circle.")
    beam_envelope_k: float = Field(2.0, gt=0, description="Beam envelope radius = k*w (k=2 -> 1-exp(-8)=99.97% power).")
    moving_envelope_rotation_samples: int = Field(9, ge=2, description="Angles sampled for rotational sweeps.")
    layout_height_tol_mm: float = Field(1e-6, ge=0, description="Height intervals overlapping by less than this are disjoint.")

    # ---- hashing / cross-runtime --------------------------------------------------------------
    hash_float_sig_digits: int = Field(9, ge=3, le=15, description="Significant digits kept when hashing results.")
    hash_float_abs_floor: float = Field(1e-12, ge=0, description="Result values below this magnitude hash as 0.")
    cross_runtime_rel_tol: float = Field(1e-9, gt=0, description="Relative tolerance for cross-runtime comparison.")
    cross_runtime_abs_tol: float = Field(1e-12, gt=0, description="Absolute tolerance for cross-runtime comparison.")


DEFAULT_NUMERICAL_CONFIG = NumericalConfig()
