# os-optics — Numerical Conventions

## 1. Units (fixed by schema, not by field-level choice)

| Quantity | Unit | Notes |
|---|---|---|
| length | mm | every `*_mm` field; radius vs diameter is explicit in the field name (`*_radius_mm`) — never both |
| wavelength | nm (vacuum) | `wavelength_nm`; fringe periods in nm |
| angle (API) | degree | `*_deg`; internal computation in radian |
| power | W | `power_W` |
| irradiance | W/mm² | conversion to mW/cm² only in exposure metrics |
| time | s | |
| dose | mJ/cm² | |
| phase | rad | |
| Gaussian radius | mm | `w` is the **1/e² intensity radius** (not FWHM, not diameter) |
| divergence | rad | half-angle, `θ = M²λ/(π n w₀)` |

## 2. Coordinates

* Global frame: right-handed, `x` right, `y` forward, `z` up (board normal).
  The baseplate top plane is `z = 0` unless the project says otherwise; the
  optical axis height is an absolute `z` of the ray.
* Element local frame: right-handed, origin at the element reference point.
  For transmissive elements local `+z` is the nominal optical axis (light
  travels along `+z`); surface normals are given explicitly per surface, so no
  hidden convention is required.
* `Pose3D = (position_mm, quaternion (w,x,y,z))`; *active* transform
  `x_g = R(q) x_l + p`. Composition `T_parent ∘ T_child`: `R = R_p R_c`,
  `p = R_p p_c + p_p`. Quaternions are Hamilton and taken by their direction:
  any non-zero, finite quaternion is normalized on lowering, whatever its length
  (one whose squared components overflow or underflow is divided by its largest
  component first); a zero one is refused by the request contract
  (`INVALID_INPUT`). `quaternion_norm_tol` is the deviation beyond which the
  Project contract rejects a stated quaternion and the input builder refuses it
  (`POSE_INVALID`).
  Stated directions (normals, axes) and a source's Jones vector are taken the
  same way (`unit_vector_tol`, `energy_conservation_tol`); only a zero-length one
  is refused (`ZERO_LENGTH_VECTOR`, `INVALID_INPUT`).
* `Pose3D.from_axis_direction(position, axis, up)` (the `compute_pose` helper):
  local `z := axis`, local `y := up` orthogonalized against `axis`, local
  `x := y × z`.
* Angle sign: positive rotation is right-handed about the stated axis.
* Surface normal `n̂`: "positive side" is the side `n̂` points toward.
  `medium_positive` / `medium_negative` name the media on the two sides. A ray
  with `d̂·n̂ < 0` arrives from the positive side.
* Mirror: `reflective_side ∈ {positive, negative, both}`. Sphere normal is the
  outward radial vector.
* Signed radius for ABCD: `R>0` if the centre of curvature lies ahead of the
  surface along the propagation direction (refraction) / on the incident
  side (mirror, concave).
* Transverse basis of a ray: `(e₁, e₂)`, `e₁ × e₂ = d̂`.
* Polarization phase convention: `E ∝ exp[i(k·r − ωt)]`; a retarder's slow axis
  gets `+δ/2`. Ideal metal mirror: `(r_s, r_p) = (−1, +1)` in the `(ŝ, d̂×ŝ)` triad, equivalently
  `E_out = −(I − 2 n̂ n̂ᵀ) E_in` (tested against exact vector reflection through non-coplanar folds). Beam splitter: the same
  reflection convention, `(r_s, r_p) = (i√R_s, −i√R_p)` in that triad and `t = √T` (PHYSICS_SCOPE P3), so a non-polarizing
  splitter's reflection (`R_s = R_p = R`) is the mirror's vector map times `−i√R` for every polarization, and a polarizing
  one scales that map's s and p components by `−i√R_s`, `−i√R_p`. At normal incidence, `|d̂×n̂| ≤ 10⁻⁶`
  (`math.vectors.NORMAL_INCIDENCE_SIN`: below it the direction of the computed `d̂×n̂` is rounding), `ŝ` is the ray's
  internal `e₁`, carried from the source's frame (a design turned rigidly gives the same run); a polarizing coating there
  depends on it (not flagged by a run; refused by the network export).
* Stated coating phases (`InterfacePhases`, from a coating's `CoatingPhaseV1`): `r = √R e^{iφ_r}`, `t = √T e^{iφ_t}` per
  polarization in the same `(ŝ, d̂×ŝ)` triad and the same `exp(−iωt)` time dependence; they replace the defaults above
  (mirror `(π, 0)`, splitter `(π/2, r_p = −r_s; 0)`, bare Fresnel phases of a coated refracting face). A tabulated phase is
  linear in vacuum wavelength between its nodes.
* Coating tables are read at the AOI in the incident medium (`θ` of the arriving ray in the medium it travels in),
  bilinear in (λ, AOI); an AOI-independent table is linear in λ alone.
* Source polarization: right-handed (`χ > 0` of `EllipticalPolarizationV1`) means clockwise for an observer looking
  toward the source; `(1, −i)/√2` is right-hand circular.
* Mount margin: a surface with a clear aperture may declare `mount_margin_mm` — the width of the
  opaque ring around the aperture. A ray hitting the ring is blocked (`CLIPPED`); a ray beyond it
  (or beyond the aperture when the margin is 0, e.g. a mirror edge) *misses* the element and passes by.
  An infinite plane never blocks anything by itself.
* Observation planes are virtual and unbounded. Without an active `aperture` every crossing is an
  arrival. With one, a beam counts as an arrival only if its chief ray is inside the area or at least
  `capture_min_power_fraction` of its Gaussian power falls inside it (behind a central obstruction: if
  the whole annulus provably does, or all but `capture_min_power_fraction` of it with a WARNING;
  part of it is recorded with a power that is not computed, `ANNULAR_BEAM_NOT_FOLLOWED`); otherwise the crossing is listed in `outside_active_area` and a
  plane no beam counts on reports `OBSERVATION_NOT_REACHED` with `nearest_miss_mm`, the nearest of the listed crossings.
  A chief ray does not cross a plane with an active area beyond the scene's reach along it (`max(laboratory_reach_mm,
  support)`, `support` the farthest along the ray of the design's physical surfaces, source points and active areas): such a
  crossing is off the area by construction, so it is neither an arrival nor listed, and the leg's partial crossing counts
  any of the beam's light on the area. A plane missed only by such crossings reports `OBSERVATION_NOT_REACHED` without
  `nearest_miss_mm`. Crossing an
  observation plane is not a physical interaction: it never counts toward `max_interactions_per_path`.
  Planes may share a point or a place: the planes the chief ray crosses within `ray_min_advance_mm` of each other form one
  crossing (transparent before absorbing, then by id), and each records every live ray that reaches it, where that ray
  crosses it.
* Known zero: a plane no traced beam reaches receives 0 W when every launched watt was traced to a known
  end and none of the light the trace does not follow (an untraced reflection or coating transmission,
  light passing an aperture or an element its chief ray misses, a beam's tail on an active area it is
  not counted on, part of a beam crossing the plane where its chief ray does not) can reach it: each such
  branch more than `negligible_power_fraction` times its line's
  input is followed through the optics by a probe trace, its own ghosts recursively (PHYSICS_SCOPE P10);
  otherwise the plane is
  `UNKNOWN` and its power `null`. Below-floor power (`BELOW_POWER_FLOOR`)
  counts as zero when it is at most `negligible_power_fraction` times the input of the lines in question
  (their negligible level, e.g. the ~1e-36 W a crossed polarizer leaves); more than that (a
  source at or below `power_epsilon_W`) makes the plane's power not known (`ZERO_POWER`), never 0 W.
* Envelope frame: `EngineeringEnvelopeSnapshot.frame_pose` places the envelope frame relative to the
  component frame. The footprint / z range / keepout / tool / cable regions live in that frame and
  must be plan-parallel. For elements built with local `z` = beam axis use
  `Pose3D.plan_frame_for_optical_axis()` (envelope x = beam axis, y = transverse horizontal, z = up).
* Two-beam phase difference: `phase_difference_rad = wrap[k₀ (OPL_a − OPL_b) − arg⟨E_a, E_b⟩]`
  with `⟨E_a, E_b⟩ = conj(E_a)·E_b` (3D fields, interface phases included, spatial fringe term
  excluded). Where a pass one beam makes and the other does not has no stated phase at the wavelength (an element
  without a thickness, a coating without a stated phase: `group_delay_unknown_at` entries not ending in `#phase` or
  `#phase<=…fs`), the value at the default interface phases with `value_basis` (`ESTIMATE`,
  `UNSTATED_PHASE_AS_INTERFACE`, the passes); None (`PHASE_UNDEFINED`) where such a pass also states no optical path.
* Phase resolution: the wrapped phases `phase_rad` (`k₀·OPL + arg(Eₓ)`, or `+ arg(E_y)` where `|E_y|²` exceeds `|Eₓ|²` by more
  than `DOMINANT_COMPONENT_REL_TOL` = 1e-9 relative: a beam at 45° keeps `Eₓ` whatever the rounding of its components) and `phase_difference_rad`
  are reported rounded to `PHASE_RESOLUTION_RAD` = 1e-6 rad (`os_optics.math.phase`). Wrapping keeps the absolute,
  not the relative, precision of `k₀·OPL`: a 200 mm path at 633 nm is `k₀·OPL ≈ 2·10⁶ rad`, so one unit in the last place
  of the optical path length (3·10⁻¹⁴ mm) is 3·10⁻¹⁰ rad of phase, and the rounding of a traced path (each intersection,
  each `OPL += n·t`) is a few of those; a 10 m path at 200 nm reaches about 10⁻⁷ rad. Digits below 1e-6 rad
  (λ/6·10⁶ of optical path) are not determined by the path and are not reported. Quantities computed from the phase
  (a collinear pair's combined power `P_a + P_b + 2√(P_a P_b)·… cos Δφ`, coherent superposition) use the unrounded value.

## 3. Tolerances (all in `NumericalConfig`, each with a stated reason)

No numeric literal used as a tolerance appears outside `config.py`, named module constants
or `units.py` (enforced by `tests/test_hygiene.py`, which AST-scans the physics packages for
unnamed floats and large integers). The table below is generated from `NumericalConfig`
(`python docs/engine/generate_docs.py`); `tests/test_docs_in_sync.py` fails if it drifts.

<!-- BEGIN TOLERANCES (generated by docs/engine/generate_docs.py; do not edit) -->

| Field | Default | Meaning / reason |
|---|---|---|
| `ray_min_advance_mm` | `1e-07` | Minimum ray parameter t for a valid hit; prevents re-intersecting the surface just left. |
| `parallel_dot_tol` | `1e-12` | |n.d| below this is 'parallel': no intersection is reported. |
| `laboratory_reach_mm` | `1000.0` | A declared numerical choice, the reach of a laboratory scene: on a leg that leaves the scene, a plane without an active area counts part of a beam up to max(this, the scene's support along the ray) from the leg's start. It is recorded with the run and fixed at this value: no other value is accepted. |
| `grazing_warn_aoi_deg` | `85.0` | AOI above which GRAZING_INCIDENCE warns. |
| `grazing_fail_aoi_deg` | `89.9` | AOI above which the interaction is rejected. |
| `antiparallel_tol` | `1e-12` | 1+cos(angle) below which two directions are treated as antiparallel (basis transport). |
| `unit_vector_tol` | `1e-09` | Deviation |1-|v|| up to which a direction's components are used unchanged; beyond it the direction is normalized. |
| `zero_vector_tol` | `1e-12` | Vectors with norm below this are zero-length. |
| `quaternion_norm_tol` | `1e-06` | Deviation |1-|q|| beyond which a stated quaternion is refused: the shared Project contract rejects it and the Engine input builder refuses it (POSE_INVALID). The Engine itself takes any non-zero, finite quaternion as the rotation of q / |q|, whatever its length; a zero one is refused by the request contract (INVALID_INPUT). |
| `aperture_edge_tol_mm` | `1e-09` | Inclusive aperture boundary tolerance (floating-point edge). |
| `tir_warn_margin` | `0.001` | Relative distance to the critical angle that triggers TIR_BOUNDARY. |
| `power_epsilon_W` | `1e-15` | Power at or below this terminates a path (ZERO_POWER); 0: no absolute floor. |
| `energy_conservation_tol` | `1e-09` | Tolerance for R+T<=1 and unit-norm Jones checks. |
| `negligible_power_fraction` | `1e-09` | Share of a source line's power below which light is not followed, probed or listed: the level of the light a result leaves out by statement (a plane reads 'nothing left out' relative to it). A caller setting. |
| `basis_orthonormality_tol` | `1e-09` | Allowed deviation of a transported transverse basis before a warning. |
| `max_interactions_per_path` | `64` | Guard against endless cavity loops: physical surface interactions per path (observation planes are virtual and do not count). A path that would exceed it stops (MAX_INTERACTIONS, UNKNOWN); its remaining power is the ledger's INTERACTION_LIMIT, so a result that light could still change is UNKNOWN. |
| `max_paths` | `512` | Trace budget: beam paths per run (source spectral lines and every branch a splitting surface creates), the strongest first. Beyond it the weakest source lines and branches are not traced; one MAX_PATHS issue gives their count and power, and a target no traced beam reaches is UNKNOWN (MAX_PATHS), not 0 W. Each traced path also carries its pupil rays into the ray data (about 70 kB of run files per trace step). |
| `max_trace_steps` | `2048` | Trace budget: chief-ray steps over all paths of a run (a path takes one step to each surface it reaches and one to leave the scene; a source line through N surfaces takes N + 1: 512 lines through one lens to a monitor plane). Once spent, the path being traced stops there and the weaker queued branches are not traced (one MAX_PATHS issue); with max_paths it bounds a run's time, memory and run files (about 150 MB) in a laptop process or a browser tab. |
| `retarder_warn_aoi_deg` | `20.0` | Retarder/polarizer AOI above which RETARDER_OBLIQUE_INCIDENCE warns. |
| `birefringent_interface_warn_aoi_deg` | `2.0` | Angle of incidence (deg, the incident wave normal on a crystal face) above which the face's power split between the crystal's ordinary and extraordinary waves warns BIREFRINGENT_INTERFACE_APPROXIMATE: it uses the isotropic Fresnel (or coating) transmittance at each wave's index, exact at normal incidence for any optic-axis direction. |
| `birefringent_gaussian_rel_tol` | `0.01` | A Gaussian beam crossing a uniaxial crystal as an extraordinary wave is carried with one diffraction index while its two axes diffract with two: once the accumulated difference of its reduced q-parameter (the uncertainty of its waist position, mm of reduced path) exceeds this fraction of its reduced Rayleigh range the Gaussian state is out of model scope (GAUSSIAN_ANISOTROPIC_MEDIUM). |
| `nonlinear_depletion_rel_tol` | `0.02` | Second-harmonic generation is computed with an undepleted pump: once the estimated relative amount by which that overstates the harmonic power (8/9 of the conversion efficiency for one fundamental wave, more for two unequal ones) exceeds this, the conversion is out of model scope (NONLINEAR_CONVERSION_OUT_OF_SCOPE). |
| `nonlinear_focusing_rel_tol` | `0.02` | A crystal whose second-harmonic generation takes the beam radius as constant along it (near field) is out of model scope when that result differs from the one that follows the beam's focusing through the crystal by more than this fraction of the larger of that focused result and the focused result at phase matching (dk = 0) (NONLINEAR_CONVERSION_OUT_OF_SCOPE). Where it is within this fraction of the focused result at dk = 0 but not of the focused result itself (near a zero of the tuning curve, which the focusing fills), the focused result is reported. |
| `nonlinear_quadrature_rel_tol` | `1e-07` | Relative agreement of two successive node counts at which the Gaussian-beam overlap integral of second-harmonic generation stops refining; a result whose last two values still differ by more than nonlinear_focusing_rel_tol is out of model scope. |
| `nonlinear_harmonic_gaussian_rel_tol` | `0.02` | Walk-off spreads the harmonic generated along a crystal over a strip: once the relative growth of its rms width along the walk-off, (s L / w0)^2 / 3, exceeds this, the harmonic is traced without a Gaussian state (HARMONIC_BEAM_NOT_GAUSSIAN); its power is unaffected. |
| `nonlinear_beam_quality_tol` | `0.05` | Second-harmonic generation is computed for a TEM00 fundamental: a beam whose M^2 exceeds 1 by more than this on either axis is out of model scope (NONLINEAR_CONVERSION_OUT_OF_SCOPE). |
| `nonlinear_linewidth_rel_tol` | `0.02` | Second-harmonic generation is computed for one frequency: for a line with a stated linewidth (or coherence length) the phase-matching factor sinc^2(dk L / 2) averaged over its Gaussian spectrum may differ from the line centre's by at most this fraction of the larger of the two (the centre's value is reported); beyond, the conversion is out of model scope (NONLINEAR_CONVERSION_OUT_OF_SCOPE). |
| `sag_bracket_samples` | `65` | Samples of F(t) = z(t) - sag(t) along a ray's crossing of a sag surface's bounding box, used to bracket the nearest intersection when the surface has even-asphere, axicon or two-variable terms (conic-only surfaces are solved in closed form); on an even asphere two intersections closer than one sample spacing (a ray grazing a local bump) are not separated, on a two-variable shape they are, at the closest approach between the samples. |
| `sag_max_iterations` | `64` | Iteration cap of the bracketed Newton / bisection refinement of a sag-surface intersection (bisection alone halves the bracket each step, so the cap is never reached for brackets below 1e6 mm). |
| `sag_intersection_tol_mm` | `1e-12` | A sag-surface intersection is converged when the last Newton / bisection step changed the ray parameter by at most this (mm); a ray whose bracket does not converge continues as missing the surface, and on a two-variable shape the run reports it (SAG_INTERSECTION_NOT_CONVERGED). |
| `sellmeier_pole_rel_tol` | `1e-09` | Relative |l^2 - C_i| / l^2 below which a Sellmeier pole is considered hit. |
| `tabulated_range_tol_nm` | `1e-09` | Slack (nm) on the inclusive wavelength validity bounds of materials and coatings. |
| `tabulated_aoi_tol_deg` | `1e-09` | Slack (deg) on the AOI bounds of tabulated coatings. |
| `gaussian_warn_divergence_rad` | `0.1` | Half-angle divergence (NA~0.1) warning threshold. |
| `gaussian_scope_divergence_rad` | `0.3` | Half-angle divergence above which the paraxial Gaussian model is out of scope. |
| `gaussian_max_w_over_radius` | `0.2` | Beam radius / |surface radius of curvature| limit for ABCD validity. |
| `gaussian_max_aoi_deg` | `60.0` | Tilted-surface ABCD validity limit on AOI. |
| `gaussian_max_ray_spread_ratio` | `2.0` | Pupil-ray RMS spread / the largest spread an aberration-free beam with the arrival's Gaussian envelope can have (1 for a full pupil sample, at most sqrt(2) for any subset of it), on the observation plane and on the Gaussian waist plane, above which the beam is aberration dominated (GAUSSIAN_ABERRATION_DOMINATED, OUT_OF_MODEL_SCOPE): the Gaussian radius there is then about this factor too small or worse. |
| `gaussian_clip_warn_fraction` | `0.001` | Truncation power loss that warns. |
| `gaussian_clip_scope_fraction` | `0.1` | Truncation power loss above which the Gaussian state is invalidated. |
| `gaussian_astig_align_tol` | `1e-09` | |sin(phi)cos(phi)| below which beam axes are aligned with (t,s). |
| `gaussian_round_tol` | `1e-09` | Relative difference below which the two transverse axes are 'round'. |
| `gaussian_umbilic_rel_tol` | `1e-09` | Relative difference of a sag surface's two principal curvatures at the chief-ray hit below which the surface is treated as umbilic there (no preferred axes): its principal directions then need not align with (t, s). |
| `gaussian_plane_axial_spread_fraction` | `0.1` | Axial spread across a tilted plane footprint vs z_R that triggers a warning. |
| `aperture_quad_angular` | `256` | Angular trapezoid nodes for circular-aperture Gaussian truncation (periodic, spectral). |
| `aperture_quad_sector_nodes` | `48` | Gauss-Legendre nodes per panel/sector for off-axis circular and rectangular truncation (building the rule grows with the square of the count; a count whose rule the machine cannot hold is refused by name). |
| `aperture_quad_extent_sigmas` | `6.0` | Gaussian integrals are truncated at this many beam radii from the beam centre (exp(-72) ~ 0). |
| `pupil_ring_count` | `8` | Equal-power rings in the default pupil sampling. |
| `pupil_points_per_ring` | `12` | Points on each ring of the default pupil sampling. |
| `first_order_position_step_mm` | `0.01` | Source-plane displacement of the differential rays (central differences of the first-order ray-transfer matrix): small against apertures and focal lengths (truncation error ~(step/f)^2 < 1e-7), large against coordinate roundoff. |
| `first_order_angle_step_rad` | `0.0001` | Source-plane tilt of the differential rays (central differences; truncation error ~step^2 = 1e-8). |
| `first_order_unitless_resolution` | `1e-06` | Reported resolution of the dimensionless A and D blocks of the first-order matrix (entries are rounded to it: finite-difference roundoff ~1e-12, so the result is identical on every runtime). |
| `first_order_b_resolution_mm` | `0.0001` | Reported resolution of the B block (mm per rad). |
| `first_order_c_resolution_per_mm` | `1e-08` | Reported resolution of the C block (per mm). |
| `power_ledger_fraction_resolution` | `1e-12` | Resolution, as a fraction of the beam power, of ledger entries computed as a small difference of two powers (aperture truncation): rounded so the ledger is identical on every runtime. |
| `interference_min_crossing_deg` | `1e-06` | Crossing angle below which the fringe period is undefined. |
| `interference_wavelength_tol_nm` | `1e-09` | Two arrivals must agree in wavelength within this to interfere. |
| `interference_min_overlap` | `0.001` | Spatial overlap ratio below which beams are considered not overlapping (visibility < 0.1 %). |
| `interference_collinear_period_ratio` | `10.0` | Fringe period / beam diameter above which two beams are treated as collinear (single-fringe detector). |
| `max_interference_pairs` | `256` | Cap on automatically analysed beam pairs per observation. |
| `max_return_analysis_work` | `4000000` | Work budget of the return-light analysis of the whole run, in work units: one per cone test of the screen for closed returns of three or more surfaces, 1000 per pair of reflecting surfaces screened, and the fold-solid analysis' tests (about 4 microseconds per unit on a desktop CPU). The analysis runs only for a wavelength whose light left out of the trace a power or exposure upper bound reads; the wavelengths share the budget in the order the run reads them (each record's work_before). An analysis that reaches the budget stops: the upper bounds that depend on it are withheld (RETURN_ANALYSIS_BUDGET, a notice; the execution stays COMPLETED and the budget is listed as exhausted in phase return_analysis), never replaced by a value the analysis did not finish. |
| `interference_det_tol` | `1e-30` | Determinant floor for the summed envelope matrix. |
| `max_scatter_origins` | `64` | Work budget: scattering origins whose scattered light the run traces as rays, the strongest first; the light of the others stays booked as scattered at its surface (SCATTERED_LIGHT_NOT_FOLLOWED on the planes it may reach). |
| `max_mesh_patches` | `1024` | Work budget: traced planar patches one faceted surface may have; a surface with more is refused. |
| `rating_wavelength_tol_rel` | `0.02` | Relative distance from a rating's test wavelength within which the light's matches it. |
| `rating_pulse_duration_tol_rel` | `0.1` | Relative distance from a rating's test pulse duration within which the light's matches it. |
| `rating_repetition_rate_tol_rel` | `0.1` | Relative distance from a rating's test repetition rate within which the light's matches it. |
| `rating_beam_diameter_tol_rel` | `0.05` | Relative distance from a rating's test beam diameter within which the beam's matches it. |
| `rating_polarization_fraction_tol` | `0.01` | Largest share of a beam's power outside a rating's stated polarization for it to match. |
| `rating_unlocated_light_share` | `0.01` | Largest share of the light a surface can receive that light of unknown size, place or state may carry for a rating condition to be verified. |
| `design_point_wavelength_rel` | `0.005` | Relative distance from a single stated design wavelength of a coating or zero-order retarder within which its published values are used. |
| `design_point_retarder_wavelength_rel` | `0.0005` | The same for a multi-order retarder (its retardance moves m + 1 times faster). |
| `design_point_angle_metallic_reflector_deg` | `5.0` | Distance (deg) from a metallic reflector's single design angle within which its values are used. |
| `design_point_angle_dielectric_high_reflector_deg` | `3.0` | Distance (deg) from a dielectric high reflector's single design angle within which its values are used. |
| `design_point_angle_anti_reflection_deg` | `3.0` | Distance (deg) from an anti-reflection coating's single design angle within which its values are used. |
| `design_point_angle_beamsplitter_deg` | `2.0` | Distance (deg) from a beamsplitter coating's single design angle within which its values are used. |
| `design_point_interference_shift_fraction` | `0.25` | Share of the narrowest spectral width an interference filter's bands leave that the angle shift of its spectrum may reach within its single design angle's tolerance. |
| `design_point_interference_angle_max_deg` | `3.0` | Largest angle tolerance (deg) an interference filter's single design angle gets. |
| `fd_step_translation_mm` | `0.001` | Central-difference step for translation DOFs. |
| `fd_step_rotation_rad` | `1e-05` | Central-difference step for rotation DOFs. |
| `lm_initial_damping` | `0.001` | Initial Levenberg-Marquardt damping. |
| `lm_damping_up` | `10.0` | Damping multiplier after a rejected step. |
| `lm_damping_down` | `0.1` | Damping multiplier after an accepted step. |
| `lm_max_damping` | `1000000000000.0` | Damping above which LM gives up. |
| `lm_max_iterations` | `100` | Iteration cap for LM. |
| `lm_cost_tol` | `1e-18` | Converged when the cost falls below this. |
| `lm_step_tol` | `1e-12` | Converged when the scaled step norm falls below this. |
| `lm_residual_tol` | `1e-06` | Default weighted-residual norm regarded as converged when the spec gives none (sensor units, e.g. mm). |
| `lm_scaling_floor` | `1e-12` | Floor on diag(J^T J) in Marquardt scaling so unobserved DOFs stay bounded. |
| `lm_gain_accept` | `0.0` | Minimum actual/predicted reduction ratio to accept a step. |
| `jacobian_rcond` | `1e-10` | Singular values below rcond*sigma_max are rank-deficient. |
| `observability_participation_min` | `0.25` | An actuator is reported unobservable when at least this fraction of its unit motion lies in the null space of the sensor Jacobian (0.5 would sit exactly on the symmetric two-DOF case). |
| `jacobian_condition_warn` | `1000000.0` | Jacobian condition number that warns. |
| `capture_min_power_fraction` | `0.001` | Detected/available power below this means the beam is lost. |
| `capture_scan_max_points` | `4096` | Maximum points of the coarse-capture raster scan. |
| `layout_geometry_tol_mm` | `1e-06` | Overlaps/clearance deficits below this are ignored. |
| `layout_arc_resolution` | `32` | Shapely buffer segments per quarter circle. |
| `beam_envelope_k` | `2.0` | Beam envelope radius = k*w (k=2 -> 1-exp(-8)=99.97% power). |
| `moving_envelope_rotation_samples` | `9` | Angles sampled for rotational sweeps. |
| `layout_height_tol_mm` | `1e-06` | Height intervals overlapping by less than this are disjoint. |
| `hash_float_sig_digits` | `9` | Significant digits kept when hashing results. |
| `hash_float_abs_floor` | `1e-12` | Result values below this magnitude hash as 0. |
| `cross_runtime_rel_tol` | `1e-09` | Relative tolerance for cross-runtime comparison. |
| `cross_runtime_abs_tol` | `1e-12` | Absolute tolerance for cross-runtime comparison. |

<!-- END TOLERANCES -->

Changing a default (or adding a setting) after a published release increments `NUMERICAL_CONFIG_VERSION`, once per release:
later changes before the next release keep that unreleased version. Every Engine input hash moves with a change, so runs
made by an earlier build read as not current.

These resolutions are fixed in the code rather than set, because a pose stated within `quaternion_norm_tol` enters every
computation as `q / |q|`, which may differ from the unit quaternion in its last bit:

* an angle of incidence below 1e-7 deg reads 0 (`math.exact_trig._INCIDENCE_RESOLUTION_DEG`; section 4): a direction
  refracted through lenses carries up to about 4e-9 deg of rounding. This holds for every angle the Engine states,
  including the angle a Gaussian state records against `gaussian_max_aoi_deg` at a thin lens or a curved surface;
* the return analysis decides which side of a surface another surface lies on, and whether a surface meets the hull
  between two others, to a length resolution of `1e-9 mm + 1e-12` times the largest coordinate compared
  (`validation.resonance.SIDE_ABS_TOL_MM`, `SIDE_REL_TOL`); within it the answer is that the surface lies on either side
  and meets the hull. Coplanar faces (a window's, a cube splitter's) sit exactly level or a few 1e-14 mm apart by
  rounding, and the screen's work count and outcome do not follow that bit;
* the iterations of a solid's first-return sums and of their certificate stop where a round moves no entry by more than
  1e-12 of the largest (`validation.solid_folds.CERTIFY_RESOLUTION`), not where the sums repeat bit for bit: they
  converge geometrically, so the round where they stop, and the return analysis's work count, do not follow the last
  bits of the poses. The certificate keeps its margin (`MARGIN`, 1e-9) and its row check wherever the iteration stops;
* the share of a beam that passes beside an element's support, or beside a surface it coincides with, is resolved upward
  (an upper bound) to `power_ledger_fraction_resolution` of the beam power, the resolution of the beam's
  `APERTURE_TRUNCATION` ledger entry: it is
  `1 - f` of a fraction `f` inside next to 1, and `f` carries the quadrature's rounding of the whole beam (a few 1e-15),
  which follows the last bits of the poses. That light, branched or left out as negligible (the faint-light bounds of the
  exposures), carries no digits below that resolution;
* a beam's footprint on a surface (`footprint_semi_axes_mm` of an exposure hit, the largest beam diameter a rating
  condition compares) is stated where `|cos(AOI)|` is at least 1e-5 (`validation.exposures.FOOTPRINT_COS_RESOLUTION`, an
  AOI within 6e-4 deg of grazing): its long axis is `w / cos(AOI)`, and `cos(AOI)` carries about 1e-15 of the rounding of
  the beam's direction and the surface's normal. Below it the footprint is not known (null, as at grazing incidence), and a
  beam that passes beside a surface it grazes has no known size there (it crosses the surface's plane `1 / cos(AOI)` away):
  the hit reads UNKNOWN. The footprint's short axis and its determinant are taken from the projected beam axes
  (`|bx_u by_v - bx_v by_u| = cos(AOI)`), not from the cancelling `a c - b²` of their quadratic form;
* a beam that met a surface near normal incidence (`|d̂×n̂|` just above `NORMAL_INCIDENCE_SIN`, 1e-6) carries a plane of
  incidence whose direction turns with that bit by up to 8 units of roundoff over `|d̂×n̂|`: the components of its
  transport basis, of its Jones vector in that basis, of its Gaussian axes and its Stokes `s1` / `s2` are resolved to
  `math.vectors.TRANSPORT_BASIS_RESOLUTION` (1.8e-9) absolute, not to the cross-runtime tolerance (a fold prism's beam
  4.4e-6 from normal on a lens: 3e-11). Its powers, positions, directions and its field in the lab frame keep the
  cross-runtime tolerance;
* a Gaussian beam's waist distance `z = Re q` is resolved to 1e-11 mm (`optics.gaussian.state.WAIST_DISTANCE_RESOLUTION_MM`),
  so its curvature radius `|q|² / z` carries a relative error of 1e-11 mm / `|z|`: within 0.01 mm of the waist
  (`FLAT_WAIST_DISTANCE_MM`, where that error exceeds the cross-runtime relative tolerance) the reported
  `curvature_radius_x_mm` / `_y_mm` is null (flat), unless the wavefront there departs from flat by more than 1e-3 rad across
  the beam (`|z| / z_R`, `FLAT_WAVEFRONT_PHASE_RAD`: a Rayleigh range below 1000 `|z|`), where the radius is stated. A beam
  focused on its detector (`z` a residue of 1e-11 mm) reads flat, not ±1e12 mm;
* a chief ray does not cross an observation plane with an active area beyond the scene's reach along it
  (`optics.rays.tracer.Tracer._off_area_far`, section 2): such a crossing lies off the area, and it is where a ray nearly
  parallel to the plane meets it (1e14 mm on for a ray 2e-12 off parallel to a plane 200 mm away), so whether it exists
  would follow the last bit of the ray's direction against `parallel_dot_tol`;
* queued packets, probe branches and scatter origins are taken strongest first by `optics.rays.scheduler.order_power`,
  their power on a binary grid of the run's launched power times 2^-40 (`order_quantum`: the largest power of two at most
  `power_ledger_fraction_resolution`), ties in queue order. Beams equal in exact arithmetic (the halves of
  a 50:50 split, symmetric ghosts) differ in their last bits by the poses' rounding; on the grid they tie, so where a budget
  binds the same beams are traced in every run. Only the order reads the grid: every booked power is the beam's own;
* a polarizer splits the light it meets into the share along its axis, `|E·a|²`, and the rest, `|E|² - |E·a|²`, each
  resolved to `power_ledger_fraction_resolution` of the beam's power (`optics.rays.tracer.Tracer._do_polarizer`), the
  resolution of a ledger entry taken as a small difference of two powers: within it of the whole power the light is along
  the axis, within it of 0 it is across. For light along (across) the axis the other
  share is 0 in exact arithmetic and a few 1e-19 W of rounding of a 1 mW beam that follows the poses' last bits; resolved,
  the power ledger books no `POLARIZER_REJECTION` (`ELEMENT_TRANSMISSION_LOSS`) of it.

## 4. Determinism

* No wall-clock, no `id()`, no dict-order dependence in results (all
  collections are sorted by stable ids before output).
* Random numbers only in Monte Carlo, from `numpy.random.PCG64(seed)` raw
  64-bit words; uniform `u = (w >> 11) · 2⁻⁵³`; normal via Box–Muller using
  `math.log`/`math.cos`. The seed is `control_spec.monte_carlo.seed`, else
  `configuration.random_seed`, else 0 (`DEFAULT_MONTE_CARLO_SEED`); the PRNG
  identifier `PCG64+BoxMuller-v1` and the seed used go into the manifest.
* NumPy `BitGenerator` raw streams are stable across versions and platforms;
  `Generator` distribution methods are deliberately not used.
* Same bits in every runtime. A native CPython / NumPy build and Pyodide (WebAssembly) share IEEE-754 binary64 with
  correctly rounded `+ − × ÷ √`; they do not share libm (`sin`, `cos`, `asin`, `atan2`, `exp`, `log`, `pow`; Python's
  `x ** 2` on a float is `pow`, `abs()` of a complex is `hypot`), NumPy's SIMD transcendental kernels (`numpy.exp`, also
  complex), BLAS / LAPACK, or fused multiply-add: on a CPU with FMA a native NumPy fuses products and sums in `einsum`,
  `matmul` (`@` on 2-D or stacked operands) and complex multiplication, WebAssembly does not. Each changes last bits, and
  `k₀·OPL` turns one last bit of a path length into 10⁻¹⁰ rad of phase. The trace and every quantity a run reports
  therefore use:
  * `math.vectors.dot` / `matvec` / `matmul` (separately rounded products summed in index order) for every projection,
    ray–surface intersection, direction, frame transport and pose (component poses, surface frames, observation planes);
  * `math.exact_linalg` (`cmul`, `cvdot`, `power2`, `abs2`, `det2`, `solve2`, `eigvalsh2`, `inverse_2x2`,
    `singular_values`) for complex products, Jones arithmetic and 2 × 2 matrices (Gaussian envelopes, overlaps);
  * `math.exact_trig` (`atan2`, `atan2_deg`, `acos_deg`, `incidence_deg`, `cis`, `exp`, `log`, `log10`, `pow10`) for angles
    of incidence (`incidence_deg`: from `|n x d|` within 1e-4 of normal incidence in the cosine, where the cosine's last
    bit is 8.5e-7 deg, and 0 below 1e-7 deg, which a refracted direction does not resolve), phases, polarization
    ellipses, image rotation, phase factors, attenuator laws, the pupil-ray samples, a fluorescence
    sample's absorbed share `1 - 10^-A` (an Engine input) and a chopper's fundamental and noise weight (a lock-in's
    noise-equivalent bandwidths are exact binary fractions over the time constant);
  * `math.exact_erf` (`erfc`, `erf_interval`) for the closed-form parts of the Gaussian aperture-truncation integrals
    (absolute error below 2e-15, relative error below 4e-15 in the tails);
  * products, not `** 2`, on Python floats.

  A Python float and a NumPy float64 give the same bits for `+ − × ÷ √` (each operation is correctly rounded on its own and
  never fused with the next), so these helpers evaluate one vector, one small matrix or one angle in Python floats and
  arrays elementwise along their longest axis; the order of the operations on each value is what fixes its bits, not the
  memory layout (a NumPy reduction such as `sum` does depend on its operand's layout, so arrays that are reduced keep theirs).
  An index-order sum stays a loop of `+`: `math.fsum`, `math.sumprod`, `numpy.sum` and BLAS sum in another way. A total of
  reported floats whose order carries no meaning (powers, spectral weights) uses `math.vectors.correctly_rounded_sum`
  (`math.fsum`, the same bits in any order on every runtime), never the built-in `sum`: it folds on Python 3.11 and compensates
  from 3.12, so its last bits depend on the interpreter.

  Where a value still passes a NumPy transcendental or LAPACK kernel, it is rounded to a stated resolution before it is
  reported (`power_ledger_fraction_resolution`, `first_order_*_resolution`). Checked byte for byte
  (`tests/test_packaging_and_pyodide.py`): run files of beams crossing
  observation planes at oblique angles, tilted windows and attenuators, the catalog wedge prism with two lines, and those
  benches turned as a whole about an oblique axis are identical natively and in Pyodide, apart from the runtime record
  (`runtime_kind`, `manifest.runtime`). `compare_results` (`cross_runtime_rel_tol` / `_abs_tol`) remains the comparison
  for results of different Engine builds or configurations.
* Hashed inputs never pass through libm: Project decimals become the nearest binary64, derived values are exact decimal
  arithmetic (`+ − × ÷ √`), and the trigonometric values of an elliptical source's Jones vector are decimal power series.
  Coating interpolation (bilinear tables, linear phase tables) is plain `+ − × ÷` on those inputs.

## 5. Error reporting, never NaN

All public numeric outputs are finite. If a computation cannot produce a
finite meaningful number it returns a structured `ValidationIssue`
(`code`, `message`, `location`, `affected_targets`, `numerical_context`,
`suggested_action`) and the dependent metric has status `UNKNOWN`,
`FAIL` or `OUT_OF_MODEL_SCOPE` with `value = null`. Result models forbid
`NaN`/`Inf` (`allow_inf_nan=False`).

## 6. Canonical serialization and hashing

* JSON, UTF-8, keys sorted, no whitespace (`separators=(",",":")`).
* Inputs hashed from the validated model dump (`mode="json"`, so defaults are
  materialized and aliases normalized).
* Floats in **results** are quantized before hashing: value `v` with
  `|v| < hash_float_abs_floor` → `0.0`; else round to `hash_float_sig_digits`
  significant digits and print with `repr`. The quantization is not what makes
  `result_hash` the same in every runtime (rounding can straddle a boundary):
  the run's arithmetic is (§4). Wrapped phases are already reported at their
  resolution (§2, phase resolution).
* Hash = SHA-256 hex. No secret keys: hashes are integrity/reproducibility
  aids, not authenticity proofs.
* `trace_hash` covers the ordered ray-event list of the chief rays.
