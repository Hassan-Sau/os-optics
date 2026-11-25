# os-optics — Scientific results and export

One local run writes one run directory. Everything below is read from that directory, from its files held in memory, or
from an evaluation held in memory (`Evaluation.results()`, [below](#results-of-a-selective-evaluation)); nothing needs a
server. The tables, monitors, rays, wavefront, irradiance, handoff and exposures read the recorded results; the imaging,
ghost and field analyses trace the run's own request again with their own sampling (each section says what it traces).

| File | Content |
|---|---|
| `simulation_result.json` | the Engine `ValidationResult`: observation planes with their resolved frame and every arrival's beam state, beam paths and segments, two-beam interference, coherent superpositions, power budget, per-surface exposures and power-rating comparisons, targets, issues, manifest; per surface with a two-variable sag shape, its normalization and what the intersection solver did on it (`surface_profiles`) |
| `ray_data.json` | `opensci.ray_data` 2.0: per beam path, the chief ray and every pupil ray at each step of the trace (source or branch point, every surface reached): point, direction, optical path length, surface id; ring layout, launch offsets, equal power weight, how each ray ended. Bound to the result by `input_hash` / `trace_hash` |
| `manifest.json` | schema 1.2: status, identities (`engine_run.engine_package`: the distribution that produced the run, `os-optics`), every file with its SHA-256, the ray data's trace hash; a 1.1 manifest (os-optics 0.10.0 / 0.11.0, without `engine_package`) still verifies |

`os-optics project verify RUN` / `verify_run_artifacts(RUN)` check every hash and cross-reference, including that
the ray data belongs to the Engine result. A run directory of manifest 1.0 (written before `ray_data.json` existed)
still verifies; its ray-based results (`rays`, `wavefront`) answer `RAY_DATA_NOT_AVAILABLE`.

## Python

```text
from os_optics.results import load_run
results = load_run("run")            # a run directory, its {file name: bytes}, or a LocalOpticalRun
```

| Call | Answer |
|---|---|
| `describe()` | run identity (Engine version / build, input / result / trace hashes, Project hash, catalog release; `engine_input_hash` repeats `input_hash` under the name the network export and `check_project` use), status, integrity, the result types present, conventions, files |
| `monitors()` / `monitor(id)` | observation planes: resolved frame, active area, absorbing extent, arrivals; in full: every `ArrivalResult`, crossings outside the active area, power by wavelength, coherent superpositions, two-beam interference, the stated detector's reading (`detection`), targets |
| `paths()` / `segments(path)` | the beam-path tree (parent, children, branch surface / kind, depth, termination, power at start / end, OPL, planes crossed) and every chief-ray segment |
| `power_budget()` | `PowerBudget` plus flat rows: per source line the input, every sink, the power observed at each monitor and the residual |
| `rays(monitor, path, crossing_index)` | chief + pupil rays at each observation crossing: u, v, x, y, z, direction (design and plane frame), OPL, launch offset, ring, power weight |
| `wavefront(monitor, path, crossing_index, reference, center)` | optical path difference per pupil ray, piston removed, RMS / PV, or `UNAVAILABLE` with a reason |
| `irradiance_map(monitor, wavelength_nm, half_extent_mm, samples, max_untraced_fraction)` | irradiance grid of the arriving Gaussian beams, incoherent and coherent sums; without `wavelength_nm` on a plane several wavelengths reach, their sum |
| `handoff(monitor, path, crossing_index, element)` | the transverse-plane handoff: every arriving beam in the plane's (u, v, n) frame (q, waist, curvature, M², Jones vector and basis, power, medium) with its validity and the TEM00 field-construction convention; with `element`, that component's clear apertures in the same frame ([below](#transverse-plane-handoff)) |
| `grating_orders(element)` | every diffraction order of every beam's pass through a grating (where more than 129 propagate, as for an acousto-optic deflector's coarse period, those it follows): order, `propagation`, `followed`, `order_path_id`, direction and `exit_angle_deg`, and apart from them `power_state` (`power_W`, `power_upper_bound_W`, `efficiency`, `relative_response`; `OUT_OF_MODEL_SCOPE` with `power_out_of_scope_at` where the incident beam's power is out of model scope), `polarization_state`, `phase_state`, `response_basis` |
| `exposures(element)` | every element surface the traced beams meet: per beam its power, both Gaussian radii, AOI, footprint, peak irradiance, P / d and pulsed peak fluence; per surface the combined values, the groups of mutually coherent beams and the comparison with the element's power ratings (bounds, every condition with its state, the applicability); per element its rating status and largest utilization (see *Exposures and power ratings*) |
| `metrics()` | every target metric's descriptor (unit, definition, sign convention, operands, frame, selectors, model basis, validity, version, and `result_fields`: where its value appears among the results) |
| `detector_response(monitor, detector, composition, **irradiance)` | photons, electrons, ADU, saturation, noise and SNR per pixel of a stated detector, from this plane's irradiance map (see *Detector response*) |
| `pupil_field(monitor, request, mask_files)` | the scalar low-NA focal field of a pupil mask lit by the beams at this plane (see *Focal field of a pupil mask*) |
| `imaging(monitor, request=None)` | point spread function, MTF, encircled energy and wavefront of every field at an image plane, an analysis document bound to the run (see *Imaging analysis*) |
| `ghosts(monitor, request=None)` | the ghost images of every lens or window face's reflections at an image plane, up to a bounce order: bounce surfaces, direction, focus, relative power and irradiance, with a bound on the ghosts not listed (proven where every ghost surface met a valid Gaussian beam on a plane or a sphere); an analysis document bound to the run (see *Ghost paths*) |
| `focal_field(monitor, request)` | the vector focal field of an ideal aplanatic objective lit by the beams at this plane, an analysis document bound to the run (see *Vector focal field of an objective*) |
| `export(kind, fmt, **filters)` | bytes of one export |

A bare Engine run: `results_from_engine(*validate_with_ray_paths(request))`.

### Results of a selective evaluation

An evaluation of selected quantities (`os_optics.local.EvaluationSession().evaluate(project, outputs=...)`, `os-optics
project evaluate`) gives its results from memory: `evaluation.results()` is a `RunResults` built from the evaluation's own
objects, without writing or parsing a file. It answers every query of what the evaluation computed; a query of a part it
did not compute raises `ResultsError` with code `NOT_COMPUTED`, whose message names each capability to request
(`os_optics.results.QUERY_REQUIRES`, every capability a query reads: `monitors` / `monitor` / `arrivals` /
`superpositions` / `irradiance_map` / `imaging` / `ghosts` / `field` read `plane_readings`; `handoff`, and
`focal_field` / `pupil_field` lit by it, `plane_readings`, `power_ledger` and `plane_bounds` (a beam's validity reads the
truncations the ledger books and the plane bounds' issues); `paths` / `segments` `beam_trace` and `secondary_rays` (the
rays of an extended emitter and of scattered light are paths of the result; the secondary rays do no work in a design
without either); `grating_orders` / `nonlinear` `beam_trace`; `rays` / `wavefront` `ray_paths` and `plane_readings`;
`power_budget` `power_ledger`; `exposures` `surface_exposures`; `detection` `plane_bounds`; `targets` `requirements`);
`evaluation.extend(outputs=[...])` computes it, reusing the Engine products the evaluation left in its session; for an
evaluation made with `exclude=` whose exclusions it needs, the refusal names each of them and a new evaluation without them. The
result of an Engine evaluation (`results_from_engine(os_optics.evaluate(request, capabilities=[...]))`) is refused the
same way, its message naming `os_optics.evaluate(request, capabilities=[...])`. Without the plane bounds a plane's `quantity_state` is `NOT_COMPUTED`, its `power_complete` false,
its `untraced_power_W` null and its power the traced power -- never a known zero or a bound; its irradiance map holds the
incoherent map, and the coherent map, the untraced share and a uniformity read `NOT_COMPUTED` (`needs: ["plane_bounds"]`).
`monitor(id)` leaves out the interference, the targets, the detection and the polarization mixtures when the capability
that produces each did not run and names them under `not_computed`; `describe()["available"]` gives `null` for them, for the path list without the secondary rays
and for `wavefront` without the ray paths or the plane readings. `describe()["evaluation"]` states the scope; a model-scope finding names every declared target that reads its light and
whose metric reads a kind of quantity the finding touches (`affected_target_ids`), whether or not the evaluation assessed that target. A target reads the light
of the beams (path and crossing) and sampled lines its selection reads on its plane -- its source, wavelength, path, crossing and pair; a target that selects no beam
reads every beam on its plane; a polarization metric (degree of polarization, Stokes, azimuth, ellipticity) whose path selects one polarization component reads
both components of that route (same source line, crossing and lineage) -- and, on the element it names, the beams of its source and wavelength, and for
`power_rating_utilization` every beam on the element, whatever source or wavelength the target states (the element's exposure by all of them). On its plane, a
beam its selection does not read, such as a beam of another wavelength outside a lens's band, leaves its assessment as it is. Each finding also lists the arrivals its light reaches (`affected_arrivals`: `observation_id`, `path_id`,
`crossing_index`), the sampled lines its light reaches (`affected_sampled_lines`: `observation_id`, `source_id`, `spectral_index`; an extended
emitter's line whose rays met the element, on the planes its rays cross; the line of the light one surface scattered
(`sampled_light[].scatter_surface_id`, `parent_path_id`) whose rays met the element, on the planes that light crosses; and the line of the
light a surface scattered out of a beam the finding reaches behind the point of the finding: on the finding's own path from where that path
meets the scattering surface at or after the finding point, on a path branched after it wherever the light scatters; such a line's planes
join `affected_observation_ids`, and the targets that read it join `affected_target_ids`), the validity check it states per value the light met (`validity`), and per
check the light met at that value (`validity_reach`: the targets that read it, `affected_target_ids`, its arrivals, `affected_arrivals`, and its
sampled lines, `affected_sampled_lines`); the value of a target it names carries the checks of the light the
target reads in a `value_basis` of kind `ESTIMATE` (`targets()`, the run summary, `evaluate_project`): a target on one beam states the wavelength, angle or
intercepted rays that beam met the element at, not another beam's. A plane a finding names (`affected_observation_ids`) reads `OUT_OF_MODEL_SCOPE`, and
its `total_power_W`, which reads every beam on the plane, carries the finding's checks of the light it reaches there in a `value_basis` of kind `ESTIMATE`; so
does the power of each arrival the finding lists and of each line (`power_by_wavelength`) whose light it reaches (`monitors()`, `monitor()`, the `monitors`
and `arrivals` tables and exports, the run summary, `evaluate_project`). An arrival the finding lists carries its checks on every number whose kind
of quantity the finding touches (the kinds of the metrics that read that number; below): its Gaussian state under `"*"`
(`beam.gaussian.value_basis`), its position and direction on the plane and in the design frame, its path lengths, angle of incidence, spot,
polarization state and margins (`monitor()` keys such as `uv_mm`, `pupil_rms_radius_mm`, `polarization_state.s1`; the `arrivals` table and its
exports by column, `beam_radius_x_mm`, `u_mm`, `position_x_mm`; the run summary and `evaluate_project` on `position_mm` and the Gaussian
radii). The coherent sums on the plane (`superpositions`: `monitor()`, `superpositions()`, the `superpositions` table and its exports,
`evaluate_project`) whose beams a finding reaches carry its checks on their powers, and a two-beam interference record (`monitor()`'s
`interference`) on the numbers whose kind the finding touches. A detector on the plane (`detection()`, `monitor()`'s `detection`, the
`detection` query and CLI) carries them in `value_basis["value"]` of each quantity that reads light a finding reaches (detected power,
photocurrent, shot and total noise, SNR, NEP, the lock-in's signal, noise and SNR), not on its bandwidths or dark noise; its `power_state`
stays the plane's. A beam a finding reaches keeps its handoff (`handoff()`, the `handoff` query, export and CLI): its numbers and
`tem00_field_applicable` stay, `validity.model_scope` lists the finding (code, element, the kinds it touches, its checks of that beam's
light), the statement names it, and `value_basis` carries the checks on the power, the chief ray, the Jones vector, the phase and optical
path length and, under `gaussian.value_basis["*"]`, the Gaussian state, each where the finding touches its kind. A field built from such a
handoff beam (`focal_field()`: its quantities `OUT_OF_MODEL_SCOPE` with the checks; `pupil_field()`), an irradiance map of its light
(`irradiance_map()`) and a camera reading of that map (`detector_response()`, every number read from the light) list the finding under
`model_scope` and carry its checks in `value_basis`; an in-band beam on the same plane keeps its own. A polarization mixture of a route whose components a finding reaches (`monitor()`'s `polarization_mixtures`) keeps its
Stokes values, degree of polarization and `quantity_state` and carries the checks on each of them. A sampled line the finding lists (`affected_sampled_lines`) carries its checks on its row's power and collection
fractions, not on `emitted_W` (`monitor()`, `monitors()`, `sampled_light()`, the `sampled-light` table and its exports, the run summary,
`evaluate_project`), and on its crossings' powers (the `sampled-crossings` rows); the plane's line of that wavelength carries them too. The
other arrivals, lines, sampled lines, sums and detectors on the plane keep their own entries: a second emitter of the same wavelength whose
light misses the element, or a 532 nm Gaussian line beside 1200 nm sampled light, keeps its own. A run of an earlier Engine, whose findings
list no sampled line, reaches a line only where sampled light of that wavelength is on the plane and a wavelength check of the finding names
that wavelength.
The kinds are `GEOMETRY`, `POWER` and `POLARIZATION`: a coating or a polarization element used outside the wavelengths or angles its data are
published for, and a reflection from behind an absorbing coating, touch `POWER` and `POLARIZATION`; every other finding touches all three. A target on a
position, a direction, a beam radius or a path length behind such a coefficient finding, and a target on a declared limit behind any finding, keeps the
Engine's assessment. `capability_descriptor("model_scope")["quantity_kinds"]` lists the kinds per finding code and per metric;
`targets()` gives a result for every target the evaluation was asked to assess (one a capability outside the evaluation would decide is `UNKNOWN`, issue code
`CAPABILITY_NOT_COMPUTED`, its message naming the capability to add), and `targets()["not_evaluated"]` lists the declared targets the caller did not select (reason
`NOT_REQUESTED`). `evaluation.requirements()` (the lead `requirements` of the `project evaluate` and
`json_api.evaluate_project` reply) names every declared target once: the assessed ones under `met`, `unmet`,
`indeterminate` and `not_applicable`, the others under `not_assessed` with the reason, so a target that was not assessed
never reads as met; `evaluation.status` is `NOT_APPLICABLE` rather than `PASS` when the Project declares targets and the
evaluation assessed none. A complete evaluation's results equal those of
its run directory: every query, table and export (the same rows in the same order), and the `status` block of
`describe()`; only `describe()["integrity"]` and `describe()["files"]` describe the directory itself. The analyses that trace a run's light again (`imaging`, `ghosts`, `field`) read the evaluation's
request, beam paths and plane arrivals only: on a `readings` evaluation they give the document they give on the complete run
(only `identity.physical.run_result_hash` names the other result), so a follow-up analysis needs no complete evaluation.

## Command line

```text
os-optics results describe RUN
os-optics results monitors RUN
os-optics results monitor RUN ID
os-optics results paths RUN [--path P]
os-optics results power-budget RUN
os-optics results rays RUN [--monitor ID] [--path P] [--crossing-index N]
os-optics results wavefront RUN --monitor ID --path P [--crossing-index N] [--reference plane|sphere] [--center x,y,z|gaussian_waist|ray_focus]
os-optics results irradiance RUN --monitor ID [--wavelength-nm W] [--half-extent-mm H] [--samples N]
os-optics results handoff RUN --monitor ID [--path P] [--crossing-index N] [--element INSTANCE_ID]
os-optics results exposures RUN [--element ID]
os-optics results grating-orders RUN [--element ID]
os-optics results nonlinear RUN [--element ID]
os-optics results metrics [--name NAME]
os-optics results columns KIND
os-optics export KIND RUN --format csv|jsonl|json|npz [--out FILE] [--monitor ID] [--path P] [...]
```

Output is JSON. Exit codes: 0 answered, 2 a wavefront / irradiance the run cannot support (JSON with `status`
`UNAVAILABLE` and a `reason`), 64 usage error, unreadable run directory, unknown monitor or path.

## Export kinds and formats

Kinds: `monitors`, `arrivals`, `rays`, `paths`, `segments`, `power-budget`, `superpositions`, `wavefront`, `irradiance`,
`sampled-light`, `sampled-crossings` (the rays of extended emitters), `metrics`, `handoff`, `grating-orders` (one row per order of a pass through a grating), `nonlinear` (one row per beam's pass into a crystal that generates its second harmonic: harmonic power, efficiency, `dk L / 2`, walk-off, the model's checks, acceptance bandwidths), `exposures` (one row per beam meeting an element surface; column `light` says `traced` or `untraced`)
and `power-ratings` (one row per surface and rating, with `exposure_lower_bound` / `exposure_upper_bound` when the exposure
is not known exactly, the rating's `applicability`, its `conditions` as `condition=STATE` with the effect or reason
(`beam_diameter=DIFFERS(AT_MOST)`, `pulse_count=NOT_VERIFIABLE(DESIGN_INPUT_MISSING)`), the `unverified_conditions`, the
rating's `basis`, the test conditions it states (`rated_conditions`: `pulse_count=100 pulses;repetition_rate=10 Hz`), the
light's value for each (`design_values`) and the thresholds its value establishes (`rated_threshold_low` /
`rated_threshold_high`); both take `--element`). `results columns KIND` (`os_optics.results.columns(kind)`) lists every column with its unit and meaning;
the column order is fixed.

* **csv**: UTF-8, a header row, `\n` line ends. Units are part of the column names (`beam_radius_x_mm`, `power_W`,
  `aoi_deg`, `opd_waves`, `irradiance_coherent_W_per_mm2`); names without a unit suffix are dimensionless, identifiers,
  flags or text. Numbers are written in their shortest round-trip form, booleans as `true` / `false`, an undefined
  value as an empty field, a list (lineage, children, path ids) joined by `;`.
* **jsonl**: one JSON object per row, keys in column order.
* **json**: `{"schema": "opensci.results_export", "schema_version": "1.0", "kind", "run", "filters", "columns", "rows",
  "summary"}`; `summary` holds the RMS / PV and reference of a wavefront, the grid shape of an irradiance map, and for a
  handoff the plane, the element outline and the field-construction convention.
* **npz**: for the dense irradiance grid only (`u_mm`, `v_mm`, `irradiance_incoherent_W_per_mm2`,
  `irradiance_coherent_W_per_mm2`; rows are v, columns u), uncompressed, with fixed timestamps (the same run gives the
  same bytes). Rays, arrivals and paths are tables of at most a few thousand rows and are exported as text.

No exported value is NaN or infinite: a quantity the model does not define is empty / `null`, with its status or reason.

## Conventions

* **Frames.** Positions and directions are in the right-handed design frame (mm, +Z up). Each observation plane has an
  origin and unit axes `u`, `v`, `n = u x v` (`ObservationResult.plane`); `u_mm = (p - origin) . u`, `v_mm = (p - origin)
  . v`; `direction_u/v/n` are direction cosines on those axes (`direction_n < 0`: travelling against the normal).
  `pose_frame` `DEVICE_LOCAL` means the Project gave the pose relative to `device_id`; the frame shown is already resolved.
* **Absorbing extent.** `NONE`: a transparent monitor; `ACTIVE_AREA`: absorbs exactly the beams that count as arriving
  (chief ray inside the active area, or at least `capture_min_power_fraction` of the Gaussian power inside it), the others
  pass on and are listed in `outside_active_area`, except a beam whose chief ray would cross the plane beyond the scene's
  reach along it (`max(laboratory_reach_mm, support)`): that crossing is off the area, the beam is not listed, and
  `nearest_miss_mm` comes from the listed crossings only; `UNBOUNDED_PLANE`: absorbing without an aperture. A plane has no front:
  it observes (and absorbs) beams from either side, and its acceptance cone is two-sided (AOI = `arccos |d . n|`).
* **Crossings.** `crossing_index` is 0 for a path's first crossing of a plane, 1 for the second (folded / double pass); a
  branch counts its own crossings from 0.
* **A plane through a surface's hit point.** A plane within `ray_min_advance_mm` (1e-7 mm) of a physical surface's chief-ray
  hit point (a monitor on a mirror face or at a lens vertex) records the power arriving at that surface before the
  interaction. Each such arrival carries `surface_coincidence`: `surface_id`; `role` `ARRIVING`, or `RETURNING` for the beam
  that leaves the surface back across the plane (on the same path at the next crossing index, or on a splitter's child path
  at index 0); `surface_event`. `surface_event` states what the chief ray does at that hit: `REFLECTIVE` where it leaves on the side it came from (a mirror, total internal reflection, a splitter) or a coated refracting face passes less than half of it; `ABSORBING` where an absorber or an absorbing plane stops it or a mirror's back side blocks it; else `TRANSMISSIVE`. Arriving and returning are two arrivals with their
  own keys. A leaving beam that does not change sign along the normal is not sampled and adds nothing to the plane's reading:
  the arrival keeps its value and `quantity_state` `KNOWN`. An absorbing plane stops the beam at the arrival: `ABSORBING`, no
  return. A power target that selects such an arrival carries `SURFACE_COINCIDENT_ARRIVAL` and is assessed on the arrival's
  value; the plane carries one `SURFACE_COINCIDENT_ARRIVAL` issue per surface, status `NOT_APPLICABLE`.
  `crossing_power_upper_bound_W` is an auxiliary value on an `ARRIVING` arrival at a flat surface tilted to the plane, with a
  valid Gaussian beam: the most power that crosses the literal plane before the surface (about 0.5 P for a 45 deg fold and a
  beam-normal plane, plus what passes beside the surface). No target reads it; it is absent elsewhere, and where within `K` beam radii a line may start beyond
  the plane or the surface on that leg. A plane through a hit point whose active area lies beside it counts no arrival there and keeps the light of the leg leaving that surface: it reads `BOUNDED` from 0 to that light's bound when that light may reach the area under a supported bound, `UNKNOWN` without one. Only a plane that counted an arrival there leaves that leg out of its reading. 1 nm from a surface a
  plane is outside this window and in the near-surface refusal band.
* **Gaussian quantities** (`*_x`, `*_y`) refer to the beam's own axes `gaussian_e1`, `gaussian_e2`. `waist_distance > 0`:
  the waist lies upstream of the plane; waist position = hit point − `waist_distance` × direction.
* **Polarization.** Jones vector `(Ex, Ey)` in the transported basis `(e1, e2)`, `e1 × e2 = d`, field
  `E ∝ exp[i(k·r − ωt)]`; Stokes `s1..s3` normalized by `S0`; `s3 = −1` right circular (clockwise viewed facing the
  source). A traced beam is fully polarized (degree of polarization 1). A partially polarized source (`PARTIAL`,
  `UNPOLARIZED`) sends each line as two mutually incoherent orthogonal components, `<source>#<line>` (the polarized state,
  `(1 + P)/2` of the power) and `<source>#<line>:p1` (`(1 − P)/2`); each arrival carries `polarization_component` (0 / 1).
  `ObservationResult.polarization_mixtures` gives per route (one line, the same interactions and crossing) the summed
  `stokes_W` `(S0, S1, S2, S3)` in W, `degree_of_polarization` and the polarized part's ellipse (`polarization_state`,
  absent for unpolarized light), with a `quantity_state`. The metrics `stokes_s1..s3`, `polarization_ellipticity_deg`,
  `polarization_azimuth_deg` and `degree_of_polarization` on such a beam read its route's sum; power metrics add both
  components; two-beam metrics are formed within one component (`PARTIAL_POLARIZATION_COMPONENTS` for a pair of one beam of
  each). The monitor document lists `polarization_mixtures` when such light arrives.
* **Time-domain delays.** `group_delay_ps`: the arrival time of a beam's envelope from its source's origin plane, group
  optical path `Σ n_g L` / c (`group_path_length_mm`); `phase_delay_ps`: `Σ n L` / c; `time_delay_ps`: the relative delay
  `t_a − t_b` of a pair (`group_path_difference_mm` / c, positive when a arrives later), the delay of a delay line or a
  pump-probe pair. A pass whose group delay the design does not state is left out of the number: the target reports
  the delay of the modelled path (for a pair, of the two modelled paths), `UNKNOWN` with `GROUP_DELAY_UNDEFINED`, its
  `value_basis["value"]` an `ESTIMATE` naming `UNSTATED_GROUP_DELAY_ZERO` at those passes, and is not graded; coatings
  whose data bound their group delay give an interval the target is graded on; a medium without a group index and two
  sources (no common time origin) leave no value.
  The pulse shape and its broadening by dispersion are not part of these values.
* **First-order transfer.** `fo_d<out>_d<in>` = the ray-transfer matrix `d(u, v, l_u, l_v) / d(x0, y0, a0, b0)` from the
  source plane (parallel displacements x0, y0 along the source frame, tilts a0, b0 in rad) to the crossing, from central
  differences of differential rays traced with the chief ray; image parity, rotation and transverse magnification derive
  from its `A` block.
* **Power.** An arrival's `power_W` is the incident beam's power at the plane. On an absorbing plane its `detected_power_W`
  is the power the plane detects from it (its share on the active area, the budget's `DETECTED`; 0 for a beam outside the
  acceptance cone, the budget's `OUTSIDE_ACCEPTANCE`), and the plane's `total_power_W`, `power_by_wavelength` and the
  `power_W` target metric sum the detected power. A transparent plane receives the whole power of a counted beam (its
  share on an active area is `active_area_power_fraction`) and nothing from a beam outside its acceptance cone
  (`within_acceptance` false); the beam passes on. The power budget lists every sink where power leaves
  the traced beams (detected, absorbed, escaped, clipped, aperture truncation, untraced reflection / transmission,
  coating / bulk / polarizer / element losses, a tabulated thin lens's unmodelled response `UNMODELLED_RESPONSE` (its
  `1 − transmittance`: light its response model does not deliver into the focused beam), scattered light the run does not
  trace (`SURFACE_SCATTER`, `BULK_SCATTER`), out-of-model terminations, trace limits; the rays of a surface's traced scatter
  book their own fates in place of `SURFACE_SCATTER`); power counted at a monitor is
  `OBSERVED`, not a sink. `residual = input − Σ sinks`.
* **Known zero, empty selection, incomplete trace.** `power_by_wavelength` lists every launched wavelength. `complete`
  (per wavelength) and `power_complete` (the plane) say whether all launched power was traced to a known end; if not,
  `untraced_power_W` is the power whose fate is not followed (the budget's `PATH_LIMIT`, `INTERACTION_LIMIT`,
  `SOURCE_NOT_TRACED`, `TERMINATED_OUT_OF_MODEL`, `UNMODELLED_RESPONSE`, `SURFACE_SCATTER`, `BULK_SCATTER`, or a non-negligible part of the lines stopped at
  the power floor) and
  the power counts the traced light only. The light the trace does not follow that may reach the plane counts too
  (PHYSICS_SCOPE P10: an untraced Fresnel reflection or coating transmission, the
  part of a beam booked `CLIPPED` that passes the opening, light passing beside an element, a beam's tail on an active
  area it is not counted on), each followed through the optics by a probe trace (with its own ghosts, to every order above
  `negligible_power_fraction` x the line power): `untraced_power_W` then includes the
  power of that light where it arrives at the plane, on a plane the line's traced beams reach too (a lens ghost beside
  the main beam), and `power_upper_bound_W` is the traced light plus that light and the light left out as negligible --
  the upper end of a `power_W` target's interval on a plane whose traced value is the model's (a sampled value takes no
  bound: below). Where nothing of a wavelength (of the plane) arrives, the power is
  `0.0` when that is a
  known zero and `null` when it is not known (`complete` / `power_complete` false), so a reader of the value alone never
  takes an unknown for 0 W (in tables: an empty field). A target that no traced beam matches is:
  * a *known zero* when its source lines were traced completely and none of their untraced light reaches the plane above
    the line's negligible level (`negligible_power_fraction` of its input):
    `power_W` and `coherent_power_W` are `0.0` (code `NO_TRACED_ARRIVAL`; a maximum passes, a minimum fails with that
    value); a metric that needs a beam (radius, position, ...) fails with `OBSERVATION_NOT_REACHED`. The light left out as
    negligible may still reach the plane: while the most it brings there (its crossings and returns included) stays within
    the lines' negligible level, it is the stated truncation of the 0 W and changes no state. The
    target keeps `0.0`, adds the code `FAINT_LIGHT_BOUNDED`, and is graded on `0.0` and that bound together: a maximum
    below the bound, or a minimum the bound could reach, is `UNKNOWN` (`INDETERMINATE`); a limit outside the interval
    they give is decided as above. Its `value_basis["value"]` entry (an omission
    `BELOW_NEGLIGIBLE_LEVEL`, setting `negligible_power_fraction`) holds the most that light brings: its power on an
    absorbing plane, which absorbs each watt once; on a transparent one that power times its open paths' crossings of the
    plane, at least twice -- crossed once either way -- and once per straight leg after each number of reflections,
    below; more where two surfaces facing each other hold it and send it across the plane: the square of their factor;
    for `coherent_power_W` the square r^2 of its field, r the summed field amplitude of its pieces, at least their power
    (the plane's own `power_by_wavelength` stays `0.0`). Where that most exceeds the level, or nothing bounds it, its
    possible return is not negligible: the target has no value and is judged on the interval from 0 to that bound
    (`FAINT_LIGHT_BOUNDED`, or the cause that leaves it unbounded). The plane is `BOUNDED` from 0 (`UNKNOWN` without a
    bound) where the power that light brings exceeds the level. Its field r^2 may exceed the level while that power
    stays within it: then the plane stays `KNOWN_ZERO` with `power_complete` true and its `power_W` target keeps `0.0`,
    while a `coherent_power_W` target has no value and no entry -- a maximum at or above r^2 is `MET`, a minimum above
    it `UNMET`, any other limit `INDETERMINATE`;
  * `UNKNOWN` with `TARGET_SELECTION_EMPTY` when a `source_id`, `wavelength_nm`, `path_id` or a later `crossing_index`
    (1 or more) selects nothing in the run (a crossing is looked for among the selected sources' and wavelength's
    beams); the message names what the run has;
  * `UNKNOWN` with the cause when part of its source lines was not followed: `MAX_PATHS` (trace budget),
    `MAX_INTERACTIONS` (per-path interaction limit), `OUT_OF_MODEL_SCOPE` (a path ended out of model scope or a line was
    not launched), `UNTRACED_LIGHT_BOUNDED` (an untraced reflection / transmission or light beside an element, followed
    by the probes, may reach the plane; the same code when other light was not followed), `CLIPPED`
    (the part of a clipped beam inside the opening may reach it), `INCOMPLETE_ENVELOPE` (a beam's tail on the active
    area), `ZERO_POWER` (light at the power floor, e.g. a source at or below `power_epsilon_W`). Where no matching beam
    arrives, that light is detected and not traced: a `power_W` or `coherent_power_W` target has no value and is graded on
    the interval from 0 to the bound of the light the probes followed to the plane (a maximum at or above the bound is
    `MET`, a minimum above it `UNMET`, else `INDETERMINATE`; `INDETERMINATE` too where part of that light has an unknown
    fate or nothing bounds it), with these codes and the bound in its message. Where some matching
    beams arrive, a summed power is bounded: the value is the traced light's, and the plane's power lies between it and
    the traced light plus `untraced_power_W` on an absorbing plane -- on a transparent one too when that power is light
    the probe traces followed to every crossing of the plane, with no upper end when part of it has an unknown fate; for
    `coherent_power_W` between `(√P − Σ√Uᵢ)²` (0 when the sum exceeds `√P`) and `(√P + Σ√Uᵢ)²` over the separate
    contributions the probes followed to the plane (mutually coherent fields add their amplitudes), and from 0 to
    unbounded when part of the untraced power has an unknown fate (it may arrive in any number of beams). Every run with an
    observation plane follows every untraced branch, and the untraced light of a line that reaches the plane counts also
    when traced beams of the same line arrive there (a weak traced beam does not stand for a strong coating transmission
    folded onto the same plane; the minimum side reads the traced light). A target where some matching beams arrive is
    `PASS` when every value of that interval meets it, `FAIL` when none does, otherwise `UNKNOWN`, always with the cause
    code and the interval in the message.

  An issue that leaves targets `UNKNOWN` or `OUT_OF_MODEL_SCOPE` names them in `affected_targets` (the run's trace
  budget, the per-path interaction limit, a path ended out of model scope, a plane no traced beam reaches while light was
  not followed, the beam-pair limit).

  A plane where the power of a beam, or of light a surface scattered out of one, is out of model scope
  (`arrivals[].power_out_of_scope_at`, `sampled_light[].power_out_of_scope_at`: behind a crystal whose second-harmonic
  generation is out of model scope there, a variable attenuator or transmission map that varies across the beam, the side of a
  gradient-index rod, an element part of the beam meets beside its chief ray without a bound of that share or that turns
  that share rather than stopping it) has `quantity_state`
  `OUT_OF_MODEL_SCOPE`: that beam's power is its chief ray's, not the light's. Nothing computed from that power is stated
  for it: an irradiance map leaves that beam out and names it (`excluded_beams`, reason `POWER_OUT_OF_MODEL_SCOPE`, with the
  surfaces; the map of the other beams stays, `map_complete` false, `untraced_power_W` null), and where it is the only light
  of the map's scope there is no map (`POWER_OUT_OF_MODEL_SCOPE`, with `power_out_of_scope_at` and the beams' `path_ids`; the
  detector response that reads it: `IRRADIANCE_UNAVAILABLE`, `upstream_reason` `POWER_OUT_OF_MODEL_SCOPE`), its handoff power (`power_W` null, `validity.beam_parameters`
  `POWER_OUT_OF_MODEL_SCOPE`) and every pupil or focal field built from it, the coherent sum of its wavelength
  (`superpositions`: `OUT_OF_MODEL_SCOPE`, no total) and a pair's intensity balance, fringe visibilities, irradiances and powers
  (`interference`: `OUT_OF_MODEL_SCOPE`; the crossing angle, fringe period, path difference and coherence stay), the ghost
  powers, ratios and bound of its line (`OUT_OF_MODEL_SCOPE`; where the ghosts land stays), the coherent field of a source
  whose axis is such a beam, exposures and power targets, the plane's own bounds (`power_upper_bound_W`, `power_lower_bound_W`,
  `absorbed_power_upper_bound_W` null; `message` names the surfaces), the Stokes parameters of a partially polarized route
  (`polarization_mixtures[]`: `OUT_OF_MODEL_SCOPE`, no `stokes_W`), the order powers of a grating or an acousto-optic deflector
  the beam meets behind those surfaces (`grating_orders[]`: `power_state` `OUT_OF_MODEL_SCOPE`, `power_W` and the bound null,
  `efficiency` and the direction stated), the second harmonic of a crystal it enters (`NonlinearConversionResult` `OUT_OF_MODEL_SCOPE`,
  `NONLINEAR_CONVERSION_OUT_OF_SCOPE`: no harmonic power or efficiency; the harmonic is still traced) and the power budget's
  observed power on that plane (`PowerBudgetLine.observed_W` leaves the plane out and lists it in `observed_out_of_scope_at`; the
  `power-budget` export's `OBSERVED` row for it has no power). The values that stay are marked: `power_by_wavelength[]`,
  `outside_active_area[]`, `sampled_sources[]` (light scattered out of such a beam) and the paths (`paths[].power_out_of_scope_at`:
  `final_power_W` and the segments' powers from there on are the chief ray's) carry `power_out_of_scope_at`; the run summary's
  arrivals and sampled lines and the 3-D view (footprint and path power null) likewise. The `arrivals`, `handoff`,
  `sampled-light`, `paths` and `grating-orders` exports list `power_out_of_scope_at`, the `monitors` export the `quantity_state`.

  A plane with no traced arrival is one of three things, and `quantity_state` says which:

  * `KNOWN_ZERO`, `total_power_W` 0.0: every launched watt of the run's lines ended somewhere known, and no untraced
    reflection, coating transmission, clipped beam, partial crossing or light left out as negligible reaches it above
    the line's negligible level (`negligible_power_fraction` of its input). A partial crossing at or below that level is left out, so a known zero means nothing above that level
    reaches the plane, with two stated limits of what is counted. On a plane with no active area it covers the reach of a leg that leaves the scene (below), and the message says so. Behind a hard edge that keeps the beam's Gaussian state (it takes at most `gaussian_clip_scope_fraction` of the power), aperture diffraction is not propagated, so a side plane reads a known zero: behind a 0.6 mm stop on a 0.5 mm beam, about 5e-8 W reaches a 20 mm disc 5 mm beside the beam 100 mm on. That diffraction is not in the result: to read a monitor there, compute the aperture's far field separately. A
    `power_W` target on it evaluates 0 W (`NO_TRACED_ARRIVAL`); a `coherent_power_W` target does where the square r^2 of
    the field the light left out as negligible may bring stays within the level, and otherwise has no value (a *known
    zero*, above).
  * `BOUNDED`, [0, `power_upper_bound_W`]: only light the probe traces followed reaches it, under a supported bound
    (an untraced reflection or coating transmission, light passing an aperture or an element its chief ray misses, a
    beam's tail, part of a beam crossing where its chief ray does not), named in `followed_light`: `total_power_W` null,
    `power_lower_bound_W` 0, status `PASS`, an `OBSERVATION_NOT_REACHED` notice. Or nothing followed reaches it and only
    light left out as negligible may bring more than the negligible level, under a finite bound
    (`FAINT_LIGHT_BOUNDED`): `total_power_W` 0.0 (the traced zero), `power_lower_bound_W` null, `followed_light` empty,
    `untraced_power_W` 0, status `UNKNOWN`. Its `power_W` and `coherent_power_W` targets are decided only in the direction
    the bound proves (for `coherent_power_W` the square of the summed field amplitudes): a maximum at or above the bound is
    `MET`, a minimum above it `UNMET`, any other limit `INDETERMINATE`, with the cause codes and no value; a beam metric
    or a pair metric there is `INDETERMINATE`.
  * `UNKNOWN`: light of unknown fate may reach it, or light the trace followed, or light left out as negligible, with no
    finite bound. This light is detected and not traced. `total_power_W` is null; `untraced_power_W` is reported beside
    it, and `power_upper_bound_W` wherever a bound is supported. A `power_W` or `coherent_power_W` target on it is
    `INDETERMINATE`, with the cause codes and no value. A beam metric (radius, capture fraction)
    and a pair metric on a plane no traced beam lights are `INDETERMINATE`, with any selector (`path_id`,
    `crossing_index`) that selects no arrival. Where only light left out as negligible may reach the plane and surfaces facing each other leave it no finite bound, the plane is `UNKNOWN` too: `total_power_W` and `power_upper_bound_W` null, `untraced_power_W` 0, targets `INDETERMINATE` with `FAINT_LIGHT_BOUNDED`. A selector that selects an arrival reads that arrival's value, with no untraced light added. The same holds for a target whose selected source lines have no traced arrival on a plane other lines light. To
    decide such a target: place the plane where the chief ray crosses it, or outside the beam's envelope, or let the trace
    follow the light named in the cause codes.

  A lit plane with a supported interval stays `BOUNDED`, and its targets are graded on the interval.

  Part of a beam can cross a plane its chief ray does not cross on that leg (`PARTIAL_CROSSING`). The bound of that part is
  the larger of the Gaussian share between the leg's start and end surfaces in the parallel-ray picture and the paraxial
  energy flow across the plane (PHYSICS_SCOPE P10); on a leg that leaves the scene only the energy flow exists. On a leg that leaves the scene, a plane without an active area counts the light that crosses it up to `max(1000 mm, support)` from the leg's start, `support` being the farthest along the ray of the design's physical surfaces, source points and active areas; no plane's origin enters it. The 1000 mm is a declared numerical choice, recorded with every run as `laboratory_reach_mm` in `manifest.numerical_tolerances` and not configurable. Part of a beam beyond that distance on a leaving leg is not counted on a plane with no active area; a chief ray that crosses the plane is an arrival wherever it crosses. To read it, state an active area or end the leg on a dump. A plane with an active area counts what crosses that area, wherever it lies; a chief ray does not cross such a plane beyond `max(1000 mm, support)` along it (that crossing is off the area), so that beam is not listed in `outside_active_area` and only its light on the area is counted. Reference: 1 mW,
  780 nm, 20 um waist, 5 mm disc 2 mm off axis at x = 90, parallel to the beam: flux by independent integration 2.71e-7 W,
  bound 2.73e-7 W, plane `BOUNDED`, a 1e-7 W maximum `INDETERMINATE`. A plane more than `K` beam radii from the chief ray over
  the stretch reads as dark, `K` about 3.3 at `negligible_power_fraction`.

  Behind a severe hard-edge truncation (`GAUSSIAN_SEVERE_TRUNCATION`) a traced beam has no Gaussian state, and its diffraction falls off too slowly for its rays to locate it. Any plane the line does not light with a point ahead of that leg's start, within that reach on a plane with no active area, reads `BOUNDED` from 0 to the beam's power: `total_power_W` null, `power_lower_bound_W` 0 and `power_upper_bound_W` the beam's power, `DIFFRACTED_BEAM` in the plane's `followed_light` and in the `OBSERVATION_NOT_REACHED` notice's `untraced_light`, status `PASS`, and `GAUSSIAN_SEVERE_TRUNCATION` among its target's codes. A `power_W` maximum at or above that bound is `MET` and a minimum above it `UNMET`; any other power limit, and any beam metric there, is `INDETERMINATE`. A plane the line does light keeps its reading, and the light is not taken out of the beam, on an absorbing plane too. To decide any other target there, open the aperture until the beam keeps its Gaussian state, or place the plane where the line lights it. A milder truncation keeps its Gaussian state, and aperture diffraction is not propagated. A beam without a Gaussian state for any other reason, and untraced light (a ghost a lens's rim truncates), is bounded by the whole beam within three envelopes of its live pupil rays; beyond that reach, or with no live pupil ray, nothing is recorded for that leg.

  Such a plane reports `OBSERVATION_NOT_REACHED`: `UNKNOWN` while launched power has an unknown
  fate or untraced light may reach it (context `untraced_power_W`, `untraced_causes`, `untraced_light`: the ledger
  sinks / kinds of light, and for untraced light whose fate a probe does not know why: `UNTRACED_LIGHT_OUT_OF_MODEL`,
  `UNTRACED_LIGHT_BUDGET`, `UNTRACED_LIGHT_INTERACTION_LIMIT`, `UNTRACED_LIGHT_POWER_FLOOR`; `PARTIAL_CROSSING`: part of a
  traced beam reaches the plane where its chief ray does not cross it, bounded by that part's Gaussian share of the
  beam's power; then also `partial_before_surface_id` / `partial_before_surface_ids`, the surface id(s) that part reaches
  the plane before, `observation/<id>` for an observation plane, which the message names "observation plane <id>").
  Where only light the probes followed reaches it under a supported bound, the plane is `BOUNDED` between 0 W and
  `power_upper_bound_W` (`followed_light` lists that light) and the issue is a notice (`NOT_APPLICABLE`); where no ray of a
  sampled line is counted on it, the plane reads that line's sampled zero (`KNOWN`, `power_sampling_resolution_W`) and the
  issue is a notice; otherwise the plane is dark (0 W), its status `PASS` and the issue a notice.

  A plane's `status`, `message`, `quantity_state` and its `OBSERVATION_NOT_REACHED` / `ZERO_POWER` issue are functions of the
  physics: adding, removing or changing a target leaves them identical. The target result carries what the target decides.
  A plane's `status` and its `quantity_state` answer two questions. `status` is the diagnostic severity of the beams
  arriving at the plane (acceptance, capture, beam-model validity) and, on a plane no traced beam reaches, of whether
  light may arrive. `quantity_state` says whether the plane's power is a known quantity, and it leads for every question
  about the power: a lit plane may read `PASS` beside an `UNKNOWN` or `BOUNDED` power (its beams arrive in model scope;
  its power is incomplete) and `OUT_OF_MODEL_SCOPE` beside a `KNOWN` one (its power is known; its beam radius is not).
  The plane's `status` is one term of the aggregate; its `quantity_state` is not. Read `quantity_state` and `value_basis`
  for "is this power usable", a target's `assessment` for "is the target met", `outcome.execution` for "did it run", and
  a plane's `status` only to sort and flag.
  The run's aggregate `status` is a summary and may read `WARNING` or `UNKNOWN` beside a met target. A trace finding's
  status says what it is: a physical outcome the trace completes is `WARNING` (`CLIPPED`); a data or model limit for the
  quantities behind it is `OUT_OF_MODEL_SCOPE` (`COATING_OUT_OF_RANGE`, `MATERIAL_OUT_OF_RANGE`, `GRAZING_INCIDENCE` at its
  limit, `PARALLEL_RAY`, a pair's `MEDIUM_MISMATCH` or a beam parallel to its plane); only a scene that contradicts itself
  is `FAIL` (the trace's `MEDIUM_MISMATCH`, `DEGENERATE_SURFACE`, `MATERIAL_INVALID`, `NUMERICAL_FAILURE`). The issue that says a power target's limit lies inside a lit plane's interval (`INCOMPLETE_TRACE`) sits at the target: `location` `target:<id>`, the plane in `numerical_context.observation_id`. A plane's own issues and the compact record's per-plane `issue_codes` do not depend on targets; only `affected_targets` names targets on a plane's issue.
* **Capture and acceptance are measurements.** A beam that overfills an active area is `INCOMPLETE_ENVELOPE`
  (a notice, `NOT_APPLICABLE`, with `active_area_power_fraction` and `capture_margin_mm`); a beam outside the acceptance
  cone is `MONITOR_MISSED` (a notice; it is not detected). Neither raises the plane's or the run's status. A requirement is a target: `active_area_power_fraction` or
  `capture_margin_mm` minimum, `acceptance_margin_deg` minimum.
* **Central obstruction.** A beam whose chief ray falls on a central obstruction is the annulus around it
  (`chief_ray_obstructed`): `power_W` is the annulus's exact power, the pupil rays are its geometric rays, the chief ray is
  a reference axis (`phase_rad` is null) and the Gaussian state is out of model scope (`GAUSSIAN_CENTRAL_OBSTRUCTION`).
  It arrives only where it provably passes every aperture on the way whole (the envelope of the unobstructed beam, or the
  image of the opening that cut it) and (with an active area) falls on the area whole or but for less than
  `capture_min_power_fraction` of it (then with an `INCOMPLETE_ENVELOPE` WARNING stating the bound, the arrival's
  `detected_power_min_W` as the lower end, and a power target on it judged on that interval: `INCOMPLETE_ENVELOPE` in its
  codes, `UNKNOWN` when its limit lies inside the interval); an aperture or area
  that takes part of it leaves the annulus's power not computed from there on (issue `ANNULAR_BEAM_NOT_FOLLOWED`): the
  rays that pass go on, and the planes behind record the arrival with `beam.power_W` null and `power_upper_bound_W`
  (the annulus's power, booked once to `TERMINATED_OUT_OF_MODEL`); their power is not stated (PHYSICS_SCOPE P1). Control readings follow the annulus by these bounds alone; a sensor whose light is not followed has a reading that
  is not known, and the control analysis is then `UNKNOWN` (`OUT_OF_MODEL_SCOPE`), never a lost beam. A beam whose Gaussian state is invalid (e.g.
  `GAUSSIAN_SEVERE_TRUNCATION`) and whose pupil rays a later aperture cuts goes on with the share of its pupil samples
  that pass (`GEOMETRIC_POWER_BASIS` WARNING; the arrivals behind it carry `power_basis` `GEOMETRIC_PUPIL_SAMPLES` and
  `power_sampling_resolution_W`, and a power target whose limit lies within that resolution of the value is `UNKNOWN`,
  assessment `INDETERMINATE`, with `GEOMETRIC_POWER_BASIS` in its codes); behind an aperture that cuts every pupil sample
  while the chief ray passes its power is not computed, and likewise behind an edge within the beam's reach where the rays do
  not follow its diffraction, or on a detector's active area on which no pupil sample falls: the beam goes on as geometry,
  each arrival behind states `beam.power_W` null with `power_upper_bound_W` (what reached that surface), the path ends
  where the geometry does with `final_power_W` null (PHYSICS_SCOPE). Behind an edge where the rays do not follow the
  diffraction, the arrivals' `pupil_rms_radius_mm` is the rays' and carries `value_basis.pupil_rms_radius_mm`: kind
  `ESTIMATE`, the check `diffraction_spread_mm` (the spread beyond the rays, the most they follow there, the surface).
  The same holds behind a coating met outside its data by the chief ray (`COATING_OUT_OF_RANGE`), and behind a retarder,
  Faraday rotator or attenuator whose response at the line is not stated (`MODEL_WAVELENGTH_OUTSIDE_BAND`,
  `ND_UNCOATED_REGION`): each issue carries `power_upper_bound_W`, and a splitter's two branches both go on. On an absorbing
  plane's active area such a beam's `detected_power_W` is the share of its samples on the area, within
  `power_sampling_resolution_W` (the resolution of the received power). The plane states the sum of its arrivals'
  resolutions as `power_sampling_resolution_W`: its `total_power_W` is a `KNOWN` sampled value, and `power_upper_bound_W`
  takes no bound from it (null on a transparent plane). The resolution is convergence evidence, not a proven bound: a
  `power_W` target is decided where its limit lies outside the value plus or minus the resolution (as an analysis
  quantity's `decides`), and its message says the interval it was judged on is that resolution, not a bound.
* **Promoted reference ray.** A beam whose chief ray falls on the rim beyond a clear aperture while some of its pupil rays
  pass goes on as those rays (the `CLIPPED` issue states how many pass, `reference_pupil_ray` and the power that passes);
  the one nearest their centroid is its reference ray from there. The arrivals behind it list that surface in
  `reference_ray_promoted_at`: `uv_mm`, `aoi_deg`, `direction_uvn`, `beam.position_mm`, `beam.direction` and
  `beam.optical_path_length_mm` are that ray's, `first_order` is absent (`first_order_issue` says why) and the Gaussian
  state is not valid. A target on the chief-ray hit point, direction cosines, angle of incidence or acceptance margin
  there is out of model scope (`CLIPPED`): that ray is one sample of the light that passed, so the target reports
  that ray's value (`value_basis["value"]`: `ESTIMATE`, `SINGLE_RAY_FOR_BEAM` at the promoting surface) and is not
  graded; the spot centroid gives the beam's position. `beam.power_W` is the Gaussian power inside the opening where the beam was Gaussian there, else on
  the geometric pupil-sample basis above. A ghost path on this basis lists the surface in its row's
  `reference_promoted_at` (`results ghosts`).
* **Launched power bounds an absorbing plane.** Each watt is absorbed once: an absorbing plane's `power_upper_bound_W` and
  the upper end of a `power_W` target's interval on it are at most the power its lines launch.
* **Share taken before a plane.** Where a surface takes part of a beam before a plane the beam's chief ray crosses (the
  physical surface it meets next, an absorbing plane, or the reflecting surface the leg began at for the part it sends
  beyond the plane), and that share has a supported bound (the larger of the parallel-ray and the paraxial energy-flow
  pictures), the arrival lists the surfaces in `power_taken_before_at` and the bound in `power_taken_before_upper_W`: the
  crossing power lies between `beam.power_W` less it and `beam.power_W` (on an absorbing plane: `detected_power_W` less it
  and `detected_power_W`). Where the next surface stops what it meets from the beam's side (the back of an uncoated mirror,
  an absorber) and no other surface counted only turns light, `power_taken_before_lower_W` states the share whose
  energy-flow lines certainly meet it first before the plane: the crossing power of a transparent plane is at most
  `beam.power_W` less it (an absorbing plane's reading is not lowered by it). A beam outside the plane's acceptance cone adds
  no share. The plane is `BOUNDED` with `power_lower_bound_W`, a `power_W` target is judged on that interval
  (`POWER_TAKEN_BEFORE_PLANE`), and the other power-derived target metrics are out of model scope. The beam's power and profile there
  are the whole beam's and where the light was taken is not known, so an irradiance map leaves that beam out and names it
  (`excluded_beams`, reason `POWER_TAKEN_BEFORE_PLANE`, with its `power_lower_bound_W` and `power_upper_bound_W`; the map of the other
  beams stays, and that power counts as light the map does not hold); where it is the only light of the map's scope the map (and a
  detector response built from it) is `UNAVAILABLE` with reason `POWER_TAKEN_BEFORE_PLANE` (`power_taken_before_at`, `path_ids`; the
  irradiance export refuses with that code, CLI exit 2). The handoff states `power_W` null with `validity.beam_parameters` `POWER_TAKEN_BEFORE_PLANE` and
  `validity.power_taken_before_at` (export column `power_taken_before_at`), and a pupil or focal field built from that beam is refused
  likewise; the beam's geometry, Gaussian parameters and polarization stay. A surface beside the chief
  ray that pupil rays meet first, or that the beam's tail may meet while no pupil ray does (within `sqrt(2) K` beam radii of
  the leg), is bounded the same way. A share taken on an earlier leg stays with the beam: a surface beside the chief ray
  that stops the light it meets (an absorber that does not scatter; met by pupil rays or by the tail) or an absorbing
  plane part of the beam crosses is listed in `power_taken_before_at`
  of every plane behind it, on every branch, and its bound is added to `power_taken_before_upper_W` in W, unscaled (every
  later element is passive, so the light missing behind it is at most the light taken). The surface carries a
  `POWER_TAKEN_BEFORE_PLANE` warning with the bound (`tail_power_upper_W`, or `taken_power_upper_W` with the pupil rays
  it meets), and the exposure of a surface behind it states `incident_power_W` with a `value_basis` entry
  (`UPPER_BOUND`, `BEAM_PART_ELSEWHERE`, `upper`) and no irradiance. A surface beside the chief ray that turns the light
  it meets (a mirror, a lens or plate face, a splitter, a scattering absorber) may send that share to a plane behind, so
  the power stated there would be no upper end: like a share without a bound it keeps the planes behind out of model
  scope. A share without a bound (pupil rays meet a face of the body the beam travels in first;
  without a valid Gaussian state, the surface the leg began at may reach beyond the plane or the surface the chief ray meets
  next may reach before it) keeps the plane's power out of model scope (`power_out_of_scope_at`): the plane's status is `OUT_OF_MODEL_SCOPE` with an
  `OUT_OF_MODEL_SCOPE` issue at the plane naming those surfaces (`surface_ids`). A beam whose chief ray passes a hole
  while the rest of the beam meets the element around it reads the beam within the share the element's faces may take
  (a 0.5 mm beam through a 0.6 mm hole 0.2 mm off its axis: `[0, P]` against 0.991 P that crosses).
* **Optical path.** `opl_mm` = Σ n × geometric length from the source (a branch includes its ancestors). Every pupil ray
  starts on the source plane with 0. `phase_rad` is a plane-wave phase without the Gaussian Gouy phase. Where
  `ray_opl_defined` is false (behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating), `phase_rad`,
  `optical_path_length_mm` and `opl_mm` are `null` and a target on them has no value (`UNKNOWN`, `PHASE_UNDEFINED`).
  `optical_path_difference_mm` has a value when the two beams make such passes alike, which cancel; a segment's own
  `optical_path_length_mm` is n × its length.
* **Metric signs.** `results metrics` gives each metric's sign convention and operand order, e.g.
  `optical_path_difference_mm = OPL(a) − OPL(b)` with `a` the beam selected by `parameters.path_id_a`.

* **Metric names and result fields.** A target metric and the result field that carries its value can differ in name:
  `divergence_half_angle_x_rad` is the arrival column `far_field_half_angle_x_rad`. Each descriptor's `result_fields`
  lists them as `<export kind>.<column>` (e.g. `arrivals.far_field_half_angle_x_rad`) and `monitor:<field>` (a field of
  `monitor(id)`, e.g. `monitor:arrivals[].beam.gaussian.far_field_half_angle_x_rad`; `[]` is every entry of a list, `[0]`
  its first entry and `[<name>]` the entry of that `name`, e.g. `monitor:detection.quantities[lock_in_snr].value`, present
  on a plane whose detection states the detector's electrical response); it is empty for a metric computed from several
  fields (`beam_radius_mm` = √(w_x w_y)).

## Value basis: what a reported number is

A number is reported with a `value_basis` entry when it is not simply the model's value, or when a caller setting other
than the Engine default produced it. The entry sits on the record that holds the number, in `value_basis`, keyed by the
name of the field it qualifies (`"*"`: every number of the record, as on a Gaussian beam state whose numbers stand or
fall together; `beam.power_W` on an arrival). It is omitted when the record has none. An analysis quantity (imaging,
ghosts, field, cavity, configurations) and a quantity of `pupil_field` / `detector_response` carry one entry,
`value_basis`.

A Gaussian beam state beyond an accuracy limit of the first-order model (`beam.gaussian.valid` false with reason
`GAUSSIAN_HIGH_NA`, `GAUSSIAN_SURFACE_TOO_STRONG`, `GAUSSIAN_LARGE_AOI` or `GAUSSIAN_SEVERE_TRUNCATION`) publishes no
number (q, radii, waists, Rayleigh ranges, divergences, curvature are `null`) and states the check that failed in
`beam.gaussian.validity`: `parameter`, `observed`, `maximum`, `setting` (the `numerical_settings` field that holds
the limit), `at` (where it was first exceeded) and `code`. It stays so for the rest of the path. A state without a
first-order description publishes `valid`, `status` and `reason` only. A target on a Gaussian metric of such a beam
has `value` null, status `OUT_OF_MODEL_SCOPE`, assessment `INDETERMINATE` and the same check in its `validity`. With
the limit raised by the caller so that the state is valid, the numbers carry a `"*"` entry of kind `MODEL` listing
the checks passed under it.

`kind` says how the model's true value stands to the number:

| `kind` | The number is |
|---|---|
| `MODEL` | the model's value; the entry records the conditions or caller settings under which that holds |
| `LOWER_BOUND` | at most the true value: what it does not count can only add |
| `UPPER_BOUND` | at least the true value: what it counts that may not all contribute can only lower it |
| `ESTIMATE` | the stated model's number outside a validity limit or with a stand-in input, with no proven relation to the true value |

Beside `kind`, three lists say why, in fixed words and numbers only: `omitted` (`part`, where: `at`, the most it can add:
`upper`, the caller setting that left it out: `setting`), `assumed` (`input`, `at`, the stand-in `value` and `unit`) and
`validity` (`parameter`, `minimum` / `maximum`, `observed`, `within`, `setting`, `at`). `code` names the issue that states
the same fact; `setting` is a key of the run's settings record (`manifest.numerical_tolerances`).

`omitted[].part`:

| Part | Class | Contribution not counted |
|---|---|---|
| `BELOW_NEGLIGIBLE_LEVEL` | accounting | light below the negligible level of its source line, not followed, probed or listed |
| `BELOW_POWER_FLOOR` | accounting | light on paths ended at the absolute power floor |
| `BUDGET` | uncertainty | light a resource budget stopped before its end |
| `OUT_OF_MODEL` | uncertainty | light on a path that ended where a surface, medium or coefficient is outside the traced model |
| `NOT_FOLLOWED` | uncertainty | light above the negligible level the trace does not follow as beams: ghost reflections, coating leakage, scatter, light beside an element, a beam's tail beside an active area |
| `POSSIBLE_RETURN` | uncertainty | light that may come back between facing surfaces or around a closed path beyond the passes counted |
| `BEAM_PART_ELSEWHERE` | uncertainty | the part of a beam that leaves it beside its chief ray while the number applies the chief ray's power to the whole beam |
| `NOT_ACCOUNTED` | uncertainty | the accounting of left-out light was not computed (the capability that bounds it did not run) |
| `EDGE_DIFFRACTION` | uncertainty | the effect of a hard edge on a beam's profile beyond the truncation of its power |
| `SOURCE_NOT_DERIVED` | uncertainty | the light of a source whose emitted power the model does not derive |
| `UNRESOLVED_MODES` | uncertainty | power coupled into resonator modes above the highest mode order resolved |

An accounting part is left out by a stated level: the number is the model's value truncated there, and the part changes
no state, status or assessment. An uncertainty part is light whose contribution is not known to be below a stated level.

`assumed[].input`:

| Input | Stand-in used |
|---|---|
| `HELD_AT_NEAREST_STATED_POINT` | a response used outside the angle or wavelength domain where it is stated is held at its nearest stated point |
| `LAW_BEYOND_STATED_RANGE` | a dispersion or response law is evaluated outside the range its data states |
| `UNSTATED_GROUP_DELAY_ZERO` | a pass whose group delay the design does not state adds none |
| `UNSTATED_PHASE_AS_INTERFACE` | a coating whose phase the data does not state has the phases of the bare interface |
| `THIN_ELEMENT_PATH` | an element without a thickness adds the thin-element optical path and no substrate path |
| `FULL_COHERENCE` | a source that declares no coherence length is coherent at every path difference |
| `NO_UNSTATED_LOSS` | a loss the design does not state is zero |
| `UNSTATED_CONDITION_AS_DESIGN` | a test condition a rating does not state equals the design's light and the exposure definition used here |
| `NOMINAL_OPERATING_POINT` | the quantity is evaluated at the operating point named in `value` |
| `GAUSSIAN_PROFILE` | light whose profile is not known to be Gaussian is a Gaussian beam of its second moments |
| `SINGLE_RAY_FOR_BEAM` | one ray (the chief ray, or the pupil ray that carries the beam's reference) stands for the beam |
| `NOMINAL_AXIS` | decentres and tilts accepted within their tolerance are taken as zero |
| `RESPONSE_MODEL_UNVALIDATED` | an element's response model is used for a quantity its data does not state as validated |
| `UNIFORM_PUPIL` | the pupil is lit uniformly, not with the beam's profile |

**How to read a number.** A number is the model's value only when no entry names it or its `kind` is `MODEL`; read
any other kind, and a kind your reader does not know, as "not the model's value". A null is no number: the state word and
the issue at the quantity say why.

**The state word beside it.** `quantity_state` (and a status, `valid` on a Gaussian state) keeps saying what is proven
about the quantity; `value_basis[field].kind` says what the number in the field is. `UNKNOWN` and `OUT_OF_MODEL_SCOPE` may
stand beside a number with its entry. A `KNOWN` quantity carries only a `MODEL` entry; `KNOWN_ZERO` and `NOT_APPLICABLE`
carry none. An entry feeds nothing else: no plane or run status, no exit code, no `outcome.execution`. A finding whose
whole content a quantity carries is a notice (status `NOT_APPLICABLE`) located at that quantity.

**Targets.** A target reports the number of the quantity it reads with that quantity's entry under `value_basis["value"]`.
The Engine decides it only where the number proves it: a `LOWER_BOUND` meets a minimum it satisfies (status `WARNING`)
and misses a maximum it exceeds; an `UPPER_BOUND` the reverse; an `ESTIMATE` decides nothing (`INDETERMINATE`, status
`UNKNOWN`, or `OUT_OF_MODEL_SCOPE` where a validity check failed); a number whose measurement is `UNKNOWN` or
`OUT_OF_MODEL_SCOPE` decides nothing either, also where it misses its bound.

**Where to find it.** `os_optics.results.basis_of(record, field)` returns the stored entry, else the entry the record's
own fields imply (`quantity_state`, `power_complete`, `untraced_power_W`, `power_upper_bound_W`, `power_lower_bound_W`,
`power_out_of_scope_at`, `power_taken_before_*`, `detected_power_min_W`, `coherence_factor_min` / `_max`), else `None`;
runs written before the field existed answer the same way. The rows of `targets()`, `monitors()`, `monitor(id)`, the run
summary and the JSON API carry the same entries under `value_basis`, keyed by the row's name of the number; the
`monitors`, `arrivals`, `superpositions`, `exposures` and `power-ratings` tables have a `value_basis` column (JSON in CSV,
empty when every number of the row is the model's value). The Local Validation record carries the numbers without it.

**A coating met outside its data.** A beam whose chief ray meets a coating at a wavelength or angle the coating's data do
not cover goes on with the coating's values at the nearest stated point. Its powers behind that surface (`beam.power_W`,
`detected_power_W`, the wavelength's `power_W`, the plane's `total_power_W`, a power-derived target's `value`) are
`ESTIMATE`: `assumed` holds `HELD_AT_NEAREST_STATED_POINT` at the surface with the wavelength or angle the data was read
at, `validity` the data's range, the value asked for and `within` false, both with `code` `COATING_OUT_OF_RANGE`; the
surface is also in `power_out_of_scope_at`. Positions, directions and beam sizes there are the model's values.

## Power on a plane where beams of one source arrive

**What `power_W` adds.** `power_W` and `ObservationResult.total_power_W` add the powers of the beams that arrive on the plane: Pa + Pb. The
sum does not depend on the phase between the beams, so it stays constant across a fringe. Where two or more beams of one wavelength arrive,
`power_basis` is `INCOHERENT_SUM`, and a `power_W` target that sums them carries `POWER_INCOHERENT_SUM` and says so in its message; the
assessment is unchanged. Elsewhere `power_basis` is absent.

**What the coherent quantity is.**

- `coherent_power_W` (target) and `CoherentSuperposition.total_power_W` (per plane and wavelength, `superpositions`): the power of the coherent
  sum of all beams through the whole plane.
- `InterferenceResult.combined_power_collinear_W` (per pair of nearly collinear beams): Pa + Pb + 2 sqrt(Pa Pb) eta_pol eta_sp Re[gamma exp(i phi)],
  with eta_pol the polarization overlap, gamma the complex degree of coherence (complex where the two paths' group-delay dispersions
  differ, its magnitude `coherence_factor`), eta_sp the envelopes' overlap and phi `phase_difference_rad`. Nearly
  collinear means a fringe period above 10 beam diameters, or no fringes on the plane.

Both need a declared coherence length (`coherence_length_mm` or `linewidth_nm`) on the source.

**How to read the range.** The catalog's splitter, mirror and anti-reflection coatings publish a group-delay bound and no phase. In a two-arm
interferometer the arms meet those coatings differently, so:

- `phase_difference_rad`, `combined_power_collinear_W` and `CoherentSuperposition.total_power_W` are stated at the trace's default interface
  phases (a splitter's reflection pi/2 from its transmission, a mirror's (pi, 0), a bare interface's Fresnel phases): each carries
  `value_basis` of kind `ESTIMATE` with `UNSTATED_PHASE_AS_INTERFACE` and the passes that leave the phase open, in the form
  `<surface_id>:<event>:<side>`. Where such a pass also states no optical path (an ideal or tabulated thin lens, a Faraday rotator, a
  grating) they have no value (`PHASE_UNDEFINED`) and `phase_statement` names the passes.
- The degree of coherence and the fringe visibility keep their values, within `coherence_factor_min` / `coherence_factor_max`. Where a
  differing pass bounds no group delay they are stated at no added delay there (`value_basis`: `UNSTATED_GROUP_DELAY_ZERO` with the passes);
  where a pass of open phase meets a beam mixing s and p, the polarization overlap and the visibilities are on the default phases
  (`UNSTATED_PHASE_AS_INTERFACE`).
- `combined_power_collinear_min_W` / `_max_W` (pair) and `total_power_min_W` / `total_power_max_W` (plane) give the power's range over every
  phase: Pa + Pb -/+ 2 sqrt(Pa Pb) eta g eta_sp, the least at 0, with g the largest degree of coherence the bounds allow (the line's where a
  pass bounds none) and eta the polarization overlap (1 where a pass mixes s and p).
- A `coherent_power_W` target reports the value at the default phases with its entry and reads that range, widened by the light the trace
  follows that may reach the plane. A limit inside that interval is `INDETERMINATE`; a limit outside it decides the target. Scan below: a 1e-3 W maximum against the range's 9.4053e-4 W is
  `INDETERMINATE`, because a ghost reaches the plane.
- `optical_path_difference_mm` gives the change of the phase between two positions of an arm: 2 pi / wavelength per mm of its change. It has
  its value while the phase itself is open, unless an element on the way states no optical path.

The pair's range uses the envelopes' overlap, the plane's the whole-plane integral including the beams' tilt. For collinear beams they are the
same number. For beams at an angle within the collinear limit the plane's cross term is smaller, by 1.2 % at that limit, and the pair's range
holds the plane's.

Two beams crossing at a large angle have no collinear range: their fringes average out over the plane, so `total_power_W` has its value there
while `phase_difference_rad` stays open. Two beams crossing the plane in opposite directions add: `combined_power_collinear_W` = Pa + Pb, with
no range.

**What makes it a value.** State the phases of the coatings the arms do not share: a `phase` table (`reflection_s_rad`, `reflection_p_rad`,
`transmission_s_rad`, `transmission_p_rad`, over the wavelengths of the part's band) on a declared coating (`model_coatings`).
`phase_difference_rad`, `combined_power_collinear_W` and `total_power_W` then have values, and `total_power_min_W` / `_max_W` give only the
narrow range the group-delay bounds leave. A pass both arms make with the same angle and polarization state cancels and needs no statement.

Scan of a catalog-built Michelson at 780 nm, one arm mirror in steps of lambda / 8:

| Coatings | `power_W` | Coherent power |
|---|---|---|
| Catalog (no phase) | 0.000470268 W at every position | range 6.095e-9 W to 9.4053e-4 W at every position, target `INDETERMINATE` |
| Declared copies with stated phases | 0.000470268 W at every position | 9.4046e-4 W, 4.7027e-4 W, 7.42e-8 W, 4.7027e-4 W, 9.4046e-4 W, ... |

With the same parts, a bright state and a dark state have the same `power_W`. They differ only in the coherent power.

## Transverse-plane handoff

`handoff(monitor)` / `os-optics results handoff RUN --monitor ID` hands the beams at one observation plane to a tool
that propagates fields itself (a scalar diffraction code, a full-wave solver for a local device, a CAD check). Place the
plane where the external computation starts -- for diffraction at a stop, just before the stop -- and read, per arriving
beam, in the plane's (u, v, n) frame (the document is `opensci.transverse_handoff` 1.2; 1.2 adds `validity.model_scope`,
the beam's `value_basis` and `gaussian.value_basis`, and the export columns `model_scope` and `value_basis`):

* power, vacuum wavelength, medium index, chief-ray hit point and direction, angle of incidence;
* the complex beam parameter per Gaussian axis `q = [z, z_R]` (`z` = waist distance, positive when the waist lies
  upstream), 1/e² beam and waist radii, wavefront curvature radii (positive: diverging; `null`: flat), M², and the two
  Gaussian axes as (u, v, n) components;
* the Jones vector and its basis (e1, e2 as (u, v, n) components), the field phase at the chief ray and the optical path
  length -- both `null` behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating, whose model states no optical path
  (`validity.phase_defined` false; the linear-optical-network export refuses such paths, `PHASE_UNDEFINED_IDEAL_LENS`);
* `validity`: `beam_parameters` `TEM00` (an M² = 1 beam no aperture cut by more than the run's
  `gaussian_clip_warn_fraction`), `EMBEDDED_GAUSSIAN` (M² > 1), `TRUNCATED_GAUSSIAN` (an aperture upstream cut more than
  that share, the level at which the trace itself reports `GAUSSIAN_TRUNCATION`), `UNAVAILABLE` (no Gaussian state, e.g.
  after `GAUSSIAN_SEVERE_TRUNCATION`) or `POWER_OUT_OF_MODEL_SCOPE` (the beam's power is out of model scope behind
  `validity.power_out_of_scope_at`: `power_W` null, no field), `tem00_field_applicable` and a statement of what the
  numbers mean. Every cut is stated in numbers, whatever its size: `truncated_upstream` lists each aperture with the
  power it removed (`power_W`), the share of the arriving power that is (`power_fraction`) and whether that share is above
  the warning share (`warned`), with the run issues there; `truncated_power_fraction` is their sum and
  `field_intensity_ripple` the sum of `2 sqrt(f) + f` over the shares `f`: the largest relative change of the on-axis
  intensity behind a centred round hard edge that cuts `f` of a Gaussian beam (1 % at `f = 2.5e-5`, an opening of 2.3 beam
  radii; 6.4 % at `f = 1e-3`), and to first order the relative drop of the focal peak against the untruncated beam of the
  same power.
* `numerical_settings`: the limits of the run the document was read from that its validity and a field stage built on it
  apply (`gaussian_clip_warn_fraction`, `gaussian_scope_divergence_rad`): a Project's `numerical_settings` hold here.

With `element` (a component instance id: `--element STOP`) the document adds each surface of that element: its clear
aperture (circle with radius and obstruction, or rectangle with half widths and four corners), centre, surface axes and
normal as (u, v, n) components of the plane, the rim centre of a curved surface, and `in_plane` when the surface lies in
the observation plane (the outline is then exact in u, v).

The document states the field-construction convention (`conventions`):

```text
E(x, y) = sqrt(2 P / (pi w_x w_y)) exp(-x^2/w_x^2 - y^2/w_y^2) exp(+i k (x^2/(2 R_x) + y^2/(2 R_y))) (J_1 e1 + J_2 e2) / |J|
        = sqrt(2 P / (pi w_x w_y)) exp(+i k x^2 / (2 conj(q_x))) exp(+i k y^2 / (2 conj(q_y))) (J_1 e1 + J_2 e2) / |J|
```

with time dependence `exp(-i w t)`, `k = 2 pi n / lambda0`, x and y measured from the chief-ray hit point along the
Gaussian axes in the plane perpendicular to the beam, |E|² in W/mm² integrating to `power_W`, the phase at the chief ray
`phase_rad` (or 0) and the Gouy phase omitted. It holds for a TEM00 beam only. M² > 1 alone does not determine the modal
content or the coherence of a beam, and intensity or power alone does not determine a complex field: OpenSci exports beam
parameters, never a sampled field. After `GAUSSIAN_TRUNCATION` the power is exact (the Gaussian power the aperture passes)
while q, spot size and irradiance remain those of the untruncated Gaussian: start a diffraction-accurate propagation at a
plane before the aperture. The export kind `handoff` writes one row per beam (CSV / JSON Lines) or the whole document
(JSON: rows plus the plane, the element and the conventions in `summary`).

Verification (`tests/test_transverse_handoff.py`): a field built from the exported q by the stated convention and
propagated with an independent angular-spectrum propagator reproduces the Engine's beam radius at a second plane (a
reversed curvature sign does not), |E|² integrates to the power, the transmitted power after a stop equals
P (1 − exp(−2a²/w²)) and its cut share exp(−2a²/w²), the ripple figure equals the largest on-axis deviation of a numerical
Fresnel integral of the hard-cut beam, and a decentred slit's outline lands where a hand-computed frame change puts it.

## Wavefront

Geometric optics. Each pupil ray starts on the source plane along a normal of the source's Gaussian phase front
(curvature `1/R` at the source plane). Its optical path from that phase front is `W0 + OPL`, with
`W0 = n_source (σ·ρ) / (√(1 + |σ|²) + 1)` for launch offset `ρ` and launch slope `σ` (exact for a round beam: its rays
meet in the centre of curvature). Each ray is continued along its direction, in the medium at the plane, to the reference
surface through the chief ray's hit point:

* `reference="plane"`: perpendicular to the chief ray;
* `reference="sphere"` about `center`: a design-frame point, `gaussian_waist` (a stigmatic beam's Gaussian waist) or
  `ray_focus` (the least-squares point closest to the arrived pupil rays).

`OPD = (optical path to the reference surface) − (the chief ray's)`, in vacuum wavelengths; positive = the ray lags the
chief. The equal-weight mean (piston) is removed; RMS and PV are reported. Pupil coordinates are the rays' launch offsets
in units of the source's 1/e² radii (fit Zernike or Seidel terms on them). Refusals: `RAY_OPL_UNDEFINED` (after an ideal
or tabulated thin lens, a Faraday rotator or a diffraction grating), `TOO_FEW_RAYS`, `CROSSING_NOT_FOUND`, `REFERENCE_UNDEFINED`, `RAY_MISSES_REFERENCE`, `RAY_DATA_NOT_AVAILABLE`.
Diffraction is not included: this is the aberration of the traced geometric bundle.

Verification (`tests/test_results_layer.py`): a plane window leaves a collimated wavefront flat; a diverging Gaussian
launch is a perfect sphere about its centre of curvature; a plano-convex singlet agrees with an independent meridional
trace to 1e-10 mm and its fitted ρ⁴ term with the Seidel thin-lens `W040` within 2 %.

## Exposures and power ratings

At a glance:

* **What the bounds count.** Every beam the trace follows; beams of one source and line add their fields with open
  phases, so a surface's exposure is a range. With power ratings also the light the trace does not follow (ghost
  reflections, a coating's transmission, light passing an aperture or an element), followed to every surface, and the
  light left out as negligible or of unknown fate whose power is known: at the diffraction limit of its in-phase power,
  times its open paths' crossings and its returns between surfaces facing each other.
* **Closed returns.** Left-out light that may circulate through three or more surfaces of separate elements (a ring, a
  folded or coupled cavity, a periscope's windows through its folds) leaves every surface it may reach without an upper
  bound: its rating comparison and `power_rating_utilization` target stay undecided, one `EXPOSURE_BOUND_WITHHELD`
  notice of the run names the return's surfaces (`closed_return_surfaces`) and the surfaces without a bound
  (`surface_ids`), and the traced hits and incident power are still reported. A stack of parallel planes (windows, plates, flat mirrors on one axis) keeps a bound that counts all its
  returns where a series of them converges, and is treated like a ring where none does.
* **A bound withheld.** `EXPOSURE_BOUND_WITHHELD` names what has no upper bound and what obtains one, and moves no bound, comparison, target or status: a rated surface through a pair whose gain bound is 1 or a refused coating use, and a plane through such a pair or a closed return (`quantity` `power_upper_bound`). A closed return leaves the surfaces of a rated run without a bound under one notice of the run (`cause` `CLOSED_RETURN_NOT_BOUNDED`), and stored light at the interaction limit carries `MAX_INTERACTIONS`.
* **Internal folds** of a plane-faced solid (a cube's diagonal and faces, a penta prism's mirrors, total internal reflection there) are counted: the closed walks inside it, stepped between its clear apertures in the mirror images of the light's entering directions, add to the open paths' count; a solid whose walks have no certified sum is a closed return. **Not counted:** light that total internal reflection holds in a curved lens body, and light that refraction turns back onto a refracting surface it already passed before its next reflection: those surfaces keep the pair and open-path bounds. Held light is zero at the reference designs' feeds (internal angle at most 0.9 deg against 41.4 deg), and 2.6-16.9 % of the power only at entry cones of 75 deg to grazing. A search of 14 fixture runs and 540 two-singlet configurations found no traced path that meets a refracting surface twice between two reflections (plane-faced prisms and retroreflectors, ball and hemispherical refractors, trains of three or more elements and catalog products were not searched). There is no thermal or damage model.
* **Coatings.** A catalog or Project coating keeps its published values beyond its published angles and wavelengths (held flat out to 0 / 90° and 100 / 20000 nm) for traced light, for left-out light met from air, for a record whose angles refer to the incident medium and for a plane's power bound; the Local Validation reports a traced beam that meets it beyond its published data (`OUT_OF_MODEL_SCOPE`). Left-out light that meets an air-equivalent record from a medium denser than air is read at the air-equivalent angles of its bounded extent: a use outside the record's angles or wavelengths, above the critical angle, or whose extent cannot be bounded leaves that face, and a pair the refusal leaves at a gain bound of 1, without an exposure upper bound (`EXPOSURE_BOUND_WITHHELD`).
* **What decides.** `WITHIN_RATING` when the exposure's upper bound is within a rating that applies as stated, or whose
  test beam is at least as large as the light, and within every rating that may apply; `EXCEEDS_RATING` when its lower
  bound is above a rating that applies as stated, or whose test beam is smaller than the light. A smaller-beam rating --
  and a rating that cannot be checked, whose threshold may be at most its value -- leaves the surface no upper bound
  unless a rating of the same unit that applies as stated or with a larger test beam has a lowest damage threshold no
  larger than its highest (`W/cm` values compared as P/d of the 1/e² diameter).
* **Several elements.** Ghosts whose Gaussian state is not valid and the light left out as negligible count at their
  diffraction limit, so the exposure's upper bound lies far above the traced beams' and a utilization target there
  passes only with a large rating margin. An exceedance in W/cm² or J/cm² is decided at a much smaller margin, since that
  light, of small power, cannot undo a traced beam over the area it covers; in W/cm the margin is larger, since a thin
  sheet of that light meeting a face at a shallow angle keeps its P/d along a strip of the face. Coatings of low
  reflectance on the faces (declared or from the catalog) and fewer surfaces that may reflect the light make the bound
  smaller; where that light alone keeps a rating undecided, the run follows it deeper by itself.

A beam whose power is not known (a grating's order without a response that gives it) has `beam.power_W` `null` and
`power_upper_bound_W` on its arrival, `beam.polarization` `null`, and `final_power_W`, `power_start_W`, `power_end_W` `null` on its path: read a
null as not known, never as zero. Its position, direction, Gaussian beam and rays hold; `observed_W` and `total_power_W` do not count it.
A plane where a beam's power is out of model scope (`power_out_of_scope_at`) is left out of its line's `observed_W` and listed
in `observed_out_of_scope_at`.

`exposures` lists every physical surface a traced beam meets (an observation plane is not a surface; a surface no beam
meets is not listed, never reported as 0). Per beam (`hits[]`, one per path and pass) at the point where it arrives,
before the surface's clear aperture:

| Quantity | Definition |
|---|---|
| `incident_power_W` | the beam's power arriving at the surface |
| `beam_radius_x_mm`, `beam_radius_y_mm` | the Gaussian beam's two 1/e² radii (its own transverse axes; second-moment radii for M² > 1) |
| `aoi_deg`, `footprint_semi_axes_mm` | chief-ray angle of incidence on the local surface normal; semi-axes of the 1/e² footprint on the surface |
| `peak_irradiance_W_per_cm2` | `2P / (π w_x w_y)` in the beam's transverse plane; `peak_irradiance_surface_W_per_cm2` is that × cos(AOI) |
| `linear_power_density_W_per_cm` | `P / d`, `d` the 1/e² diameter `2 w` of the narrower axis |
| `pulse_energy_J`, `peak_fluence_J_per_cm2` | pulsed source: `E = P / f_rep` (P average power), `2E / (π w_x w_y)` |

"Peak" is the spatial peak of the Gaussian profile. `P` is the average power, so for a pulsed source
`peak_irradiance_W_per_cm2` is the spatial peak of the **average** irradiance, not the irradiance at the top of a pulse;
a pulse is described by `pulse_energy_J` and `peak_fluence_J_per_cm2`. On a surface several beams reach, a peak fluence
is the energy of one pulse per area. A pulse holds every line of its source along one path; the pulses of different
sources, or of one source along different paths, need not arrive together: they are bounded from below by the strongest
single pulse and from above by their sum.

A beam without a valid Gaussian state there (invalidated upstream, behind a central obstruction, aberration dominated) is
`UNKNOWN` with its reason; a beam that crossed a tabulated thin lens whose `supported_metrics` do not cover the exposure
behind it (`power_rating_utilization`, or `power_W` with `beam_radius_mm` or both `beam_radius_x_mm` and
`beam_radius_y_mm`) is `OUT_OF_MODEL_SCOPE` (`METRIC_NOT_VALIDATED_FOR_ELEMENT`): the spot behind the lens is the ideal
focusing's. Their beam-size quantities are `null`.

How the light on a surface combines. The exposure is the intensity of the summed fields of the light arriving at the
surface from either side, in irradiance units. Light of different spectral lines or of different coherence groups
(mutually incoherent sources; one group per source unless the Engine source states `coherence_group`) has no
interference term in the time average: its values add. Beams of one coherence group and line are mutually coherent and
add their fields with phases the exposure leaves open -- two branches of an interferometer, a window's rear face and its
own double reflection, and also beams crossing the surface in opposite directions (a beam and its reflection returning
from a mirror, a window's front face and the reflection of its rear face inside the glass), which form a standing wave.
For the beams' values `v` (any quantity of the table) the combination lies between

* at least `max(0, √v_k − Σ_{j≠k} √v_j)²` and `Σ v − 2 Σ_{i<j} γ_ij √(v_i v_j)` (the larger of the two), and
* at most `Σ v + 2 Σ_{i<j} γ_ij √(v_i v_j)`, which is the in-phase maximum `(Σ √v)²` for full coherence,

with `γ_ij` the degree of coherence from the source's declared coherence length (`linewidth_nm` or
`coherence_length_mm`), as for the coherent superpositions and the two-beam analyses: a finite bandwidth decorrelates two
fields by the delay of their envelopes, so `τ` is their relative *group* delay -- the difference of their group optical
paths `Σ n_g L` (`n_g = n − λ dn/dλ`), not of their phase paths `Σ n L` -- and the Gaussian spectrum of `L_c` after that
delay and their relative group-delay dispersion `B = Σ k₂ L` (`k₂ = λ³/(2π) d²n/dλ²`) gives `(1 + b²)^(−1/4)
exp(−(τ/L_c)² / (1 + b²))`, `b = 2B/L_c²` (`exp(−(τ/L_c)²)` without dispersion; higher orders of the dispersion are not
included), taken at its largest over the surface where both beams carry light. An interferometer balanced for its group
delay -- the white-light fringe -- is coherent there although its phase paths differ by many coherence lengths. `τ` is
the chief rays' value only along the chief rays; across the surface it changes with the beams' tilt
(`|n_i d_i − n_j d_j|` per mm, on a plane only its part along the plane), offset and wavefront curvature, and for a beam
that crossed a plate of glass with the rays' angles (a ray at angle θ crosses `L cos θ_c / cos θ` of it: at most
`|n_g − n| L (cos θ_c / cos(θ_c + √K s) − 1)` over the rays within `√K` slope envelopes `s = w|1/q|`), so the bound
takes, over radii `√K w` around the first beam's chief ray, the largest value inside plus `exp(−K)` for the profiles
outside, and keeps the smallest. Collinear beams (an interferometer's recombined arms, a standing wave at normal
incidence on a plane) keep about their chief rays' value; beams crossing at an angle are fully coherent somewhere in
their overlap (`γ = 1`). The lower bound at a point uses the bound at that point. `γ = 1` when the source declares no
coherence length, the path difference is not defined (behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating), a beam's Gaussian state is
not known, a beam crossed glass other than between two parallel planes (a lens, a prism, a cube: its envelope delay across
the beam is not followed), the two beams crossed passes of undetermined group delay differently (below), or the chief
rays' value is at least 0.999. A pulsed source's pulses may be mutually coherent (a mode-locked train: each pulse a copy
of the last): two of its beams are coherent again near every whole multiple of its period length `c / f_rep`, where one
beam's pulse overlaps a later pulse of the other, so the bounds take the larger of the coherence at the delay and at its
distance to the nearest non-zero multiple, and a coherence metric -- the coherence margin among them -- has no value
where such a revival could change it (`COHERENCE_UNDEFINED`). A bare interface and an uncoated mirror delay nothing beyond the media's group paths; an
element without a thickness in the model (an ideal or tabulated thin lens, a splitter plate, a polarizer, a retarder, a Faraday rotator, a diffraction grating, an
attenuator, an aperture that attenuates) and a coating whose phase is not stated constant over wavelength -- also a coated face met beyond the critical
angle -- delay light by an amount the design does not state. Each such pass is listed on the beam
(`arrivals[].group_delay_unknown_at`: `<surface_id>:<event>:<side>@<angle of incidence>`, with `[s1,s2,s3]`, the beam's
normalized Stokes parameters along the pass's own axes, where the delay may differ between polarizations: s and p of a
coating met obliquely, a retarder's fast and slow axes, a polarizer's two axes when it passes light along both); two
beams' relative envelope delay is known only when they crossed the same ones at the same angles in the same polarization
states -- otherwise their degree of coherence is 1 in every bound, and the coherence metrics (`coherence_factor`,
`fringe_visibility`, the pair's peak irradiance) are stated at no added delay at those passes, with `value_basis` of
kind `ESTIMATE` naming them (`UNSTATED_GROUP_DELAY_ZERO`): reported, never deciding a target. `coherent_power_W` reports
its value at no added delay with that entry too, and a target on it is judged on `total_power_min_W` /
`total_power_max_W` (any delay there, a degree of coherence up to 1, every phase), widened by the other light that may
reach the plane: traced light outside the superposition (ghosts), surface reflections the trace does not follow
(`UNTRACED_REFLECTION`) and light left out as negligible, which may leave the interval without an upper end. A limit
outside that widened interval decides the target; a limit inside it is `INDETERMINATE`, as for any coherent sum. A
polarization interferometer whose two beams cross a common retarder or an oblique coating as s and p therefore has its
coherence there on that convention only. Two equal branches
in phase give 4 times one branch, in antiphase 0. Beams at different places combine through their Gaussian profiles: the
lower bound is the combination at a traced beam's centre (each beam's value there, `exp(−2ρ²)` of its peak with `ρ` the
distance in units of its 1/e² radii), the upper bound the combination of the peaks. Light of unknown place -- a ghost
of unknown size, the light left out as negligible (below) -- may be at its bound at a beam's centre but cannot be strong
over the whole area the beam covers: in peak irradiance and peak fluence the lower bound is also a traced beam's mean
against a window over its area (`exp(-s ρ²)`, the best of a range of widths), less the other beams' overlap with that
window and `H √P` for light of power `P` crossing the surface from one side, whatever its directions (`H` the window's
overlap with plane waves weighted by `1 / cos`) -- for a beam of one Gaussian mode on a plane or a sphere, within the
clear aperture. In `P / d` the window counts that light as Gaussian beams no more focused than the diffraction limit
and crossing the surface from one side (their far-field cone on one side of it near the beam): each lowers the mean by
at most its `P / d` over the strip of the window it covers. A thin sheet meeting the surface at a shallow angle covers
a strip about `w² / w₀` wide (`w` its narrower radius, `w₀ = λ / (π n)`), so `P / d` gains less from the window than
the irradiance -- for any traced beam on a plane or a sphere. A single beam, or beams that cannot interfere and whose peaks
coincide, have equal bounds: the exposure is exact. `coherent_groups` lists the groups of two
or more mutually coherent contributions (`members` as `traced:<path>:<pass>`, `untraced:...`, `beside:...`, the
`incidence` -- `ALONG_NORMAL`, `AGAINST_NORMAL`, or `BOTH_SIDES` for a standing wave -- and the
`largest_coherence_factor`); a group with a traced beam names `COHERENT_BEAMS_BOUNDED` and makes the surface's exposure a
range (`complete` false). The surface values `incident_power_W`, `peak_irradiance_*`, `linear_power_density_W_per_cm` and
`peak_fluence_J_per_cm2` are the traced beams' combination at its largest (their sum where no two can interfere): upper
bounds, exact for a single beam. `incident_power_W` is a flux through the surface: beams crossing it in opposite
directions add their powers (no cross term), coherent beams from one side have cross terms up to
`2γ√(P_i P_j)(cos a_i + cos a_j)/(2√(cos a_i cos a_j))`. When the run's trace is incomplete (trace budget,
interaction limit, a path ended out of model scope, the light a tabulated thin lens does not deliver into its focused
beam) `complete` is false. In a run with power ratings, light of unknown fate whose power is known is bounded with the
light left out as negligible (below); light at the per-path interaction limit -- which may be light stored in a cavity --
leaves every surface without an upper bound and, coherent with the traced beams of its line, their lower bound at 0 (the
beams of other lines and sources keep theirs); without ratings any unknown fate does. In a run with power ratings, light
the trace does not follow (untraced ghost reflections, a reflector's coating transmission, light passing an aperture or
an element its chief ray misses) is followed through the optics to every order above `negligible_power_fraction` of its
line's power, each probe with the pupil rays of two whole rings of the beam it leaves, checked for aberrations like a
traced beam (an aberration-dominated probe arrival, e.g. a leak focused by a strongly aberrating lens, counts with the
diffraction limit of its power, never as its paraxial spot): where it meets a surface -- also one traced beams of the
same line meet, e.g. a window's own ghosts on its faces -- it is listed in `untraced_hits` (`complete` false): it raises
the upper bound and, coherent with the traced beams of its group, may lower the lower bound (it never raises it). A beam whose chief ray misses a rated surface while its Gaussian footprint reaches the surface's
support (within `aperture_quad_extent_sigmas` beam radii) is listed in `beside_hits` (`INCOMPLETE_ENVELOPE`): its
`miss_distance_mm` and the profile's values at that distance (`exp(-2 d²/w_max²)` of the peaks) bound the exposure from
above -- never a known zero. A beam without a valid Gaussian state (behind an axicon, an annulus) has no known profile
beyond its pupil rays: a rated surface within the reach they imply (`aperture_quad_extent_sigmas` times the largest
local beam radius `r / rho` of a live pupil ray launched `rho` source radii off the chief ray and found `r` mm off it
there; every crossed rated surface for a beam without pupil rays) is listed with no bound (`UNKNOWN`, the Gaussian
state's reason and `INCOMPLETE_ENVELOPE`); an untraced beam without a valid Gaussian state counts with the diffraction
limit of its power (below). Every arrival of the probes is listed, whatever its power. A bound decided by light the probes
followed names `UNTRACED_LIGHT_BOUNDED` (not `OUT_OF_MODEL_SCOPE`, which stays for light of unknown fate).

Light left out as negligible -- the branches of at most `negligible_power_fraction` times their line's input (or a deeper
pass's level) that neither the trace nor the probes follow, and the paths ended at the power floor `power_epsilon_W` --
has a known power but no known place. Focused, it can reach an irradiance far above its share of the power, so its
field is bounded by its diffraction limit: light of power `P` at vacuum wavelength `λ` in a medium of index `n` that
crosses a plane from one side reaches at most `I_max = 2πn²P/λ²` anywhere on it (writing its field as a sum of plane
waves, the Cauchy-Schwarz inequality over their transverse wavevectors `|k_t| ≤ 2πn/λ` with the power as the flux
through the plane; it is the peak `2P/(πw²)` of a Gaussian beam of waist `w = λ/(πn)`, and the same waist gives the
fluence `2E/(πw²)` and `P/d` bounds). A broad spectrum adds the limits of its parts: `I_max` scales with the mean of `(λ/λ')²` over it
(the Gaussian spectrum of a declared coherence length: `1 + λ²/(2π² L_c²)`; a pulse's transform-limited spectrum:
`1 + ln2 λ²/(2π² c² τ²)`, 1.0025 for 10 fs at 800 nm). The light is left out in pieces -- each branch not followed,
each path stopped at the floor -- and the pieces of a line are mutually coherent: where they meet, their fields add in
amplitude, so the bounds use their power together in phase, `P = (Σ√p_k)²`, not their total power (two equal pieces in
phase give twice their total). `exposures.faint_light` lists per source line the pieces' total power `power_W` -- with
the light of unknown fate whose power is known (the trace's or the probes' budget, a path ended out of model scope, a
line not launched, a tabulated thin lens's undelivered light) --, that in-phase power `in_phase_power_W`, and the level
its untraced light was followed to (`followed_to_W`). On every surface that light joins the classes of its line as a
field of open phase, place and polarization of amplitude at most `A = c √I_max(P)` (`n` the larger index of the
surface's two media), `c` the most its open paths' crossings bring onto a surface or across a plane: between two
reflections the light meets each lens and each refracting surface between media of different index at most once and
travels in at most `L = T + 1` straight legs (`T` those lenses and refracting surfaces, which may turn it), each meeting
any other surface at most `K` times, as often as a straight line can (a plane once -- a mirror, a splitter, a refracting
plane between media of one index, an observation plane --, a sphere or a conic twice, an asphere at most twice the
highest power of `ρ` in its sag, that power itself on a paraboloid or a flat base, twice that with an axicon term) -- it
meets a lens or a refracting surface between different media at most `m = 1` time and any other surface at most
`m = K L` times, also where a prism or a lens turns it back across a plane or onto a mirror or a splitter, plane or
curved --; after `j` reflections it is made of pieces from different sequences of `j` of the `N` surfaces that may
reflect it (no surface reflects it twice on an open path), mutually coherent, whose fields add: at most `a_j` times one
crossing's field, the smaller of the sum over those ordered sequences of the products of `m_s √R_s` (`R_s` a surface's
reflectance bound over every angle and side, `m_s` its meetings) and the square root of their number (the products of
`m_s`) times the power they share (the sum of the products of `m_s R_s`, at most 1). So `c = √2 m Σ_{j=0..N} a_j` (`√2`
for the light split between the two sides each time), at least 2 (the light once from either side): 2.26 on the faces of
four lenses with 0.25 % coatings and about 20.3 on an observation plane after them, 5.41 on an uncoated window's faces (a
bare face counts 1: the reflectance just below its critical angle), and it grows quickly where many surfaces may reflect
everything. On an observation plane the same light adds at most `m Σ_{j=0..N} min(1, R^j)` (`R` the sum of the
reflectance bounds, each times its `m`), at least 2, times its total power `power_W` to a transparent plane's power (each crossing's power added, as the plane adds its
beams'), `power_W` to an absorbing one's, and a field of at most `c √P` to the coherent power. It is larger where two surfaces facing each other return
it again and again and send it on to this one. The returns are those between two reflecting surfaces that face each other: a window's or a lens's
faces, a cube's, a window and a mirror, the mirrors of a linear cavity, also with lenses or plates between them
(`validation/resonance.py`). Each round trip keeps at most `g = r_a r_b` of the light's field, the two amplitude
reflectances bounded over every angle at which that pair's light can meet them. Between two surfaces light travels along
legs from a point of one's aperture to a point of the next one's (within the angle their bounding spheres subtend), and a
pair's light arrives at one of its surfaces along the leg from the other one or from a lens or refracting surface between
them -- one that meets the two's hull, or any lens or refracting surface two of them lie on opposite sides of, which may
turn the light far off the line between them (a thin lens turns light by at most `2 atan(ρ/2|f|)`, the slope-form lens's
largest deflection, `ρ` the largest height of its aperture above its axis; light after a lens meets a surface the lens
straddles from either side). Between two parallel planes with nothing between them that turns
light (and no lens or refracting surface off their hull through which light may go from one to the other, turned within
a lens's largest deflection or a refraction's largest deviation `acos(n_min/n_max)`; a refracting plane parallel to the
two keeps the light's momentum along them and counts only in the walk, a tilted plate or a wedge, which shifts the light
sideways, counts), it keeps its momentum along them, `n sin θ`: the momentum it brings from the rest of the scene -- along the leg
from every other surface that redirects light (from a plane, within the mirror images and refractions of the legs into
it), and within each source's beam, 8 far-field divergences of its direction -- passed on from plane to plane of that
normal as far as its walk along them, `s = Σ t k/√(n² − k²)` per pass, lets it reach their apertures; its hits on each
plane are `2s` apart and each return onto one plane needs a hit on the other between, so it comes back at most
`min(ρ₁/s, ρ₂/s + 1)` times (`ρ₁ ≥ ρ₂` the two apertures' radii) and its returns bring at most `min(M g, g/(1 − g))`. A bare interface counts Fresnel's s
reflectance at the widest angle; from its denser side only the angles below the critical angle count (beyond, the light
is held by total internal reflection), so a range that reaches the critical angle counts 1, the reflectance just below
it; a coating counts its largest over the range, at its ends and table angles (a catalog or Project coating holds its
published values beyond its published data, as above), and 1 where the range is not wholly inside the validity range an
Engine request states for a coating: its pair then bounds nothing. A record that states an air-equivalent angle range, met from
a medium denser than air, takes the use of the left-out light (the coating angle reference, below). An interface takes only light in its own medium on each side (indices compared, so air
and vacuum hold the same light). What each return sends out of the pair -- through its surfaces, past an aperture's edge, reflected by
a surface between them -- is at most that return's field and may reach any surface: every surface and observation plane,
from either side, gets `f = 1 + Σ g/(1 − g)` over the scene's pairs, `faint_light_return_factor` (about 1.04 in a scene
whose only pair is an uncoated window, 1.0025 for one lens with a 0.25 % coating, 1e6 with a cavity whose mirrors transmit
1e-6; `FAINT_LIGHT_RECIRCULATES`). A pair whose gain bound is 1 -- two surfaces that reflect everything, a coating met
outside its data, the faces of an uncoated lens, which light inside it can meet up to the critical angle -- leaves every
surface without an upper bound.

A pair whose gain bound is 1, or a closed return no series bounds, leaves a plane without a power upper bound, and the run names each such plane with `EXPOSURE_BOUND_WITHHELD`, with or without a rating. A pair at gain bound 1 also leaves every surface of a rated run without an exposure upper bound, and the run names each with it. The notice has status `NOT_APPLICABLE`: no bound, comparison, target or status moves, and the manifest's `warnings` lists it. Stating a coating whose reflectance bound is below 1 on the pair's surfaces, or blocking the closed path, obtains a bound. Which code each case carries:

- a pair at gain bound 1, rated surface: `EXPOSURE_BOUND_WITHHELD` at `surface:<id>`, `cause` `RETURN_PAIR_GAIN_BOUND_1` with `return_pair_surfaces` and, per pair in the same order, `return_pair_saturation` (why each of its two surfaces has a reflectance bound of 1: `CRITICAL_ANGLE_IN_RANGE`, `GRAZING_INCIDENCE_IN_RANGE`, `COATING_DATA_RANGE`, `COATING_EVALUATION_ISSUE`, `INDEX_UNKNOWN`, `REFLECTS_EVERYTHING`, `REFUSED_COATING_USE`) and `return_pair_angles_deg` (the angles of incidence its light meets each surface at); `return_pair_nominal_surfaces` / `return_pair_nominal_gain` name the pairs whose records state a round-trip gain at their centre-line incidence, with that gain; `return_scope` `SCENE_WIDE_NOT_LOCALIZED` says the pair's factor applies to every surface and plane (the surfaces its light can reach are not narrowed); `suggested_action` follows the cause;
- a pair at gain bound 1, any plane with `power_upper_bound_W` null: `EXPOSURE_BOUND_WITHHELD` at `observation:<id>`, `quantity` `power_upper_bound`, the same `cause`;
- a refused coating use, rated surface: `EXPOSURE_BOUND_WITHHELD` at `surface:<id>`, `cause` `COATING_USE_REFUSED` with `coating_record`, `incident_medium`, `incident_index`, `valid_aoi_deg`, `extent_in_medium_deg`, `air_equivalent_deg` and `coating_refusal` (`OUTSIDE_DECLARED_VALIDITY`, `OUTSIDE_DECLARED_WAVELENGTH`, `NO_AIR_EQUIVALENT_ANGLE`, `EXTENT_NOT_BOUNDED`); a surface held only through the pair a refusal leaves at gain bound 1: `COATING_USE_REFUSED_ON_FACING_SURFACE`; planes keep their bound and have no notice;
- a closed return, a rated run: one `EXPOSURE_BOUND_WITHHELD` at `exposures`, `quantity` `exposure_upper_bound`, `cause` `CLOSED_RETURN_NOT_BOUNDED`, with `closed_return_surfaces`, the surfaces and elements it leaves without a bound (`surface_ids`, at most 12, with `surface_count`; `element_ids`), the surfaces of every group the trail search screens (`closed_return_candidates`: closed returns between separate elements run only among the surfaces of one group, so the list says where any other closed return can be), the round gain bound of the trail found (`closed_return_gain`: the product of its surfaces' reflectance bounds) and, for a solid whose folds have no certified sum, why (`solid_fold_reasons`: `HELD_LOSSLESS_STATE_WALK`, `CONES_NOT_CLOSED`, `ROUNDS_NOT_SETTLED` or `SUM_NOT_CERTIFIED` -- what the fold analysis could not certify, not a statement that the light circulates); any plane with `power_upper_bound_W` null: `EXPOSURE_BOUND_WITHHELD` at `observation:<id>`, `quantity` `power_upper_bound`, `cause` `CLOSED_RETURN_NOT_BOUNDED`, with the same facts (a dark plane also `OBSERVATION_NOT_REACHED`, `UNKNOWN`);
- light at the interaction limit: `MAX_INTERACTIONS`, status `UNKNOWN`, at `path:<id>` (a dark plane also `OBSERVATION_NOT_REACHED`).

A surface notice has `element_id`, `surface_id`, `quantity` `exposure_upper_bound`, `wavelength_nm` and `suggested_action`, and names an undecided `power_rating_utilization` target of its element in `affected_targets`. The run's closed-return notice names the undecided `power_rating_utilization` targets of its `element_ids`. A plane notice has `observation_id`, `wavelength_nm` and `suggested_action`, and names no target.

**Coating angle reference.** Every coating record states the medium its angle range refers to: 629 of 707 records (catalog release 2026-10-28.1) declare an air-equivalent angle, 78 the incident medium. Met from a medium denser than air, an air-equivalent record is evaluated at the bounded angular extent of the light the trace leaves out, converted by `asin(n sin theta)`; a use outside the record's declared angle or wavelength range, a use above the critical angle (no real air-equivalent angle) or a use whose extent cannot be bounded is refused rather than approximated, and that face has no exposure upper bound (`EXPOSURE_BOUND_WITHHELD` names it, with the record, the medium, the extent and the air-equivalent angle). The refusal concerns the exposure upper bound of the refused face and of a pair it leaves at a gain bound of 1; a plane's power bound and every other surface read the record's stated value. The reference-design lens coating `AR_400-1100R0.01T0.99_AOI0-10` is published for 0-10 deg air-equivalent. In `focus` and `folded` the left-out light meets it from the glass within 9.82 and 9.80 deg air-equivalent: the value holds and the bounds stand. In `beam_expander` and `relay_4f` faint ghosts fill the lenses, their extent is not bounded below the record's range, and the four faces of each design have no exposure upper bound; the planes keep their power upper bound. At the 5.715 mm clear radius the face's normal alone gives 19.6 deg air-equivalent on R 25.8 mm and 9.65 deg on R 51.5 mm; the refusal stands on the ghosts' own rays, which reach 22-75 deg air-equivalent on those faces. To obtain a face's bound, use a coating whose record states those angles, or keep the beam well inside the clear aperture. A glass-side reflectance at high angles needs per-side or angle-resolved data. The extent covers left-out light from the reflection that leaves it out up to its next reflection, through refractions, plates and lenses, at the line's own level (`negligible_power_fraction` x its input); each later reflection of that light takes the coating's stated value at a field of at most `sqrt(R)` per reflection. Air-side incidence of an air-equivalent record, `INCIDENT_MEDIUM` records and uncoated surfaces read their stated values. The traced light's power reads the coating's value at the chief ray (a chief ray outside the record is `COATING_AOI_OUTSIDE_RANGE`).

Closed returns through three or more surfaces of separate elements (a ring, a folded or coupled cavity, a periscope's
windows through its folds) withhold the upper bound of every surface they may reach, with one `EXPOSURE_BOUND_WITHHELD`
notice of the run naming the return (`closed_return_surfaces`) and those surfaces; a stack of parallel planes keeps an all-orders series of its
returns where one converges (`1/(1 − A)` with `A = Σ m_s r_s < 1`, else the alternating series of its reflections along
the planes' normal) and is withheld the same way where none does (reference: rings, folded and coupled cavities, plate
stacks, `tests/test_closed_returns.py`, `tests/test_closed_return_stacks.py`).

**Internal folds inside one plane-faced solid.** A plane-faced solid (beamsplitter cube, polarizing cube, penta prism, right-angle prism) whose left-out light can return to a face through two or more of its other faces has those returns counted. The directions inside are the mirror images of the cones the light enters with (from every surface outside the solid that redirects light and every source's beam, refracted in); a step from one face to the next exists only where a straight segment of that cone joins the two clear apertures, and carries the face's reflectance bound over the step's angles, 1 where they reach the critical angle (total internal reflection); a step that meets an air-equivalent coating record beyond its stated angles reads the record's stated value, and that face has no exposure upper bound (`EXPOSURE_BOUND_WITHHELD` names the record), as for the first run of the left-out light; the closed walks at each face are summed with a certified bound, and the open paths' counts add every walk that closes a loop other than an alternation with a face the pair bounds already count. The closed walks back to an entrance face sum to 0.0076 on the catalog beamsplitter cube, 0.0102 on the polarizing cube and 0.0049 on the penta prism (its fold, 0.005 x 0.99^2); exact ray trees traced inside these solids never exceed them. The faces' exposure upper bounds rise by 0.03-1.2 % on the cubes, whose decisions stay, and by 3.9-10.3 % on the penta prism, whose S4 then reads `CONDITIONS_NOT_VERIFIED`. A solid whose closed walks have no such bound (a lossless return, more than 192 direction cones) is a closed return: no surface the left-out light reaches has an exposure upper bound, and an `OUT_OF_MODEL_SCOPE` issue names its faces. A solid whose light meets each face once between the returns of one pair (a window, a plate, a wedge, a dove prism) keeps the pair bound; there the pair model leaves out total internal reflection at a face met beyond its critical angle (a wedge's walk toward its apex). A solid with a crystal body, a non-convex body or a curved face keeps the pair and open-path bounds without its internal folds. Which solids are analysed: the plane faces of one element with one isotropic body index, convex (every corner of a polygon face, as a mesh solid's planar patches are, and every other face's aperture extent on the body's side of each face's plane) and with no source inside: catalog prisms and cubes, and mesh solids (`FACET_SOLID`) alike, each cell of an arrayed mesh member its own solid. A closed mesh solid of glass (a corner cube, a penta, rhomboid or dove prism, a Fresnel rhomb, a light guide) has no certified sum: the analysis keeps, per direction cone and face, one part of the face's aperture for all the visits there, so light that advances along faces met beyond the critical angle or silvered closes a lossless walk of states although the light itself moves on and leaves (a rhomboid's or a light guide's zigzag between two parallel faces, a dove prism's between its base and top, a penta's two silvered faces and a face met beyond the critical angle), and a corner cube's cones, imaged again by its partly reflecting entrance face, do not close within 192 (nor do those of a mesh whose rounded coordinates leave its faces slightly off right angles); such a solid is a closed return (`POTENTIAL_RETURN`, no exposure upper bound where its left-out light reaches). The analysis does not bound it; it has found no lossless return of the light: ray trees traced inside these meshes over the light the model admits find every circulation passing a partial reflection, and finite per-face sums of a ray's visits in amplitude (a corner cube's back face 1.25 with a bare entrance face, 1.053 with an AR coating of reflectance 0.0025; a penta's 1.25; a rhomboid's 3.75 and a light guide's 8.5 at the faces of their zigzag, a few lossless visits per pass). A mesh cube of index 1.05 lit across a corner (no face near its critical angle) is bounded, and the traced walks stay within its sums. The open paths' counts leave out the walks whose only loops are runs of alternations with one paired face, which the pair factor counts; that factor counts one such run per walk and a run with one partner, so a walk whose loops are alternation runs at two or more faces of one open path, or a run alternating with two partners at one face, is counted by neither. Its share is second order in the runs' sums `A - 1`: at most `prod A - 1 - sum (A - 1)` over a solid's faces, 1.1e-3, 1.5e-3 and 3.7e-3 of the solid's whole closed-walk product on the catalog beamsplitter cube, polarizing cube and penta prism (0 on the right-angle prism, which pairs no face); exact ray trees inside these solids have not exceeded the bounds.

**Light held by total internal reflection in a lens body.** The exposure upper bound on a lens face counts the single return between the face and its pair; light that enters the glass body and reaches the total-internal-reflection range is not counted. In the reference designs (focus, folded, beam expander, 4f relay, three-lens AR train) the internal angle reaches at most 0.9 deg against a critical angle of 41.4 deg (N-BK7 at 780 nm), no ray enters the TIR range, and the omitted term is zero (K = 1.000000). That angle is the entering beam's: ghosts that fill the same lenses reach 39.8 deg in glass, without reaching the TIR range, and the coating angle reference below states what they do to the exposure bound. A curved lens body has no bound on consecutive total internal reflections over arbitrary light inside it: in the 0.25 %-AR biconvex singlet (R 101.3 mm, 4 mm thick, 12.7 mm clear radius), rays started inside it in random directions meet its faces beyond the critical angle up to 12 245 times in a row before a partial reflection, so no series over its reflections converges for such light (this does not show that light the lens receives is held: at the reference designs' feeds none reaches the total-internal-reflection range); a bound on that light needs a limit on the power it holds, which the exposure model does not state. For wide entry cones the held power grows: at a 75 deg cone the 1 %-AR catalog lens holds 2.6 % of the power (4.6 % of rays ever reach TIR; amplitude factor 1.27, returned-power ratio 4.3); near grazing it holds 13.0 % (amplitude factor 2.36, returned-power ratio 16.0, up to 5 consecutive TIRs), and a 0.25 %-AR singlet reaches 16.9 % held (amplitude factor 4.53, returned-power ratio 69.6, up to 30 consecutive TIRs). A power-based bound on TIR-held light is needed only for such wide or grazing feeds; the reference designs are unaffected.

**Returns among refracting surfaces.** The pair model, the open-path count and the `A < 1` series count a refracting surface at most once between two reflections; light that refraction turns back onto a surface it already passed before being reflected again is not counted. A bounded search (14 executed fixture runs and 540 configurations of a two-singlet strong-curvature train, reflections traced with an explicit budget) produced no traced path that meets a refracting surface twice between two reflections; refracting-surface returns themselves are traced (0.25 %-AR faces return 0.99 % of the incident power to a plane behind the source). The search did not cover plane-faced prisms and retroreflectors, ball and hemispherical refractors, trains of three or more elements, or catalog products. The rule: this return class is outside the exposure bound.

Each class's upper bound grows by `2A Σ√v + A²`,
its bound at a beam's centre is taken with `A` off the amplitudes (`(√v_k − Σ√v_j − A)²`, cross terms `−2A√v`), the
window bound with `H c f √P*` off the beam's mean (`P*` the in-phase power, `f` the returns), and
`faint_light_peak_W_per_cm2` reports `Σ A²` over the surface's classes (`FAINT_LIGHT_BOUNDED`, `complete` false). A
rated surface no traced or followed light reaches is listed with no hits and bounds between 0 and that value -- never a
known zero. Wherever the scene has a pair whose gain bound is 1, and wherever light met the per-path interaction limit
(light that may be stored in a cavity the traced light shows), nothing bounds it: no upper bound. Where it
alone keeps a comparison from being decided -- the rating lies between the bounds with it and would not without it, or
its share of the light decides a condition (below) -- the run follows the lines concerned deeper (up to four more probe
passes, each down to a level the decision needs with headroom, never below the power floor), so that the probes' known
spots replace the bound, and computes the exposures again; a pass that runs out of the trace budget, meets light whose
fate it cannot follow or leaves no less light out is discarded and the bounds before it stand -- except one that meets
the interaction limit, whose light may be stored: it stands (no upper bound). Typical effect at the first pass: a 1 W
beam on an uncoated window at normal incidence, whose left-out light has `Σ A²` of about 3 % of the beam's peak on the
rear face: its range is [0.91, 1.49] times the traced value (the lower end from the window bound); a deeper pass narrows
it when a rating lies there (rated at 1.1 times the traced value: [0.92, 1.09], within the rating). `untraced_light_included` says whether this was done; without ratings the values cover the traced
beams only.

Each surface is compared with the power ratings of its element (a catalog product's `specifications.power_ratings`, a
declared model's `power_ratings`; a compound unit's member ratings apply to that member's surfaces). A comparison records
every condition a rating can state against the light on the surface (traced, untraced, beside it) in `conditions` -- one
entry each for `regime`, `wavelength`, `pulse_duration`, `repetition_rate`, `beam_diameter`, `pulse_count` and
`polarization`, with its `state`: `VERIFIED`, `DIFFERS` (with `effect`: `THRESHOLD_AT_LEAST_RATED`, the damage threshold
for this light is at least the rated value; `THRESHOLD_AT_MOST_RATED`, at most the rated value; or
`NO_ESTABLISHED_RELATIONSHIP`), `NOT_VERIFIABLE` (with `reason`: `DESIGN_INPUT_MISSING`, `RATING_INFORMATION_MISSING`,
`BEAM_SIZE_UNKNOWN`, `LIGHT_STATE_UNKNOWN`, `COHERENT_STATE_UNDETERMINED`, `PULSES_PER_SITE_UNDETERMINED`) or `NOT_STATED` (the condition
does not apply to the rating, such as a pulse count to a CW rating: never counted as verified, never keeps the rating
from applying) -- and the rated and design values, the relative tolerance and a `message`. A test condition the rating does not state -- its test beam diameter, a pulsed rating's repetition rate or pulse count, its polarization, its value's definition -- is `NOT_VERIFIABLE`
(`RATING_INFORMATION_MISSING`): the threshold depends on it. The
`value_definition` condition says what a J/cm² or W/cm² value is (below):

* the regime (`CW` for continuous sources, `PULSED` for pulsed ones), the wavelength within 2 % of the rating's, for a
  pulsed rating the pulse duration (read as a FWHM) and the repetition rate within 10 %: verified, else they
  differ with no established relationship. Followed light (traced, untraced, beside the surface) of a line the rating is
  not for that carries at most 1 % of the most light the surface can receive, in peak irradiance and every quantity that
  gives a comparison its direction (those of the test beam below), counts in the exposure but not in these conditions; a
  note names it and its share, and beyond that share the condition differs. Light left out as negligible of a line the rating is not for leaves each
  condition that line differs in (its regime; else its wavelength, pulse duration, repetition rate) not verifiable when it
  may carry more than 1 % of the most light the surface can receive, in peak irradiance or in the rating's own quantity
  (whichever share is larger). A surface only light left out as negligible reaches is compared with that light's lines,
  each with its regime, wavelength and pulse train: the lines that meet a condition verify it, a line that differs leaves
  it not verifiable beyond that share -- and whatever its power where light may be stored between surfaces facing each
  other (`MAX_INTERACTIONS`, or a pair whose gain bound is 1: nothing bounds that light), which holds for the lines of
  the stored light too;
* the test beam on the sample (`beam_diameter_mm`, `beam_diameter_definition`; not stated: not verifiable): its 1/e²
  diameter is read as the stated value (`1/E2`), the stated value / 0.5887 (`FWHM`) or from the stated value to 1.70
  times it when the definition is not stated (`rated_low` / `rated_high`; the standard width definitions of a Gaussian
  beam lie in that range). Every beam of known size is compared by its smallest 1/e² diameter across the beam
  and the largest 1/e² extent of its footprint on the surface (`design_low` / `design_high` over the beams): verified when
  the definition is stated and both transverse diameters are within 5 % (`exposures.tolerances.beam_diameter_rel`); a
  threshold at least the rated value when the footprint is no larger than the smallest reading; at most the rated value
  when the beam is larger than the largest reading across its narrower axis; not verifiable between the readings of a
  definition the rating does not state; no established relationship when a beam's axes straddle a stated diameter or
  beams lie on both sides. On a metallic reflector (`PowerRating.surface_kind`: a catalog product whose coating is a
  metallic reflector; a declared model with a metallic-reflector coating on a surface) a pulsed rating keeps a direction
  only as a single-pulse rating applied to one pulse per site; a reflecting surface of unknown kind leaves such a
  direction not verifiable. A `W/cm` rating applied to a larger beam has no established relationship. A beam whose size
  is not known makes it not verifiable (`BEAM_SIZE_UNKNOWN`), and so does light bounded by its diffraction limit (untraced
  light of unknown size, light left out as negligible) that may carry more than 1 % of the most light the surface can
  receive, unless the beams already give a threshold at most the rated value. The weakest beams that carry, with the
  followed light of unknown size, at most 1 % (a faint ghost of another size) are left out of the comparison; the light
  left out as negligible is then checked with them, so beyond that share it leaves the size not verifiable and never
  makes a faint ghost decide it. Both shares are measured in every
  quantity that gives the size its direction: the rating's own (peak irradiance, `P / d` -- both readings when the
  definition is not stated -- or peak fluence) and, for a CW rating, the power and `P / d`, for a pulsed one the peak
  fluence and the pulse energy; a beam whose chief ray misses the surface counts in each with the share of it its tail
  bound gives (`exp(-2 d^2 / w^2)`, `d` the miss distance, `w` its larger 1/e² radius), and a beam without a value in
  one of them is never left out;
* the pulse count `S` against the pulses one site of the surface receives (`design`, or `design_low` / `design_high`):
  each crossing of a source's light brings that source's `pulse.total_pulses` (the pulses it sends while the optics stay
  in place; `UNLIMITED` for a source that keeps pulsing as long as they are in use, more than any rated count; the lines
  of one source along one path are one crossing), and crossings that together carry at most 1 % of the most light the
  surface can receive (in the quantities above) do not count. Compared per tested site for a single-pulse rating or a
  count with `test_protocol` `S_ON_1` (each tested site received it at one fluence level): every site receiving `S`,
  verified; at least `S`, a threshold at most the rated value; at most `S`, at least the rated value (damage is
  irreversible); crossings that may bring a site fewer or more (or light left out as negligible, or stored light, that
  may bring more): `PULSES_PER_SITE_UNDETERMINED`, the message naming the cause; a source that states no `total_pulses`,
  beyond that share: `DESIGN_INPUT_MISSING`. A rating that states no count, or a count above 1 without its protocol:
  `RATING_INFORMATION_MISSING`, whatever `total_pulses` says. Continuous light: differs with no established
  relationship;
* the test polarization: `polarization_condition` written as `s-polarized` / `p-polarized` (also `S-Pol`,
  `s polarized`, `s`, `TE` / `p`, `TM`, ...), `linear`, `circular` or `unpolarized` (the Engine rating's `polarization`
  `S`, `P`, `LINEAR`, `CIRCULAR`, `UNPOLARIZED`) is compared with the light on the surface as a whole, from each beam's
  Jones state there. A rating whose `polarization_condition` names another state than its `polarization` (`TM` beside
  `S`, `circular (RHC)` beside `LINEAR`) is refused as input; one whose text says more than its field (`linear,
  vertical`) has that part not verifiable. It is verified when every beam has at most 1 % of its power outside one common
  state -- s or p in its own plane of incidence, one linear axis (the major axis of the strongest traced beam), one
  circular state (the strongest traced beam's, compared in the laboratory frame so a beam's own reflection back along it
  keeps it) -- since any incoherent mixture of such beams, and any sum of beams in one state, is then within it too.
  Mutually coherent beams in different states add their fields with a relative phase the trace leaves open, and each
  beam's own share does not bound their sum's: two equal coherent beams linear at +2° and -2° to one axis, each 0.49 %
  off it, sum to circular light for one phase. Their sum is checked for every phase, in field amplitudes `a` (square
  roots of the peak irradiance; also of the peak fluence for a `J/cm2` rating): its part outside the state is at most
  `U = Σ √f_i a_i` and its part in it at least `W = √(1 - f_j) a_j - Σ_{i≠j} √(1 - f_i) a_i` (`f` a beam's own share
  outside the state, `j` a traced beam), so at most `U² / (U² + W²)` of it is outside; in the state of one traced beam
  `j` it keeps at least `B = a_j - Σ a_i |e_j* · e_i|` and the rest is at most `D = Σ a_i √(1 - |e_j* · e_i|²)`, so at
  most `(B √f_j + D)² / (B² + D²)` -- the smaller bound counts. That bound holds at every point only for beams of one
  amplitude profile on the surface (the same centre and footprint; on a curved surface, which leaves its tangent plane,
  the same direction, radii and axes): beams displaced from each other, of different sizes or meeting the surface from
  different directions change their amplitude ratio across it, so their combined state varies there and is not verified
  (`COHERENT_STATE_UNDETERMINED`) unless none of them has a part outside the state. Beyond 1 % the polarization is not
  verifiable (`COHERENT_STATE_UNDETERMINED`, with that bound in the message); within it -- a beam with weak coherent
  ghosts in other states on its own profile -- it is verified and the note gives the bound. A violation at a traced
  beam's centre for every phase is checked first. Beams of another state that carry at most 1 % of the
  most light the surface can receive (in peak irradiance and every quantity that gives a comparison its direction)
  count in the exposure, not in the state, and the left-out light's share is checked with them entirely outside it.
  It differs only where
  the light at a traced beam's centre violates it whatever the phases of coherent beams and the amount of other light
  there: the least share a mixture there can have in the orthogonal state above 1 % (`UNPOLARIZED`: one polarization state
  carrying more than half the power plus 1 %; every traced beam is fully polarized). Two mutually incoherent beams in
  crossed linear states, each linear, do not hold `linear` (where they overlap their mixture is partly unpolarized) and do
  not violate it for certain: not verifiable. A polarization in other words, s or p at normal incidence (no plane of
  incidence; the rating states no test angle), beams in different states whose mixture neither holds the condition
  everywhere nor violates it for certain (mutually coherent beams of different states whose sum may leave the state for
  some phase), light of unknown state that may reach the surface (light of unknown fate; faint light not followed that
  may carry more than 1 % of the most light there, in peak irradiance or in the rating's own quantity) are not
  verifiable.

The windows and shares these conditions are matched in are settings of the run, stated in a Project's
`numerical_settings` and reported in force in `exposures.tolerances`:

| `exposures.tolerances` | Setting | Default | Decides |
| --- | --- | --- | --- |
| `wavelength_rel` | `rating_wavelength_tol_rel` | 0.02 | the relative distance from the rating's wavelength within which the light's matches |
| `pulse_duration_rel` | `rating_pulse_duration_tol_rel` | 0.10 | the same for the pulse duration |
| `repetition_rate_rel` | `rating_repetition_rate_tol_rel` | 0.10 | the same for the repetition rate |
| `beam_diameter_rel` | `rating_beam_diameter_tol_rel` | 0.05 | the same for both transverse diameters of a beam against a stated test beam |
| `polarization_fraction` | `rating_polarization_fraction_tol` | 0.01 | the largest share of a beam's power outside the rated polarization state |
| `unlocated_light_share` | `rating_unlocated_light_share` | 0.01 | the largest share of the most light a surface can receive that light of unknown size, place or state, and followed light the rating is not for, may carry before a condition is not verifiable |

Each condition records the window it was matched in (`conditions[].tolerance`). A comparison decided in a window other
than the default states the match on its `utilization` (`value_basis.utilization.validity[]`: `within` true, `setting`
the setting's name). `applicability` follows from the conditions, first match:
`NOT_APPLICABLE` (a condition differs with no established relationship, or two in opposite directions), `NOT_VERIFIED`
(a condition is not verifiable), `THRESHOLD_AT_LEAST_RATED`, `THRESHOLD_AT_MOST_RATED`, `AS_RATED` (every stated
condition verified; `conditions_match` is true for it only). `mismatches` and `unverified_conditions` list the messages
of the conditions that differ and of those not verifiable. The exposure in the rating's unit is the combined peak
irradiance (`W/cm2`), `P / d` with the rating's diameter definition (`W/cm`: 1/e² `2w`, FWHM `w √(2 ln 2)`, and from the
1/e² to the FWHM value when the definition is not stated; each beam's value weighted by its profile like the
irradiance) or the combined peak fluence (`J/cm2`), computed wherever the rating's quantity exists for the light on the
surface -- also for a rating of another wavelength, pulse duration or repetition rate, whose comparison stays
`CONDITIONS_MISMATCH` (a `J/cm2` rating has no exposure for continuous light); `notes` say
which definition and polarization the comparison used. `exposure_value` without a `value_basis` entry is the exact
exposure. Otherwise the exposure lies between `exposure_lower_bound` (`null`: nothing above 0) and `exposure_upper_bound`
(`null`: nothing bounds it), and `exposure_value` is the exposure of the followed beams alone (`null` where they
interfere with each other or that number lies outside the bounds) with the entry `value_basis.exposure_value`: kind
`MODEL` where only light left out as negligible is not counted (`omitted[].upper`: the most it can move the exposure),
`ESTIMATE` where light the trace does not follow, light of unknown fate or light that may return between surfaces may
reach the surface too. `utilization` is exposure / rated value. Without an entry it is decided: the exact exposure, or
the lower bound / rated value when that is above 1 (entry kind `LOWER_BOUND` where light not counted may add), for a
comparison that applies (as stated, at least or at most rated) with a value that establishes a threshold. Every other
comparison that reports an exposure states the same ratio with the entry `value_basis.utilization`:

| In the entry | Says |
| --- | --- |
| `assumed[]` `UNSTATED_CONDITION_AS_DESIGN`, `at` the condition names | the test conditions the rating does not state, taken as the light's own |
| the same with `at` `value_definition`, `value` `PEAK` | a value of unstated definition read as a peak value (read as `TOTAL_PER_1E2_AREA` the ratio to the threshold is half) |
| the same with `at` `rating_status`, `value` `TYPICAL` | a typical value read as the part's own |
| `validity[]` with `within` `false` | a condition the light differs in: `parameter` the condition, `observed` the light's value, `minimum` / `maximum` the rating's range |
| `validity[]` with `within` `null` | a condition that cannot be checked against the light |
| `omitted[]` | the light the exposure does not count, as on `exposure_value` |

Its kind is `ESTIMATE` unless every stated condition is verified and the exposure is the model's value. The status,
the applicability, the surface's and the element's `rating_status` and a target's assessment are decided by the bounds
and the conditions; none of them reads these two numbers.

The rated value is a damage threshold for the peak exposure (`rated_threshold_low` / `rated_threshold_high`, equal, in
the rating's unit; absent: the value establishes none). A `J/cm2` or `W/cm2` value stated as the test beam's peak
(`value_definition` `PEAK`) is the threshold; one stated as the total energy or power over the area of its 1/e²
diameter (`TOTAL_PER_1E2_AREA`) is half a Gaussian beam's peak, so the threshold is twice it; one whose definition is not
stated -- no catalog rating states it -- establishes none (condition `value_definition` `NOT_VERIFIABLE`,
`RATING_INFORMATION_MISSING`; the comparison is `CONDITIONS_NOT_VERIFIED`). A `W/cm` value is its own threshold. A `TYPICAL` value is a product's typical value, not one each part meets: it bounds
no threshold, and a note says so. Fluence and irradiance are compared in the beam's transverse plane; no rating states the
angle of incidence or the plane of its test. Status per comparison: `AS_RATED` -- `WITHIN_RATING` (the exposure, or its upper
bound, is at most `rated_threshold_low`), `EXCEEDS_RATING` (the exposure, or its lower bound, is above
`rated_threshold_high`), else `UNKNOWN`; `THRESHOLD_AT_LEAST_RATED` -- `WITHIN_RATING` or `UNKNOWN`;
`THRESHOLD_AT_MOST_RATED` -- `EXCEEDS_RATING` or `UNKNOWN`; a `TYPICAL` value -- `UNKNOWN`; `NOT_VERIFIED` --
`CONDITIONS_NOT_VERIFIED` (the exposure still reported, no verdict); `NOT_APPLICABLE` -- `CONDITIONS_MISMATCH` (never an
exceedance). A surface's utilization (exposure / damage threshold: lower bounds against `rated_threshold_high`, upper
bounds against `rated_threshold_low`) lies between the largest lower bound of its `AS_RATED` and at-most comparisons and, when it has an `AS_RATED` or at-least
comparison, the largest upper bound of those and of the comparisons not verified (a rating that may apply as stated may be
the stricter one). An at-most comparison -- and one not verified, which may have a threshold at most its value -- leaves
the surface no upper bound (its threshold may lie below the exposure) unless a comparison of the same unit that applies
as stated or with a threshold at least the rated value has a lowest threshold no larger than its highest under every
reading (the threshold then lies between the two; `W/cm` values are compared as P/d of the 1/e² diameter, a `FWHM` value
standing 1/0.5887 above the same threshold and one of unstated definition between); where an at-most comparison has only
such comparisons whose lowest may be larger the ratings disagree, as where the lower bound lies above the upper bound:
the lower bound stands and there is no upper bound; comparisons of different units are criteria of their own. Its status is `EXCEEDS_RATING` when the lower bound is above 1, `WITHIN_RATING` when the upper bound is
at most 1, else `CONDITIONS_NOT_VERIFIED`, `UNKNOWN` or `CONDITIONS_MISMATCH`; a surface no rating of its element names is
`RATING_MISSING`. An element's status and utilization come from its rated surfaces (those a rating applies to); its other
surfaces are listed in `unrated_surface_ids`. A rated surface no light reaches is a known zero only when the trace is
complete and no untraced light may reach it. An exceeded rating raises the notice `POWER_RATING_EXCEEDED` with the
numbers (listed in `manifest.warnings` too); its status is `NOT_APPLICABLE`, so it never changes the run's status (and
so never the Local Validation): an exceedance is reported, a requirement only through a target. The target metric `power_rating_utilization` (the element's
`device_id`) is the largest exposure / damage threshold: `max_utilization` when exact (no `value_basis` entry), else
graded on [`utilization_lower_bound`, `utilization_upper_bound`] (PASS only when the whole interval meets the target,
FAIL when its lower end does not), the target's `value` then being the largest exposure / damage threshold of the
ratings that apply -- the exposure exact, or its lower bound --, as a power target reports the traced light's value.
Where the utilization is not exact, the element's `max_utilization` is the largest reported exposure / damage threshold
(the rated value where it establishes no threshold) over the comparisons of the ratings not known to be for other light
(every comparison where all are), with the entry `value_basis.max_utilization`: that comparison's entry and what the
other rated surfaces leave open. A target on an element of which nothing is decided (no rating applies, none can be
checked, nothing bounds the utilization) reports that number as its `value` with the same entry under
`value_basis.value`; its status stays `UNKNOWN` and its assessment `INDETERMINATE`. The value is `null` when no
comparison reports an exposure (only light the trace leaves out as negligible or does not follow reaches the rated
surfaces, or beams whose fields may cancel), as a power target on a plane no traced beam reaches has
no value while light may arrive. The lower bound is the largest of the surfaces' lower bounds (an exceedance a rating
that applies decides stands, a FAIL; 0 where only ratings with a threshold at least the rated value bound a surface),
the upper bound the largest of their upper bounds; while a rated surface has none there is no upper bound, so the target
is not PASS (`POWER_RATING_CONDITIONS_NOT_VERIFIED` when a rating could not be checked,
`POWER_RATING_CONDITIONS_MISMATCH` when no rating applies there), and `UNKNOWN` without a lower bound. An element's
`rating_status` is the most severe label of its rated surfaces; the results are the comparisons. On a rated element,
`rating_status_counts` gives how many comparisons have each status, and where the utilization has no upper bound
`utilization_open_by` lists each reason with its surface and, where one comparison is the reason, that comparison
(`rating_index`, `applicability`): `NO_UPPER_RATING` (no rating there applies as stated or with a threshold at least its
value), `EXPOSURE_UNBOUNDED`, `NO_THRESHOLD`, `THRESHOLD_OPEN_BELOW` (a threshold at most the rated value, or a rating
not verified, that no applying rating of its unit bounds), `RATINGS_DISAGREE`. The target's
message names what makes the utilization an interval: ratings that bound it from one side only (their test beam smaller
or larger than the light), ratings that disagree, conditions not verifiable, the two readings of an unstated `W/cm`
diameter definition, or the exposure's own range. Power of
unknown fate may reach any element, so a maximum utilization is `UNKNOWN` in such a run -- including behind a tabulated
thin lens whose transmittance is below 1 (the rest of its response is not followed), for elements upstream of the lens
too. Ratings are compared as stated: no scaling between conditions, no thermal or damage model.

## Irradiance map

For the rays of an extended emitter, see "Extended emitters" below: a histogram with explicit bins. For beams: paraxial Gaussian beams with plane-wave phases on the plane (the model of the coherent superpositions): power flux
through the plane per unit area, incoherent (sum of the beams) and coherent (the degree of coherence from the beams'
relative group delay within a coherence group; `null` with `COHERENCE_UNDEFINED` when a pulse train states no coherence
or two beams crossed passes of undetermined group delay differently; a line that states no coherence is taken as
monochromatic, and the map's `value_basis` names that stand-in under `coherent_W_per_mm2`). Whole plane, no
wavefront curvature across the plane, no Gouy phase, no diffraction.

* **Several wavelengths.** With `wavelength_nm` the map is that line's. Without it, on a plane several wavelengths reach, the
  map is their sum on one grid (lines of different wavelengths are mutually incoherent: their irradiances add):
  `wavelength_nm` is null, `wavelengths_nm` lists them and `per_wavelength` gives each line's beams, grid powers,
  `coherent_reason`, `map_complete` and untraced light. The coherent map is the sum of the lines' coherent maps when every
  line has one. A detector reads `pixel_power_W` from such a map; photons and electrons need one wavelength
  (`WAVELENGTH_REQUIRED`).
* **Grid.** `grid.center_uv_mm`, `grid.half_extent_mm` and `grid.shape` reproduce the grid. A call that states no `samples`
  takes 101 points per axis, or, where the beams make fringes that count does not resolve and a count up to 1001 does, the
  smallest count that gives four points per fringe period (`sampling.samples_basis`: `DEFAULT`, `FRINGE_PERIOD` or `STATED`).
  Where no allowed count resolves the fringes, or the stated one does not, the coherent map is withheld
  (`FRINGES_UNDERSAMPLED`, with the period and the half width that would resolve it). State `samples` to fix the count.
* **Beams the map cannot hold** are left out and listed in `excluded_beams` (`path_id`, `crossing_index`, `uv_mm`, `reason`,
  `surfaces`, and what is known of the power): `POWER_OUT_OF_MODEL_SCOPE` (the beam carries its chief ray's power, not the
  light's: no power), `POWER_TAKEN_BEFORE_PLANE` (`power_lower_bound_W`, `power_upper_bound_W`) and `GAUSSIAN_NOT_AVAILABLE`
  (a power and no profile: `power_W`). The map of the other light stays, `map_complete` is false and the most the excluded
  beams carry is in `untraced_power_W` (null when one has no bound). Where no beam or line of the scope is left there is no
  map: `UNAVAILABLE` with that reason.
* **The limit is the caller's.** `max_untraced_fraction` (default 0.1; `null` / `none`: no limit; CLI
  `--max-untraced-fraction`) is the share of the map's light that may be missing from it above which a coherent map and a
  uniformity that light can move are withheld. The document states the limit applied beside `untraced_fraction`, and a
  detector response applies the same one.

## Focal field of a pupil mask

`os_optics.results.pupil_field(request, mask_files=None, handoff=None)`, `os-optics results pupil-field --request R.json [RUN --monitor ID] [--mask-file F]` and `json_api.pupil_field` take a pupil-plane field and a mask
(`os-optics schema pupil-field-request`; `PHYSICS_SCOPE.md`, P10e) to the back focal plane of an ideal lens. The request states `wavelength_nm`, `focal_length_mm`, `sampling` (`pixel_count`, `pitch_mm`, `fill_factor`, `fft_size`),
`aperture`, `illumination` (`UNIFORM`, `GAUSSIAN`, `BEAMS`, `HANDOFF_BEAMS`: the beams a run delivers to a plane, `path_ids` to choose among them), `mask` (`BLAZED_RAMP`, `BINARY_GRATING`, `SAMPLED`, `FILE` with its SHA-256, `SPOT_ARRAY`), `device_efficiency` and
`output` (`half_extent_mm`, `center_uv_mm`, `include_field`, `include_mask`).

The reply holds `status`, `power` (the ledger from the incident power to the focal window), `validity` (`pupil_na`, `field_sine`, `marginal_ray_sine`, `limit`, `in_domain`), `quantities` (each `MODELED`, `EXTERNALLY_CHARACTERIZED` or `UNAVAILABLE`),
`conventions`, `not_modeled`, and, in the layout of an irradiance map (`grid`, rows are v, columns u), `coherent_W_per_mm2` or `incoherent_W_per_mm2`, `grid_power_*_W` and `power_outside_grid_W`. A coherent result gives its field with
`include_field` (`field_sqrt_W_per_mm2`: `re`, `im`). `spots` lists each beam or target spot with `power_W` within `radius_mm` and `peak_W_per_mm2` (the largest focal sample there), and `array` the `efficiency` and `uniformity_power` /
`uniformity_peak` (`1 - (max - min) / (max + min)` of value / weight).

`status`: `OK`; `PARTIAL` above the model's validity limit, with the pupil stage and its ledger and the focal quantities `UNAVAILABLE` with their reason (an answer: exit code 0); `UNAVAILABLE` with a `reason` when no field can be built (exit code 2: `GRID_INVALID`, `NO_LIGHT`, `MASK_FILE_MISSING`,
`MASK_FILE_HASH_MISMATCH`, `MASK_FILE_INVALID`, `HANDOFF_REQUIRED`, `NO_BEAMS`, `BEAM_NOT_TEM00`, `BEAM_NOT_IN_AIR`, `PHASE_NOT_DEFINED`, `POWER_OUT_OF_MODEL_SCOPE` (a handoff beam whose power is out of model scope), `POWER_TAKEN_BEFORE_PLANE`, or `BEAM_POWER_UNKNOWN` with `spot_positions`: a grating or deflector with no stated efficiency says where its beams would focus). A request that is not valid is refused with `PUPIL_FIELD_INVALID`.

Handoff beams: a beam's radii and curvatures lie along its own two Gaussian axes, whatever their orientation on the plane (an elliptical beam turned against `u` and `v` included); one `COHERENT` beam without a phase (behind an ideal thin lens, a Faraday rotator or a grating) is computed with its constant phase taken as 0 (`phase_reference` `UNDETERMINED_CONSTANT`: no irradiance depends on it), while several coherent beams must each state one (`PHASE_NOT_DEFINED`). An `INCOHERENT` sum leaves out the beams no field can be built from (power out of model scope, part taken before the plane, no known power, not TEM00, not in air) and lists them: `excluded_beams` (`path_id`, `reason`, `surfaces`, `spot_uv_mm`, `power_W`), `illumination_complete` false and, in the layout a detector reads, `map_complete` false with `untraced_power_W` (null when one of them has no known power): each spot of the map is right, the map does not hold all the plane's light. A `COHERENT` sum refuses instead, naming the beams: a field left out could cancel the others. With no usable beam the reply is the refusal.

## Vector focal field of an objective

`os_optics.results.focal_field(request, handoff=None)`, `os-optics results focal-field --request R.json [RUN --monitor ID]` and `json_api.focal_field` take an ideal aplanatic objective and
the light on its pupil (`os-optics schema focal-field-request`; `PHYSICS_SCOPE.md`, P10g) to the vector field near its focus. The request states `wavelength_nm`, `objective`
(`IDEAL_APLANATIC`: `numerical_aperture`, `focal_length_mm` the effective focal length, `focal_medium_index`, `window`), `illumination` (`UNIFORM`, `ANNULAR`, `GAUSSIAN`, `BEAMS`,
`HANDOFF_BEAMS`), `polarization` (`LINEAR`, `CIRCULAR`, `JONES`, `UNPOLARIZED`; from the run for `HANDOFF_BEAMS`), `pupil_phase` (`ZERNIKE`), `sampling` (`theta_nodes`, `phi_nodes`,
`grid_cells`) and `output` (`half_extent_um`, `samples`, `z_planes_um`, `include_components`, `encircled_energy_radii_um`).

The reply is an analysis document (`os-optics schema analysis-result`): `identity` (the Engine build, the run's input and result hashes for a run's beams, `settings_hash`), `execution`,
`settings` (the request, the effective sampling, the method and the reference method), `models` (the vector Debye model with its validity checks; the paraxial Gaussian model for a Gaussian
beam), `quantities` (power ledger, energy balance, longitudinal share; per spot `peak_position_x/y/z`, `peak_electric_irradiance`, `fwhm_x/y/z`, `radius_1e2_x/y`,
`encircled_radius_50/80`, `power_within_disc_W`, `strehl_ratio`; `gaussian_waist_radius`), each with its `convergence`, and `arrays`: `energy_balance`, `encircled_energy` (the curve as a
share of the transmitted power), `reference` (the independent method's check) and `maps` (`planes` with `electric_irradiance_W_per_um2`, `axial_flux_W_per_um2` and optionally
`field_sqrt_W_per_um2`; `section_xz_...` and `section_yz_...`; rows are y or z). Several spots name their quantities `<label>.<quantity>`.

Outside the Debye model's validity the pupil ledger stays and the focal quantities are `OUT_OF_MODEL_SCOPE` (`DEBYE_APPROXIMATION_INVALID`); the analysis ran to its end, so its execution is `COMPLETED`.

Exit codes: 0 the analysis ran to its end (each quantity states its own state and reason), 1 refused (`FOCAL_FIELD_INVALID`, `HANDOFF_REQUIRED`, `NO_BEAMS`, `BEAM_NOT_TEM00`,
`BEAM_NOT_IN_AIR`, `PHASE_NOT_DEFINED`, `POLARIZATION_NOT_DEFINED`, `BEAM_POWER_UNKNOWN`, `POWER_OUT_OF_MODEL_SCOPE`, `POWER_TAKEN_BEFORE_PLANE`), 64 usage error.

Handoff beams: a beam's radii and curvatures lie along its own two Gaussian axes, whatever their orientation on the plane (an elliptical beam turned against `u` and `v` included); one `COHERENT` beam without a phase (behind an ideal thin lens, a Faraday rotator or a grating) is computed with its constant phase taken as 0 (`phase_reference` `UNDETERMINED_CONSTANT`: no irradiance depends on it), while several coherent beams must each state one (`PHASE_NOT_DEFINED`). An `INCOHERENT` sum leaves out the beams no field can be built from (power out of model scope, part taken before the plane, no known power, not TEM00, not in air) and lists them in `arrays.illumination` (`complete` false, `excluded_beams`): the quantities are those of the beams the field is built from. `arrays.phase_reference` states `UNDETERMINED_CONSTANT` for a single beam without a phase.

## Imaging analysis

`RunResults.imaging(monitor, request=None)`, `os-optics results imaging RUN --monitor ID [--request R.json]` and `json_api.run_results(files, {"kind": "imaging", "monitor", "request"})`
trace the run's fields again with a dense pupil grid (`os-optics schema imaging-request`; `PHYSICS_SCOPE.md`, P10h). The request states `fields` (source ids), `wavelengths_nm`,
`crossing_index`, `launch_grid`, `launch_radius_beam_radii` (the pupil, when stated), `max_launch_radius_beam_radii` (how far the search for the system's aperture grows the launch disc, default 16 beam radii; where
nothing cuts it the field is analysed on a disc of 2 beam radii and each number carries `value_basis` `ESTIMATE` / `UNIFORM_PUPIL` beside the state `UNKNOWN`, `PUPIL_NOT_BOUNDED`), `psf_samples`, `psf_half_width_um`, `frequencies_cyc_per_mm`, `reference` (`CENTROID`, `CHIEF`) and `include_psf_map`,
and what the analysis computes: `convergence` (default `true`: the analysis is done again with twice the rays per axis and every value is that finer grid's, with
its difference from the `launch_grid` value as its convergence; `false`: once, on `launch_grid` rays, the values that grid's and each value's `convergence.method`
`NONE`; on that coarser pupil a diffraction quantity can be `OUT_OF_MODEL_SCOPE` (`PUPIL_UNDERSAMPLED`) where the default computes it: raise `launch_grid` for a finer pupil) and `diffraction` (default `true`; `false`: the point spread function is not computed and the diffraction quantities -- `strehl_ratio`,
the MTF curves and the encircled-energy radii -- are `NOT_COMPUTED` with reason `NOT_REQUESTED`; `include_psf_map` needs it; a field without a wavefront or without its geometric quantities states them as the full analysis does). The doubled grid traces four times the
rays of `launch_grid`, and the point spread function with its peak search and encircled energy is most of the rest of the time (`benchmarks/bench_evaluation.py`, section
`analyses`). The settings name them only when they narrow the analysis.

The reply is an analysis document (`os-optics schema analysis-result`) bound to the run (`identity.physical`: the run's input and result hashes). Per field (quantity ids
`<source>#<spectral index>.<quantity>`, the line as in the run's path ids; its wavelength is in the quantity's `scope`): `image_space_na`, `geometric_spot_rms_radius`, `opd_rms_waves`, `opd_pv_waves`, `strehl_ratio`, `mtf_tangential` / `mtf_sagittal` and
`diffraction_limited_mtf_*` (arrays over `<field>frequencies_cyc_per_mm`), `geometric_mtf_*` (stated whenever the rays form one spot; below one wave RMS of wavefront error, or without a wavefront, the state is
`OUT_OF_MODEL_SCOPE` / `UNKNOWN` and the curve carries a `value_basis` of kind `ESTIMATE` with the regime check), `encircled_radius_50` / `_80` and `diffraction_limited_encircled_radius_80`, each with
its model basis (`models`: the geometric ray bundle, the geometric MTF and the scalar or vector Debye integral with their validity checks) and its convergence against twice the rays
per axis (with `convergence` true). `arrays` holds per field `rays` (launch disc, `pupil` -- `APERTURE` or `LAUNCH_DISC` -- and `pupil_search_radii`, rays arrived, path and crossing, reference point, axes, sphere radius), `psf` (window share of the light, half width, samples,
peak, Nyquist and cut-off frequencies, kernel; with `include_psf_map` the map in 1/µm²) and `encircled_energy`. `settings.random_seed` is null: the sampling is deterministic.

Model scope: where the run found a field's light outside an element's published model (a `model_scope_issues` finding whose `affected_arrivals` on the plane include the arrivals the field analyses: the line's strongest route at the plane, in both polarization components, at the `crossing_index` asked for, else its last crossing), every number of the field that reads what the finding touches is `OUT_OF_MODEL_SCOPE` (reason the finding's code), keeps its value, and carries a `value_basis` of kind `ESTIMATE` with the finding's validity checks of that light (`parameter`, `minimum` / `maximum` as published, `observed`, `within` false, `setting` of a single point's tolerance, `at` the element, `code`). A finding about a coating's or polarization element's coefficients leaves `image_space_na`, `geometric_spot_rms_radius`, `opd_rms_waves` and `opd_pv_waves` as they are. Another line on the same plane whose light the model covers keeps its quantities.

Execution: the analysis has no resource budget of its own, so it runs to its end (`COMPLETED`); a field or a quantity that is not available says so in its own state and reason (a requested field that is not a beam source: one `NOT_APPLICABLE` row).

Exit codes: 0 the analysis ran to its end, 1 an invalid request (`IMAGING_INVALID`), 64 usage error, unknown plane or unreadable run.

## Ghost paths

`RunResults.ghosts(monitor, request=None)`, `os-optics results ghosts RUN --monitor ID [--request R.json]` and `json_api.run_results(files, {"kind": "ghosts", "monitor", "request"})`
trace the run's request again with the reflection of every lens or window face followed as a ghost branch (`os-optics schema ghost-request`; `PHYSICS_SCOPE.md`, P10i). The request
states `fields` (source ids), `wavelengths_nm`, `max_bounce_order` (1-4, default 2), `relative_power_floor` (default 1e-9 of the line's input), `max_listed` (per line, default 64)
and the ghost trace's own `max_paths` / `max_trace_steps`, and `convergence` (default `true`: the trace is done again with twice the pupil rings and points per ring, and
the values are the finer trace's; `false`: one trace with the sources' own pupil sampling, each value's `convergence.method` `NONE` with its sampling resolution kept).

The reply is an analysis document (analysis `GHOST_PATHS`) bound to the run. Per line, `arrays["<source>#<spectral index>.ghosts"]` lists the ghost paths, strongest on the image
plane first: `path_id`, `bounce_order`, `bounce_surfaces`, `direction` (`FORWARD` for an even order, `BACKWARD` for an odd one), `termination`, `reaches_image_plane`,
`crossing_indices`, `focus_point_mm` and the ids of its quantities. Quantities (ids `<source>#<spectral index>.<path id>.<quantity>`, with `c<k>.` before the quantity when a path
crosses the plane more than once): `reflected_power_relative_to_line`, `end_power_relative_to_line` (a ghost that does not reach the plane), and per crossing `relative_power`
(over the primary light on the plane), `power_relative_to_line`, `focus_offset_mm` (the rays' least-squares focus from the plane, positive beyond it), `waist_offset_x_mm` /
`_y_mm` (the Gaussian waists; `OUT_OF_MODEL_SCOPE` with the Gaussian state's reason where it is not valid), `spot_rms_radius_um` and `irradiance_ratio` (Gaussian peak irradiance,
or the geometric spot-area estimate: its `basis` says which). Per line also `primary_power_relative_to_line`, `primary_spot_rms_radius_um` and the bound
`unlisted_power_upper_relative_to_line` / `_to_primary` (`BOUNDED`; a `PROVEN` interval, or `NOMINAL` where a ghost surface met a beam without a valid Gaussian state, an
annular beam or a sag surface and its reflectance is bounded over the beam's pupil rays only). `completeness` states what was enumerated, lists every piece of light not followed
(`not_followed`: `BOUNCE_LIMIT`, `BELOW_FLOOR`, `PATH_BUDGET`, `OUT_OF_MODEL`, `SCATTERED`, `UNMODELLED_RESPONSE`, `UNTRACED_<kind>`, `NOT_LISTED`, `POWER_NOT_KNOWN`, each
with its power bound) and carries `unenumerated_power_upper_W`, the bound on the power every unlisted ghost brings to any plane (`proven` says whether it is proven), with every
ghost surface's reflectance and transmittance bounded over the beam's angles of incidence; `arrays["<line>completeness"]` breaks it down and counts the ghost-surface meetings
whose angles are sampled. Powers are ratios: no absolute radiometry. Each quantity carries its convergence against twice the pupil rings and points per ring (with
`convergence` true), and a power on the
geometric pupil-sample basis its sampling resolution (`convergence.resolution`).

Model scope: a run's model-scope finding that lists an arrival of a line on the plane marks every power, ratio and bound of that
line; one that reaches the line anywhere and names an element a ghost bounces at marks that ghost's numbers, and every finding
that reaches the line marks the bound on the ghosts not listed. A marked number reads `OUT_OF_MODEL_SCOPE`, keeps its value
and carries the findings' `ESTIMATE` in `value_basis`; focus, waists and spot are marked only by a finding that touches the
geometry. `arrays["model_scope"]` lists the findings applied.

Execution: `COMPLETED` when the enumeration ran to its end, `PARTIAL` (`BUDGET_EXHAUSTED`) when the ghost budget ran out. What the enumeration could not state is not an
execution state: light that ended outside the model, scattered light and light an element's response left undelivered are listed in `completeness.not_followed` with their
bounds; a power out of model scope, a missing primary and a requested field that is not a beam source are their quantities' own states and reasons; `completeness.proven`
says whether the bound holds for all of it.

Exit codes: 0 the enumeration ran to its end, 2 partial (the ghost budget ran out: `execution.budgets`), 1 an invalid request (`GHOSTS_INVALID`), 64 usage error, unknown
plane or unreadable run.

## Detector response

`detector_response(irradiance, detector, composition=None)`, `os-optics results detector --detector D.json (RUN --monitor ID | --field FILE)` and `json_api.detector_response` read an irradiance document (of a run or of a focal field) with a stated
detector (`os-optics schema detector-model`). Every parameter is optional; what is not stated leaves the quantities that need it `UNAVAILABLE`, naming them in `missing`, and the others unchanged (`PHYSICS_SCOPE.md`, P10e, has the table).
State `0` for a dark current or a read noise you take as negligible.

The reply holds `status`, `inputs` (each parameter as stated, with its source, or `UNAVAILABLE`), `pixels` (the array: `u_mm`, `v_mm`, `shape`, rows are v, `active_width_mm`, `integration`), `quantities` (each with `status`, `unit`, `basis`, and `value` or `values` per pixel),
`unavailable_quantities` and `totals`. An irradiance document with two different maps needs `composition` (`COHERENT` or `INCOHERENT`). A document without a map gives `status` `UNAVAILABLE`, `IRRADIANCE_UNAVAILABLE` with the `upstream_reason`. A document of status `OK` that lacks a member the response reads (`grid` with `u_mm` and `v_mm`, a map of `len(v_mm)` rows of `len(u_mm)` values) or holds one of the wrong form is refused with `IRRADIANCE_INVALID`, the member named (the command exits 64).

A map says what it holds. `scope` is its wavelength and source, `outside_scope` the other light on the plane with its known power and upper bound. `untraced_power_W` is the most power of the scope that is not in the map and may reach the plane (a beam of known place and unknown power, listed in `omitted_beams`; an order not followed; a ghost; null: nothing bounds it), `untraced_fraction` that power over the known power, `map_complete` that it is zero, and `interference_open` that it may interfere with the map's beams. A beam outside the plane's acceptance cone does not count. A plane only beams of unknown power reach gives `BEAM_POWER_UNKNOWN` with the beams.

A plane that does not absorb may have no bound (`power_upper_bound_W` null). Light of unknown fate may go on crossing it, as in a cavity: nothing bounds that, `untraced_power_W` is null, values have no upper end and `saturated` reads `true` only where the known light alone reaches the full well. Where the only open question is light left out as negligible, `absorbed_power_upper_bound_W` counts it once, and the map uses that. A detector judges each quantity against that power, taken whole on one pixel: a pixel of known power `p` holds between `p` and `p + U`, or between `(sqrt(p) - sqrt(U))^2` and `(sqrt(p) + sqrt(U))^2` where the light may interfere. Values are `BOUNDED`, `bound` `LOWER` or `INTERVAL`, with `upper` and `lower`. `saturated` and `adc_clipped` are `true` when the lower end reaches the limit, `false` when the upper end stays below it and nothing lies outside `scope`, and null otherwise. `snr` and a `uniformity` keep their values with a stated range (`relative_error_bound`, `roi_untraced_fraction`) up to the map's `max_untraced_fraction` (default 0.1: there the field of that light is 0.32 of the known field, and interference moves a pixel holding all the known light by 0.47 to 1.73 times, about a factor of two). Above it they are `UNAVAILABLE` or null, with the reason, the fraction and the limit; and where that light may also interfere, the coherent map is withheld (`coherent_reason`) and everything that reads the light is `UNAVAILABLE`, unless `composition` `INCOHERENT` states the beams add in irradiance. A map made with a larger `max_untraced_fraction` (`irradiance_map(..., max_untraced_fraction=...)`, `--max-untraced-fraction`) holds its coherent sum and its uniformity, and the detector quantities are stated with their two ends, which hold at any fraction; made with no limit where nothing bounds that light, a pixel lies between 0 and no upper end and `snr` stays `UNAVAILABLE`. The reply passes on `excluded_beams` and `max_untraced_fraction`. A ghost of an uncoated lens is about 0.002 of the light and keeps every value. State the element's efficiency to close the range, or read one source with `source_id`; the quantities are then of that source, and `saturated` reads `true` or null, because the camera sees the other light too.

With `electrical` (a stated bandwidth and dark-noise floor, `PHYSICS_SCOPE.md` P9d) each pixel is also a photodetector element:
`photocurrent_A` (`QE q lambda / (h c)` times its power; it needs no exposure), `electrical_noise_current_A` (its shot noise and the
floor in the noise-equivalent bandwidth) and `electrical_snr`, bounded like `snr` on a map that does not hold all the light, and null
where the pixel's photocurrent is above a stated `electrical.saturation_current_A` (as `snr` is at a saturated pixel). A detector
that does not state `electrical` gives none of them: its reply is the camera response as before.

## Detection on an observation plane

A plane that states `detection` reports `ObservationResult.detection` in every run: `model`, `power_state`, `contributions` (each
source at each wavelength the plane counts: `power_W`, `modulated_fraction`, `responsivity_A_per_W`), `quantities` (each with
`name`, `status`, `unit`, `value` or `reason` and `missing`, `basis`: `detector_bandwidth_Hz`, `noise_equivalent_bandwidth_Hz`,
`detected_power_W`, `photocurrent_A`, `dark_noise_density_A_per_rtHz`, `shot_noise_current_A`, `dark_noise_current_A`,
`total_noise_current_A`, `detector_snr`, `noise_equivalent_power_W_per_rtHz`, and with a lock-in `modulation_frequency_Hz`,
`lock_in_enbw_Hz`, `modulated_photocurrent_A`, `lock_in_signal_A`, `lock_in_noise_A`, `lock_in_snr`,
`detector_response_at_modulation`; where the plane's light holds a sample's sampled rays, `sampling_relative_sigma`: the
quantity's relative standard error from the lines' `collection_fraction_sigma`, an upper guide; on a lit plane whose power
is `BOUNDED` or `UNKNOWN`, `value_basis` with the key `value`: the number is computed from the traced light, and the entry
gives its kind and the parts the plane's power leaves out, `PHYSICS_SCOPE.md` P9d) and `validity` (each with
`parameter`, `value`, `limit`, `comparison`, `within`, `meaning`: `modulation_frequency_within_detector_bandwidth`,
`modulation_periods_per_time_constant` with a lock-in, `peak_photocurrent_within_linear_range` where the detector states
`saturation_current_A`). With a first-order response the lock-in's signal and noise are those at the detector's output
(times `|H(f)|`, `detector_response_at_modulation`), in photocurrent units. `os-optics results detection RUN
[--monitor ID]`, `RunResults.detection(monitor)` and json_api `run_results` kind `detection` print it. `os-optics results
sample-emission RUN` (kind `sample_emission`) prints how each fluorescence or Raman sample's line powers follow from its
excitation (`PHYSICS_SCOPE.md` P9c).

## Source spectra

`ValidationResult.source_spectra` is present when a Gaussian-beam source states a continuous spectrum (`spectral_density`;
`PHYSICS_SCOPE.md`, P2), one entry per Gaussian-beam source: `DISCRETE` (its lines as stated, `line_count`) or `SAMPLED_CONTINUOUS`
with the `shape`, `band_nm`, the `quadrature` (`MIDPOINT_BINS`), `line_count` and `line_width_nm` (the bin width, each line's
linewidth), the spectrum's `centroid_nm` and `rms_width_nm`, its `coherence_length_mm` (the smallest path difference with
`|γ| = 1/e`), the same of the traced lines (`coherence_length_lines_mm`) and of twice as many lines
(`coherence_length_doubled_lines_mm`), and `gamma_max_difference` / `gamma_max_difference_doubled`: the largest difference between the
lines' `|γ|` and the spectrum's over `0 … convergence_range_mm`. `reason` says why a length has no value. Each line's beams carry
their own `coherence_length_mm` (its bin width as linewidth); `power_by_wavelength` lists every quadrature line.

## Extended emitters

The rays of an `EXTENDED_EMITTER` source (`PHYSICS_SCOPE.md`, P8a) have their own results; beams keep theirs.

* `ValidationResult.sampled_sources`: per spectral line the emitted power, area, angular law, the measure and size of the sampling
  domain, the derived radiance or intensity, the sample count, rays traced, seed, sequence id and the weight of one ray.
* `ObservationResult.sampled_light`: per line on the plane `power_W`, `collection_fraction` (`power_W / emitted_W`),
  `collection_fraction_half_samples` (the same fraction from the first `sample_count / 2` rays alone, a stratified sample by itself) and
  their difference, which shows how far the estimate moved when the rays doubled, and `collection_fraction_sigma`, the standard error if the
  rays were independent random samples (the stratified rays usually do better; two strata can also agree by chance, so read both).
  `crossings` holds one row per crossing.
* `total_power_W`, `power_by_wavelength`, `PowerBudgetLine.observed_W` and a `power_W` target include the counted rays.

```
os-optics results sampled-light RUN_DIR --monitor D
os-optics results irradiance RUN_DIR --monitor D --samples 40 --half-extent-mm 20 --center-uv-mm 0,0 \
  --roi '{"shape":"circle","center_uv_mm":[0,0],"radius_mm":10}'
os-optics export sampled-crossings RUN_DIR --monitor D --format csv --out crossings.csv
```

The irradiance map of sampled rays is a histogram: `grid.u_edges_mm` / `v_edges_mm` are the bin edges, `grid.u_mm` / `v_mm` their centres,
`grid.bin_area_mm2` the area, `incoherent_W_per_mm2` the power crossing each bin per unit area of the plane and `sample_counts` the
crossings behind each bin (a bin of `n` crossings has a sampling noise of at most about `1 / sqrt(n)`). `coherent_W_per_mm2` is null with
`coherent_reason` `INDEPENDENT_SAMPLES`. `power_outside_grid_W` is the crossing power beyond the grid. Gaussian beams of the same wavelength
add at the bin centres; `source_id` selects one source's light. Without `center_uv_mm` and `half_extent_mm` the grid covers the active area,
or is centred on the crossings' power centroid and holds 95 % of their power; without `samples` it takes about 16 crossings per bin.

`roi` adds `uniformity`, over the cells whose centres lie in the region: `min_over_mean` (smallest / mean), `std_over_mean` (population
standard deviation / mean), the mean, smallest and largest irradiance, the cell count, `mean_crossings_per_cell`, and `half_samples` (the
same two numbers from the first half of the rays, so the distance between them shows how much of the spread is sampling noise).
The `sampled-crossings` export gives the same numbers: bin `power_W` by `(u_mm, v_mm)` over the same edges and divide by the bin area.

A complete emitter:

```json
{"instance_id": "LED", "optical_role": "source",
 "custom_part": {"custom_part_key": "led", "description": "4 x 2 mm LED die."},
 "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
 "source_model": {"model_kind": "EXTENDED_EMITTER", "wavelength_nm": 550, "power_w": 0.1,
                  "emitting_surface": {"shape": "RECTANGLE", "width_mm": 4, "height_mm": 2},
                  "angular_law": "LAMBERTIAN", "sampling": {"sample_count": 4096, "seed": 1}}}
```

Export kinds: `sampled-light` (one row per plane and line), `sampled-crossings` (one row per crossing). Query kinds: `sampled_light`,
`irradiance` with `source_id`, `center_uv_mm`, `roi`.


## Field propagation (sampled-field and speckle sources)

`os-optics results field RUN --monitor ID [--request field.json] [--field-out field.npz]` (`RunResults.field`, json_api
`run_results(files, {"kind": "field", "monitor", "request"})`; request schema `os-optics schema field-request`) propagates the
coherent field of a `SAMPLED_FIELD` or `SPECKLE` source to an observation plane and returns an analysis document (`AnalysisResultV1`,
analysis `FIELD_PROPAGATION`). Request: `source_id` (omitted: the run's only field source), `method` (`AUTO`, `ANGULAR_SPECTRUM`,
`PARAXIAL_TRANSFER`), `padding` (the transform window in source-grid extents, default 2), `region` (`center_uv_mm`, `half_width_mm`;
default the output window, which depends on the method: state a region to compare a mean irradiance or a contrast between
methods), `realizations` (the first N of a speckle source), `crossing_index`, `refinement` (attach the window
refinement, realization 0 on a window twice as wide, as convergence records; the window conditions are checked either way),
`include_field`, `value_tolerance` (the accuracy asked of the plane's values: the largest change of them the window refinement and the
rays' aberration may make, default 0.01, at most 0.1; a request that states another value has it in the settings, the two checks
name it and each plane value carries a `value_basis` of kind `MODEL` with them), `map_points` (default 256).

Quantities (`<source>.<name>`, scope plane, source and wavelength): `geometric_power_W` (the field's rays on the plane), `power_window_W`,
`power_region_W`, `mean_irradiance_W_per_mm2`, `peak_irradiance_W_per_mm2`, `centroid_u_mm`, `centroid_v_mm`, `rms_radius_u_mm`,
`rms_radius_v_mm`, `contrast` (`speckle contrast of one pattern` for a speckle source, with a one-standard-error `SAMPLING` interval)
and, with two or more realizations, `contrast_averaged` and `contrast_averaged_independent` (`mean(C_m) / sqrt(M)`); a speckle source
adds `screen_phase_rms_rad` and `unscattered_power_share` (`|<exp(i phi)>|^2` over the illumination). A single pattern's peak,
centroid, rms widths and region values carry the window refinement as a `REFINEMENT` convergence record. The rms widths need 8 grid
points of the region along their axis and the contrasts 8 along both, else they are `NOT_APPLICABLE` (`TOO_FEW_POINTS`). A plane
outside the model's conditions has every plane quantity `OUT_OF_MODEL_SCOPE` with the first failed condition as reason and every
failed condition in `depends_on`
(`WINDOW_LOSES_POWER`: light leaves the output window or the band limit cuts it; `WINDOW_NOT_CONVERGED`: a window twice as wide
changes the plane's values by more than `value_tolerance`; `COUPLED_TRANSFER`, `NOT_SYMPLECTIC`, `NOT_PARAXIAL`, `APERTURES_CUT_FIELD`:
the field's own rays are cut, or the elements transmit them differently than along the axis (a clear aperture that only cuts the
Gaussian tail of the probe beam giving the axis's transmission does not count: that cut is undone);
`ABERRATED_BEYOND_FIRST_ORDER`: the rays' aberration beyond first order, applied to the field as a phase, changes the plane's
powers, irradiances, widths or contrast by more than half of `value_tolerance` or its centroid by more than `value_tolerance` of its rms width, `GRID_INVALID`, `NOT_FREE_SPACE`, `PLANE_NOT_PARALLEL`, `TRANSFER_UNAVAILABLE`); a source axis that
does not reach the plane is `PLANE_NOT_REACHED` (the probe follows the source's axis: a field away from the axis that passes an
opening the axis does not is refused this way), and one whose power is out of model scope there (`power_out_of_scope_at`) leaves the field
`OUT_OF_MODEL_SCOPE` (`POWER_OUT_OF_MODEL_SCOPE`). Arrays: `method`, `realizations`, `grid` (the computed grid's points and pitch),
`map_u_mm`, `map_v_mm`, `irradiance_mean_W_per_mm2` (`[v][u]`, block means, zero outside the region), `contrast_per_realization`,
`power_region_per_realization_W`, `conventions`, `speckle_fully_developed` (a speckle source: `exp(-sigma^2)` of the screens' rms
phase at most 0.01) and, with `include_field`, `field_u_mm`,
`field_v_mm`, `field_re`, `field_im` (realization 0 over the region, sqrt(W/mm^2)). Where the rays' aberration changes the complex
field by more than 0.1 relative rms (`aberration_field_error`) the field arrays are left out and `field_withheld` states the check,
its observed value and its limit; the plane's values are those of the same request without `include_field`. `--field-out` writes them as an `.npz`
(`field_re`, `field_im`, `u_mm`, `v_mm`, `pitch_mm`), which `os-optics results pupil-field` takes as a `FIELD` illumination
(`--mask-file`) on a pupil grid of the same points and pitch. Model scope: where a run's model-scope finding lists the field's line on
the plane (`affected_sampled_lines`), every quantity with a number is `OUT_OF_MODEL_SCOPE` (reason the finding's code), keeps its value
and carries a `value_basis` of kind `ESTIMATE` with the finding's checks of that line; `arrays.model_scope` lists the findings (code,
element, the kinds they touch, their checks), the `.npz` holds them as `model_scope_json`, and a pupil field lit by that file lists them
under `model_scope` with their checks, as one built from a handoff beam. A field source whose light the model covers keeps its
quantities. The model and its conditions: `docs/engine/PHYSICS_SCOPE.md` P8b.

## Runs of many beams

**Pairs of independent sources.** Two beams of one wavelength from mutually incoherent sources (different coherence groups,
such as an array of independent lasers) form no lasting fringe: their irradiances add. The run counts those pairs per plane
in `simulation_result.json` `incoherent_pairs` (`observation_id`, `pair_count`, `coherence_group_count`) with one
`INCOHERENT_SOURCES` issue per plane (status `NOT_APPLICABLE`, `pair_count` in its context), and lists in `interference` the
pairs that can interfere -- every same-wavelength pair of one coherence group -- plus every pair a pair-metric target selects
(two independent sources named by `source_id` / `source_id_b`: the pair is analysed, and `fringe_visibility` is 0) and every
pair whose beams arrive in media of different index. Only the listed pairs count against `max_interference_pairs`. A plane
with 625 independent beams lists none of its 195,000 pairs. A pair-metric target that names neither a source nor a path
selects every pair of its plane, so the run lists them all.

**Ray data recording level.** `ray_data.json` records every chief and pupil ray by default (about 100 kB per beam path with
the default 96 pupil rays). `run_optical_project(..., ray_data="summary")` or `os-optics project run --ray-data summary`
records the chief rays only, with each path's pupil sampling (`pupil.ray_count`, `ring_count`, `points_per_ring`,
`power_weight_per_ray`) but no `pupil.rays`: about 1 % of the size. The trace and its sampling are the same, so the result,
its hashes, the Local Validation record and the 3D view are those of a full run; `ray_data.json` states
`"recording": "summary"`, and so do the manifest's `ray_data` entry and `describe()["available"]["ray_data_recording"]`.
`results paths` and everything read from the Engine result work as before; `results rays` and `results wavefront` refuse a
summary run with `PUPIL_RAYS_NOT_RECORDED` -- run the Project again with `--ray-data full` for them.

**Memory.** The run directory is written one file at a time, each serialized and hashed as it is written, with the long
arrays of the result and the ray data produced element by element; the result hash is computed the same way. The bytes are
those of `run_artifact_files`. `benchmarks/bench_scale.py` measures wall time, peak memory and file sizes per phase for the
625-beam core, `m8-o1` and the return-light analysis case (`m8-line`).

**Progress, cancellation and resource limits.** A run passes through named phases (`os_optics.run_control.RUN_PHASES`):
`lower` (steps: each component lowered, each optical body and each pair of bodies checked), `preflight` (before the trace,
and again when it ends), `trace` (beam paths), `emitters`, `observation_results` (arrivals), `untraced_light` (probe paths),
`return_analysis` (work units, inside `bounds` and `exposures`), `bounds`, `exposures`, `interference` (beam pairs),
`targets`, `result` (the result, its notices and hashes), `ray_paths` (beam paths), `model_scope` (paths), `record`,
`scene` (beam paths), `write` (bytes) and, for a JSON API caller, `reply` (the run's summary and documents). A `RunMonitor`
(`os_optics.run_monitor`; `project run` and the JSON API install one) reports each phase's start and end and, at most
every half second, its count, the elapsed time and the process's memory; its checkpoints are deterministic and change no
number. It stops a run when the caller cancels it (`monitor.cancel()`, Ctrl-C, a host's Stop button), when the memory
read at a checkpoint is at or above `max_memory_bytes` (read at most every 0.1 s, so a short allocation between two reads can
pass it), or before the run files pass `max_artifact_bytes`. Before the trace it refuses a run whose estimate exceeds a
limit (beams x the per-beam memory and file sizes measured on 2500 independent beams, two paths each, 8 x 12 pupil rays,
`os_optics/local/resources.py`, from the memory once the Project is lowered; a scene whose beams split more or meet more
surfaces needs more). When the trace ends it estimates the run again from the beam paths and steps the trace counted and
the Project's optical elements (`estimate_after_trace`: measured runs wrote 0.66 to 1.42 x the estimate; the range
stated runs from the bytes the run writes for certain -- the Project and Engine request files, and each path and step of
the result at no fewer bytes than its shortest record, `certain_run_file_bytes` -- to 1.5 x the estimate, and memory from
the memory then up to 1.6 x its estimated growth + 32 MiB) and refuses the run only when the low end of that range
exceeds a limit: the run would write at least that much, or already used that much memory. The checks during the run hold whatever the estimates said; `RunMonitor(preflight=False)`,
`--no-preflight` or `resource_limits.preflight` false report the estimates and let the run go on. Progress reports,
`RunMonitor.summary()` and a stopped run's `progress` carry them (`estimate`, `estimates`: `stage` `before_trace` /
`after_trace`), each with how it was made: `method` (`PER_BEAM_REFERENCE` before the trace, `COUNTED_PATHS_AND_STEPS`
after it), `reference` (the measured runs its costs come from, a key of `os_optics.local.resources.REFERENCES`),
`measured_ratio_low` / `measured_ratio_high` (the run files measured runs wrote, as multiples of their estimate's
`artifact_bytes`: 0.57 to 234 before the trace, where the estimate is a bound in neither direction; 0.66 to 1.42 after
it), `memory_platform` (whose memory costs it uses: `CPYTHON` or `BROWSER_WASM`; null without a memory estimate) and
`lower_bounds` (the fields that are counted, not fitted: the run needs at least that much). `artifact_bytes_low` is
counted only for a monitor with a run-file limit that refuses on its estimates, and is null otherwise. In a browser the memory
is the Engine's WebAssembly memory and the per-beam cost before the trace the one measured there (625 such beams in
Chrome). A stopped run has no result: `RunStopped.stopped` (`StoppedRun`) states
execution `CANCELLED` (termination `CANCELLED`), `stop` (`cause` `CALLER` / `MEMORY_LIMIT` / `ARTIFACT_LIMIT`, `phase`,
`phases`, `completed_phases`, `done` / `total` / `unit`, the limit and the measured or estimated bytes), every observation
and target in `affected_observation_ids` / `affected_target_ids`, and the monitor's `progress` (seconds per phase, peak
memory read, run-file bytes written). Its record is written into the run directory it was given (`run_stopped.json`).

**The run directory rule.** A run that completes replaces the run in its directory; an attempt that fails or stops
before the run is moved into place leaves the directory as it found it; one that fails while the run is moved into place
leaves the Engine's records, by which any run then completes there.

* A completed run writes every file to a temporary name beside its own (`.<name>.partial`) and moves the run into place
  once all are written: each manifest the move removes is first copied beside it (`.manifest.json.earlier`), the stop
  record and the manifest there are removed, the files are renamed (`project.json`
  first), what the earlier run held and the new one does not write is removed by name (a `configurations/<id>/` run
  directory, `comparison.json`, the files of a single run where a run of configurations is written), `manifest.json`
  is moved last and the copies are removed. A process killed during these steps past the first, a step that fails
  there (a file held open elsewhere) or Ctrl-C there leaves no manifest, which is no run, and
  the copies, which list by name what the earlier run still holds there; the same run started again, or any other run,
  completes there and removes by name what the earlier run and the killed commit (its staged manifest) left there and it
  does not write. The first step removes, in this order, the run files of each configuration that did not run in the new
  run, the stop records, the configurations' manifests and the directory's own. Stopped while those run files are
  removed (one held open elsewhere, a kill, Ctrl-C), it leaves the directory's `manifest.json` and that configuration's,
  without the run files removed before: `project verify` reports `INCONSISTENT` (exit 66) and names them as missing.
  Stopped while the configurations' manifests are removed (a configuration's `manifest.json` held open elsewhere), it
  leaves the directory's `manifest.json` beside a configuration whose manifest is gone, recorded by its copy
  (`configurations/<id>/.manifest.json.earlier`): `project verify` reports `INCONSISTENT` (exit 66) and names the
  missing configuration manifest. Either way any run, the same one or the earlier design, then completes there.
* A run replaces only what the Engine wrote. A file at a run-file name (`manifest.json`, `engine_request.json`,
  `simulation_result.json`, `ray_data.json`, `local_validation.json`, `optical_scene.json`, `optical_scene.glb`,
  `comparison.json`; under a run of configurations, the same names in each `configurations/<id>/` it writes) is replaced
  when the Engine manifest there (or its copy a killed commit left) lists that name, or a staged manifest a killed
  commit left (`.manifest.json.partial`, kept as `.manifest.json.interrupted-<n>` by the next run that stages its own
  there, so a second interruption loses no record; removed when a commit completes there) does; a `manifest.json` must be an Engine manifest; a `configurations/<id>/` that
  holds a run must be named by the directory's manifest. The check reads the manifests only: no file of an earlier run is
  read or hashed. A file the
  Engine cannot show it wrote -- the person's own result file, another tool's output, a run kept under
  `configurations/<id>/` -- refuses the write (`RunDirectoryOccupiedError`, an `OSError`; `project run` exits 64, the file
  named) before anything is written, and stays. A file of the earlier run that a completed run could not remove (held
  open by another program) stays and is recorded by name in `.manifest.json.leftover`, beside the run; the next run
  there, the same one or another, accepts it and removes it. A name the commit did remove is not recorded: a file put
  there later is the person's own. `run_stopped.json`, the copies, the kept staged manifests, that record and the
  temporary names are the Engine's own. Every other file in the directory is left alone.
* `project.json` is also the name of a design file, so it keeps one protection: it is replaced when it is missing, holds
  the Project being run (the same `ProjectDocumentV2` payload, whatever its formatting; it is written again in canonical
  form) or is listed with its present SHA-256 by the Engine manifest there (or its copy) or by the staged manifest of a
  killed commit. A `project.json` holding another design that
  no manifest lists refuses the write (`RunDirectoryOccupiedError`, an `OSError`; `project run` exits 64, the file named)
  and stays.
* A run never writes through a link: if the run directory itself, `configurations/`, a `configurations/<id>/` it
  writes, a run-file name or its temporary name is a symbolic link or a junction, the write is refused
  (`RunDirectoryLinkError`, an `OSError`; exit 64), and the link and its target stay as they were. Links are never
  followed when an earlier run's files are removed.
* An attempt that does not complete before the run is moved into place -- a usage error, a Project that is refused, a
  failure, a stop in any phase or while the files are written, Ctrl-C -- removes its own temporary files and changes
  nothing else: an earlier run there still verifies (`project verify` answers `CONSISTENT`) and still loads. A failure
  or Ctrl-C while the run is moved into place (a file held open elsewhere) leaves what a process killed there leaves:
  past the move's first step, no manifest, which is no run (`project verify` and `results` find no run to read), the
  copies (`.manifest.json.earlier`) and the staged `.manifest.json.partial`. At the first step it leaves instead the
  directory's `manifest.json`: while the run files of a configuration that did not run in the new run are removed (one
  held open elsewhere), beside that configuration's `manifest.json` and without the run files removed before, which
  `project verify` (`INCONSISTENT`, exit 66) names as missing; while the configurations' manifests are removed (a
  configuration's `manifest.json` held open elsewhere), beside a configuration whose manifest is gone, recorded by its
  copy (`configurations/<id>/.manifest.json.earlier`), which `project verify` (`INCONSISTENT`, exit 66) names. Either
  way any run, the same one or the earlier design, then completes there. A refused attempt writes nothing; its reply and exit
  code state it. A stopped run adds `run_stopped.json`, and the readers of the kept run name it: `project verify`,
  `load_run(...).describe()` and `json_api.run_results` give `later_attempt` (`execution`, `cause`, `phase`,
  `project_hash`, `same_project`), and the `results` commands print one line on stderr. A directory holding a stop record
  and no manifest holds no run (`verify_run_artifacts` says the run written there stopped).
* Whether a kept run is current for a Project is `run_freshness` (`results freshness RUN --project P`): the manifest binds
  the Project hash and the Engine input hash of the run it describes, so a run kept behind a later attempt on an edited
  Project reads `HISTORICAL` for that Project.

`json_api.run_project` takes
`resource_limits` (`max_memory_mib`, `max_artifact_mib`) and answers a stopped run with `ok` false, `error.code`
`RUN_STOPPED` and `stopped`.

## Runs of a Project with configurations

A Project with `configurations` (zoom positions, switched states) runs each configuration as the ordinary Project
`configured(id)`. The run directory holds `manifest.json` (schema `opensci.local_configurations_run` 1.0: the Project hash,
the run's `engine_input_hash` -- `hash_exact({"configurations": [[id, engine_input_hash], ...]})` over the configurations
run, in the Project's order --, per configuration its `directory`, `project_hash`, `engine_input_hash`, `result_hash`,
`validation_status`, `execution` and `problems`, and the SHA-256 of its run directory's manifest), `project.json` (the
Project with its configurations), `comparison.json` and `configurations/<id>/`, an ordinary run directory of each
configuration that ran, byte for byte the run of `configured(id)` on its own. `load_run` and every `results` command read a
configuration's directory; given the top directory they answer `CONFIGURATIONS_RUN` and name the configuration
directories. A configuration whose Engine execution failed has no directory and is listed `FAILED` with its problems.

`comparison.json` is an analysis document (`AnalysisResultV1`, analysis `CONFIGURATIONS`): `arrays.configurations` (per
configuration: label, directory, input and result hashes, validation status, execution record, requirement record,
problems), `arrays.targets` (per target and configuration: whether it applies, its value, status and assessment) and the
quantities, each scoped to its configuration: `power_W` per plane with its quantity state and the run's evidence (a `BOUNDED` power
carries the run's lower and upper bounds, a proven interval unless the power rests on pupil samples; a sampled power carries its
sampling resolution, which decides no threshold within it), and per arriving beam `position_u_mm`, `position_v_mm`, `beam_radius_x_mm`, `beam_radius_y_mm` and
`spot_rms_mm` (quantity id `<configuration>|<plane>|<path>|<crossing>|<name>`). With a paraxial request (`os-optics results
configurations RUN --paraxial IDS --wavelength-nm W`, `json_api.configurations_comparison(files, paraxial)`), `efl_mm` and
`bfl_mm` per configuration (`NOT_APPLICABLE` / `AFOCAL` for an afocal system; a configuration the paraxial analysis refuses
carries the refusal code, `OUT_OF_MODEL_SCOPE`). Its `execution` is `COMPLETED` when every configuration completed, else
`PARTIAL` naming the configurations that did not; the exit code of `results configurations` is then 2.

## Not modelled

In a run: diffraction beyond the Gaussian beam model (no PSF / MTF / diffracted fields), the Gouy phase in `phase_rad`, depolarizing
elements and the fringes of a partially polarized mixture, scattering, ghost reflections that are not traced (their power appears as
`UNTRACED_REFLECTION`), and coherent sums beyond the paraxial Gaussian superposition. The focal field of a pupil is computed apart from a run, in its
scalar low-NA domain (*Focal field of a pupil mask*). A vector focal field is not computed.
