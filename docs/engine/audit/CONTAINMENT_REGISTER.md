# Containment register

A containment is a value the Engine withholds because no computed or qualified value can stand. This register lists every
one in the Engine: the places where the tracer marks a beam's power as not the light's, and the tests that hold each
case. The rule it serves is in
[DEVELOPMENT.md](../DEVELOPMENT.md), section "Findings and tests": compute the value, else report it qualified
with its `value_basis`, and withhold it only where no value can stand, with the physical reason.

How the register is kept:

* **One row per site and per test.** Section 1 has one row for every method of `optics/rays/tracer.py` that adds a
  surface to `_Packet.power_scope` (exported as `ArrivalResult.power_out_of_scope_at`). Section 2 has one row for every
  test function of `tests/test_closeout_containments.py`, `tests/test_out_of_scope_power_results.py` and
  `tests/test_out_of_scope_orders_and_conversions.py`. `tests/test_containment_register.py` fails when a site or a test
  has no row, or a row names a site or a test that no longer exists.
* **The change that fixes a row edits the row** in the same commit: the state, what is reported, and the test's new
  assertion. The test keeps its scene and its independent reference and asserts the reference value within the stated
  error (and the `value_basis` entry of a qualified value). No test is removed.
* **A row that stays withheld** states the physical reason here and in the test's docstring.

States:

| State | Meaning |
|---|---|
| `computed` | The value is reported and the test compares it with the reference. |
| `qualified` | The value is reported with what bounds it (an interval or a bound in the result's fields); the test checks that the reference lies inside. |
| `withheld` | No value is reported. The row names the physical reason and the open work that would fix it. |
| `control` | The test's case next to the finding where the value was always right; it stays computed. |

A test function with several cases has the state of each case in its row.

## 1. Sites: where a beam's power stops being the light's

Each site adds the surface to `_Packet.power_scope`. Behind it the arrival still carries the chief ray's power in
`beam.power_W` with the surface in `power_out_of_scope_at`; the plane's `quantity_state` is `OUT_OF_MODEL_SCOPE`
(`basis_of(observation, "total_power_W")` gives kind `ESTIMATE`, part `BEAM_PART_ELSEWHERE`, at the surfaces), and every
quantity computed from the beam's power states no value: power targets, the irradiance map, the detector response, the
handoff power, pair and coherent-sum powers, grating-order powers, Stokes parameters, harmonic power, ghost powers,
exposure hits, the observed row of the power budget.

Site S-13 is the one `qualified` site: a coating met by the chief ray outside its data also joins the packet's `held`
list, and the arrival's `beam.power_W` and `detected_power_W` carry a stored `ESTIMATE` entry with one
`HELD_AT_NEAREST_STATED_POINT` assumption per held surface and parameter (the data's range as a failed check) and
`BEAM_PART_ELSEWHERE` for any other surface in `power_out_of_scope_at` (`validation/observations.py::held_basis`). Its
finding is a notice. Only the run trace holds coating data this way; sampled rays, untraced-light probes, ghost, imaging
and field analyses end the path at such a coating (`Tracer._live_coating`, `hold_coating_data` false).

Two neighbouring mechanisms are not sites: a retarder, Faraday rotator or attenuator whose response at the line is not
stated passes the light on with its power not computed and bounded by what arrived (`Tracer._pass_not_computed`),
and a share taken beside the chief ray that has a bound travels with the beam as an interval (`_Packet.taken_path`).
Both report what bounds the value instead of marking the power out of scope.

| Row | Method | What the chief ray's power still counts, or misses | Bound known at the site | Open work | State |
|---|---|---|---|---|---|
| S-02 | `Tracer._observation_event` | A surface takes or turns part of the beam before a plane the chief ray crosses (`_taken_before`), and at least one counted share has no supported bound: an unbounded tail, a surface without a support integral, or a beam without a Gaussian state whose start surface reaches beyond the plane. | No. Where every counted share is bounded the plane is already `qualified` (`power_taken_before_*`, state `BOUNDED`) and the site is not reached. | Open: a bound of the share an unbounded tail takes and of the share a surface without a support integral takes; without a Gaussian state, power without a valid Gaussian state. | `withheld` |
| S-03 | `Tracer._tail_beside_chief` | The beam's tail may meet a surface beside the chief ray that no pupil ray meets, before the surface the chief ray meets next. | Where the tail share is bounded (`tail_fraction_upper`, `tail_power_upper_W` in the issue); not for a tail the bound does not cover. | Delivered for the bounded share (it travels with the beam, `_Packet.taken_path`: the planes behind read `[P - S, P]`); the unbounded tail stays with the observation-plane residuals. | `withheld` |
| S-04 | `Tracer._met_beside_obstruction` | Pupil rays of an annulus (a beam behind a central obstruction) meet another surface first, beside the obstruction's support; the annulus's Gaussian share still counts their light. | No: only the share of the pupil samples (an estimate, not a bound). | Open: power without a valid Gaussian state. | `withheld` |
| S-05 | `Tracer._obstruct` | A curved obstruction is met at an angle to its axis: the share stopped is computed for the support mapped onto the plane at the chief ray's hit, which is not the shadow of the sagged surface along the beam. | No: that share is neither the share stopped nor a bound of it. | Open: the annulus geometry; the share itself needs the shadow of the sagged support along the beam. | `withheld` |
| S-06 | `Tracer._met_beside_chief` | Pupil samples of a Gaussian beam meet another surface first on the way to the surface the chief ray meets (for example the back of a secondary mirror beside a decentred beam). | For a plane surface, and a sag surface met along its axis: the Gaussian integral over the surface's support (`_support_share_beside`). Otherwise no. | Delivered where a bound of the share meeting the surface exists (it travels with the beam, `_Packet.taken_path`); the rest stays with the observation-plane residuals. | `withheld` |
| S-07 | `Tracer._gradient_side` | Part of the beam reaches the side of a gradient-index rod (pupil samples, or the envelope beyond them) while the chief ray leaves through the end face; what the side does with that light is not followed. | No: the share of the pupil samples is an estimate. | Open: a bound of the share that reaches the side. | `withheld` |
| S-08 | `Tracer._twins_mixed` | Two coherent crystal waves of one field (ordinary and extraordinary) meet a surface that mixes them: a second split, or a surface that passes the two polarizations by different powers (a polarizer, an oblique interface, a grating; `_twins_polarized`). The light behind it is the sum of their fields with a relative phase the trace does not carry; the sum of their powers is no bound of it. | No. | Open: the pair is to be summed as fields, with the phase from the two waves' optical paths. | `withheld` |
| S-09 | `Tracer._second_harmonic` | A crystal's second-harmonic conversion is outside the model's conditions (a pulse train, a mixture of lines, a fundamental without a Gaussian state, a fundamental whose own power is out of scope): the harmonic and the depletion of the fundamental are not computed. | No. | Open; not scheduled. A crystal that is not a slab is already reported in the same per-crystal report (`OUT_OF_MODEL_SCOPE`, no conversion computed). | `withheld` |
| S-10 | `Tracer._do_complex_transmission` | A transmission map's power factor `A^2` varies across the beam more than the quadrature resolves, or across a beam without a Gaussian footprint. | No. | Open; not scheduled. | `withheld` |
| S-11 | `Tracer._evanescent_share` | A transmission map turns part of the beam beyond grazing and `A^2` varies across the live pupil rays. | No. | Open; not scheduled. | `withheld` |
| S-12 | `Tracer._do_neutral_density` | An attenuator's density varies across the beam's footprint by more than `ND_FOOTPRINT_OD_TOLERANCE`: the chief ray's transmittance is not the share the light keeps. | No bound; the value itself is computable (the Gaussian profile weighted by `10^-OD`, as the transmission-map element weighs `A^2`). | Open: weight the Gaussian profile by 10^-OD across the footprint. | `withheld` |
| S-13 | `Tracer._hold_coating` | The chief ray meets a coating outside the wavelengths or angles its data covers; the beam goes on with the data's values at the nearest stated point. | No: the coating states nothing there. | Delivered: the power behind is reported as an estimate (`value_basis`: `HELD_AT_NEAREST_STATED_POINT`, the data's range as a failed check). | `qualified` |

## 2. Tests: one row per finding held by a test

Columns: the finding and the wrong value that was reproduced; what the Engine reports today; the reference the test
holds that does not use Engine code; the open work or what was delivered; the state.

### `tests/test_closeout_containments.py`

| Row | Test | Finding and the wrong value | Reported today | Independent reference in the test | Open work | State |
|---|---|---|---|---|---|---|
| T-01 | `test_an_absorbing_plane_reads_its_detected_power_within_the_share_a_surface_takes_before_it` | An absorbing plane just before a dump or fold detected the whole beam (1 mW) with no flag while the surface takes up to 42 % of it first. | `detected_power_W` with `power_taken_before_at`, `power_taken_before_upper_W` at least the share taken, no certain share; plane `BOUNDED`, `power_lower_bound_W` at or below the truth. | The share a straight edge takes of a Gaussian beam, `0.5 erfc(sqrt(2) y0 / w)`. | None needed. A tighter `S` belongs to the observation-plane residuals. | `qualified` |
| T-02 | `test_a_lens_beside_the_chief_ray_before_a_dump_leaves_no_certain_share` | The dump's certain share counted lines a lens beside the chief ray turns away first: a false `UNMET` at 0.995 of the truth. | `[P - S, P]` with both surfaces named and no certain share; the target is not `UNMET`. | The product of the two edge shares (closed form). | None needed. | `qualified` |
| T-03 | `test_a_beam_outside_the_acceptance_cone_adds_no_share_taken_to_the_plane` | A beam outside the plane's acceptance cone had its taken share subtracted from the plane's reading. | The plane reads beam A's 1 mW to 1e-12; both minima `MET`. | The source power of the one beam the plane accepts. | None needed. | `computed` |
| T-04 | `test_a_monitor_after_a_fold_without_a_gaussian_state_is_out_of_scope_where_the_mirror_reaches_beyond_it` | A monitor 0.25 mm after a fold read the whole beam as exact while 17.8 % of it leaves the mirror beyond the monitor. | Arrival with `M/face` in `power_out_of_scope_at`, target `OUT_OF_MODEL_SCOPE`; a monitor beyond the mirror's reach reads the beam (`MET`). | The share stated in the docstring (17.8 %); the control reads the full beam. | Open: power without a valid Gaussian state. | `withheld`: the beam has no Gaussian state (twisted astigmatism), so its power is its pupil samples' share, and near a waist the samples lie well inside the beam and do not sample the part that leaves beyond the monitor. Control: `computed`. |
| T-05 | `test_a_dump_beside_the_chief_ray_that_no_pupil_ray_meets_bounds_the_plane_and_the_beam_behind` | The tail beside a waist. No pupil ray meets a dump that 23 % of the beam's energy-flow lines meet; the plane behind read 1 mW exact (a 0.9 mW minimum `MET`). | Plane T: `[P - S, P]` with `S` above the reference share, `BOUNDED`. Plane T2 behind the next fold: `[P - S', P]` with `S'` the bound of the share the dump may take on the leg to the mirror, which holds `P (1 - share)`; the minimum not `MET`. Dump 1.5 mm farther off: both planes `KNOWN`. | `_tail_reference()`: the beam's energy-flow lines followed to the disc's plane, mass by a 2-D midpoint rule. | Delivered: the bounded share travels with the beam. | T, T2: `qualified`. Control: `computed`. |
| T-06 | `test_an_obstructing_absorber_with_a_hole_proves_no_later_hole_dark` | Light through an absorber's hole leaves through a fold mirror's hole in line with it; the plane read the whole annulus, 1.7 % high (a 7.40e-4 W minimum passed against 7.31e-4 W). | The annulus is not followed past the holed mirror (`ANNULAR_BEAM_NOT_FOLLOWED`); the plane states no value. | The arriving power stated in the docstring (7.31e-4 W). | Open: the annulus geometry. | `withheld`: the annulus's power is a Gaussian share beside one support; with a hole in that support and a second holed surface behind it the share that passes both is not that integral. |
| T-07 | `test_a_curved_obstruction_met_off_its_axis_puts_the_annulus_power_out_of_scope` | A convex absorber cap met 20 and 35 degrees off its axis: the stated share on its support was off the shadow's by +1.8 % and +4.6 % and was carried as a lower bound (a 6.9e-4 W minimum passed against 6.74e-4 W). | At 20 and 35 degrees: `DUMP/dump` in `power_out_of_scope_at`, plane and target `OUT_OF_MODEL_SCOPE`. Along the axis: unchanged. | The arriving power stated in the docstring (6.74e-4 W). | Open: the annulus geometry; site S-05. | Off axis: `withheld`: the shadow of a sagged support along a tilted beam is not the support mapped onto the plane at the chief ray's hit, and the mapped share is no bound of it. Along the axis: `control`. |
| T-08 | `test_a_plane_bounded_by_light_taken_before_it_states_no_map_handoff_power_or_detector_response` | On a plane bounded by light taken before it, the irradiance map stated the whole beam (1 mW against 0.765 mW arriving) and the detector response and the handoff power followed. | Map `UNAVAILABLE` (`POWER_TAKEN_BEFORE_PLANE`, surfaces and paths named); detector response unavailable through it; handoff power null with geometry and Gaussian parameters kept; a field built from the handoff is refused. | The tail scene's reference share; for the control the Gaussian integral over the map's grid. | Open: profile quantities on a bounded plane. | `withheld`: the profile on the plane is not the unclipped beam's, and a clip of share s can move the peak by about 2 sqrt(s). Control: `computed`. |
| T-09 | `test_parallel_tilted_plates_close_together_take_nothing_from_each_others_legs` | The tail check weighed two close parallel plates by balls about as large as the plates and counted a share: a 0.5 mW minimum 100 mm on was undecided. | The plane reads the beam with no share taken and no scope mark; `MET`. | Geometry: the first plate's faces lie behind the plane of the second's back face. | None needed. | `computed` |
| T-10 | `test_a_crystal_wave_plate_before_a_polarizer_states_no_power_behind_it` | Crystal twins at a polarizer. A calcite plate at 45 degrees before a polarizer read 3.62e-4 W at every thickness (the sum of the two waves' powers); the fields' sum is about 1e-7 W at the half wave and 7.24e-4 W at the full wave. | Both arrivals carry `POL/element` in `power_out_of_scope_at`; both power targets `OUT_OF_MODEL_SCOPE` with no value; the issue is at `surface:POL/element`. | The Jones-calculus powers stated in the docstring. | Open: coherent crystal waves summed as fields; site S-08. | `withheld`: the two waves are one coherent field and the polarizer passes the sum of their fields; the trace carries no relative phase between them, and the sum of their powers is no bound of the result. |
| T-11 | `test_a_crystal_plate_with_its_axis_along_the_polarization_and_a_displacer_without_a_polarizer_keep_their_powers` | Controls of T-10, and a displacer before a polarizer. | Axis along the polarization: one wave, targets decided. Displacer without a polarizer: two beams, powers stated. Displacer before a polarizer: `POL/element` in `power_out_of_scope_at`. | Fresnel transmission of the plate's faces (`fresnel_t(N_O)`), to 1e-12. | Open for the last case: coherent crystal waves summed as fields. | Controls: `control`. Displacer before a polarizer: `withheld`: whether the two separated beams still overlap at the polarizer is not tested; separated beams add as powers, overlapping ones as fields. |

### `tests/test_out_of_scope_power_results.py`

Results computed from a beam whose power is out of model scope were stated as valid values.

| Row | Test | Finding and the wrong value | Reported today | Independent reference in the test | Open work | State |
|---|---|---|---|---|---|---|
| T-12 | `test_every_result_of_a_plane_behind_a_varying_attenuator_is_out_of_model_scope` | Behind an attenuator of OD 0 to 8 across the beam the chief ray's transmittance (1e-4) was the stated power; the light keeps 1.94e-4. | Plane `OUT_OF_MODEL_SCOPE` naming `ND/attenuator`; map, detector response, handoff power, pupil field and exports state no value. Uniform attenuator: every one stated. | `_passed_share`: the Gaussian profile integrated against `10^-OD(x)`; the Gaussian integral over the map's grid. | Site S-12. | `withheld`: the chief ray's transmittance is not the mean transmittance over the beam's profile. Control: `computed`. |
| T-13 | `test_a_ghost_analysis_states_no_power_of_light_whose_power_is_out_of_model_scope` | Ghost powers, ratios and the bound of the ghosts not listed rested on the chief ray's power behind the varying attenuator. | Those quantities `OUT_OF_MODEL_SCOPE` with no value; focus offsets stated. Uniform attenuator: stated, except the two back-reflected ghosts that end beyond the attenuator's coated region. | The uniform case's ratios are positive `KNOWN` values; no closed form of a ghost power is held. | Site S-12. Delivered: the analysis states its own execution outcome (`execution`, `COMPLETED`). | `withheld`, as T-12. Control: `computed`. |
| T-14 | `test_a_pulsed_harmonic_has_no_irradiance_map` | The map of a pulsed second harmonic was the continuous-wave one (3.27e-6 W), about 1.5e5 times too low. | Plane `OUT_OF_MODEL_SCOPE`; map `UNAVAILABLE` naming `X/front`. Continuous-wave source: map stated. | The Gaussian integral over the map's grid (continuous-wave case). | Site S-09. | `withheld`: a pulse's conversion depends on its peak power and temporal shape, which the continuous-wave model does not carry. Control: `computed`. |
| T-15 | `test_a_beam_leaving_a_gradient_index_rod_through_its_side_has_no_irradiance_map` | 44 % of a tilted beam reaches the rod's side; the map behind the rod read 9.28e-4 W, 82 % above the light that arrives. | Map `UNAVAILABLE` naming `G1/S2`. At 0.45 rad (no ray at the side): map stated. | `test_gradient_index._side_share`: RK4 rays of the beam's Wigner function. | Site S-07. | `withheld`: what the rod's side does with the light that reaches it is not followed. Control: `computed`. |
| T-16 | `test_light_scattered_out_of_a_beam_whose_power_is_out_of_model_scope_is_out_of_model_scope` | Light scattered out of a beam behind a varying attenuator carried the scattered share of the chief ray's power; its plane read `BOUNDED` with a value. | The scattered line names `ND/dump`; plane and target `OUT_OF_MODEL_SCOPE`, no value, no map. Fixed OD 2: stated. | A Lambertian scatterer's share on a coaxial disc, `a^2 / (a^2 + h^2)` = 0.2. | Site S-12. | `withheld`, as T-12. Control: `computed`. |
| T-17 | `test_the_coherent_sum_and_the_pair_quantities_of_a_beam_out_of_model_scope_are_withheld` | The coherent sum and the two-beam power quantities read the powers of a beam behind a varying attenuator. | Pair status `OUT_OF_MODEL_SCOPE`; visibility, balance, irradiances and powers null; coherent sum null. Crossing angle and fringe period stay. | Geometry: the crossing angle and `lambda / sin(alpha)`; for the control the sum of the two powers. | Site S-12; open: the pair quantities behind a bounded share. | Power quantities: `withheld`, as T-12. Geometry and control: `computed`. |
| T-18 | `test_a_control_sensor_behind_a_varying_attenuator_has_no_reading` | A control sensor behind the varying attenuator read the chief ray's power, about half the light's. | The reading is unknown. Uniform attenuator: the beam's power. | `_passed_share`, to 1e-4. | Site S-12. | `withheld`, as T-12. Control: `computed`. |

### `tests/test_out_of_scope_orders_and_conversions.py`

A result that reads the power of a beam whose power is out of model scope was stated as a
value.

| Row | Test | Finding and the wrong value | Reported today | Independent reference in the test | Open work | State |
|---|---|---|---|---|---|---|
| T-19 | `test_a_crystal_fed_by_a_beam_whose_power_is_out_of_model_scope_converts_nothing_stated` | A dump beside the beam takes 30 % of it before a doubling crystal; the conversion was computed from the chief ray's full 1 W (`shg_power_W` and `shg_conversion_efficiency` `PASS`), at least twice the light's. | Conversion `OUT_OF_MODEL_SCOPE` naming `DUMP/dump`; no harmonic power or efficiency; three targets `OUT_OF_MODEL_SCOPE`; the harmonic beam is traced. Dump 4 mm beside: the closed form's value, `PASS`. | `_disc_share` (2-D Gaussian integral over the disc); `gaussian_near_field` (coupled-wave closed form, `P2 ~ P1^2`). | Delivered: the bounded fundamental; the harmonic of a bounded fundamental is site S-09. | `withheld`: the beam that enters is weaker by the share taken and no longer Gaussian, and the harmonic grows with the square of its power. Control: `computed`. |
| T-20 | `test_grating_orders_of_a_beam_whose_power_is_out_of_model_scope_state_no_power` | Behind the same dump each grating order stated the table's 0.3 / 0.6 of the chief ray's 1 W; the light carries at most 0.21 / 0.42 W. | Each propagating order `OUT_OF_MODEL_SCOPE`, no power and no bound, efficiency and direction stated, `DUMP/dump` named. Without the dump: `P x efficiency`. | `_disc_share`; the order efficiencies of the absolute table. | Delivered: the beam is bounded; the orders stay with the observation-plane residuals. | `withheld`: the beam carries the bound of the share the dump takes, but not how that share divides between the orders. Control: `computed`. |
| T-21 | `test_an_acousto_optic_deflector_behind_a_varying_attenuator_states_no_order_power` | Behind an attenuator of OD 0 to 4 across a 0.5 mm beam each order's power from the chief ray is 15 % low (1e-2 against 1.18e-2 of the beam). | Every order `OUT_OF_MODEL_SCOPE`, no power; the run summary marks each arrival; the 3-D view states no footprint or path power. Uniform attenuator: `KNOWN`. | `_lognormal_share`: the mean of `10^-OD` over the Gaussian profile (closed form). | Site S-12. | `withheld`, as T-12. Control: `computed`. |
| T-22 | `test_partially_polarized_light_behind_a_share_taken_states_no_stokes_sum_and_the_plane_its_interval` | Behind the dump the Stokes parameters of partially polarized light, the plane's bounds and the power budget's observed power were stated from the chief rays' powers. | The plane reads `[P - S, P]`, which holds `0.7 P`; the wavelength's power carries the bound as its entry; the budget observes the plane's value; the mixture's Stokes parameters `OUT_OF_MODEL_SCOPE` naming the dump; the crossing outside the active area names the dump. | The Stokes vector of partially polarized light after a mirror, `(P, 0.5 P, 0, 0)`; `_disc_share`. | Delivered for the plane and the budget. | Plane and budget: `qualified`. Stokes parameters: `withheld`: the shares the dump took of the two polarization components are not known. Control: `computed`. |
| T-23 | `test_light_scattered_out_of_a_beam_whose_power_is_out_of_model_scope_names_it_as_a_source_and_in_the_ledger` | The source row of light scattered out of a beam behind a varying attenuator stated the chief ray's share as its emitted power, and the budget observed it. | The source row names `ND/dump`; the budget observes no power of it. Fixed OD 2: `0.1 x 0.01 x P`. | The fixed attenuator's transmittance times the scattered share. | Site S-12. | `withheld`, as T-12. Control: `computed`. |

## 3. Open rows by the work that would close them

| Work | Rows |
|---|---|
| What a bounded share taken beside the chief ray leaves open behind it (delivered: the share travels with the beam) | S-03, S-06 (unbounded shares), T-19, T-20, T-22 (Stokes parameters) |
| Bounds that do not exist yet (an unbounded tail, a surface without a support integral, the side of a rod) | S-02, S-07, T-15 |
| Power without a valid Gaussian state, the annulus | S-04, S-05, T-04, T-06, T-07 |
| Profile quantities on a bounded plane | T-08 |
| Coherent crystal waves summed as fields | S-08, T-10, T-11 (last case) |
| Not scheduled | S-09, S-10, S-11, S-12, T-12, T-13, T-14, T-16, T-17, T-18, T-21, T-23 |
