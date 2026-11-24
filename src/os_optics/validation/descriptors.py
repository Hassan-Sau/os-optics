"""The public metric descriptors: one ``MetricDescriptor`` per target metric (``metric_descriptors()``).

``validation/metrics.py`` evaluates the metrics; this module states what each one means. A test keeps the two registries
in step (every evaluable metric has exactly one descriptor).
"""

from __future__ import annotations

from ..contracts.metrics import MetricDescriptor

OBS_SELECTORS = ("observation_id", "source_id", "wavelength_nm", "parameters.path_id", "parameters.crossing_index")
PAIR_SELECTORS = ("observation_id", "source_id", "source_id_b", "wavelength_nm", "parameters.path_id_a", "parameters.path_id_b",
                  "parameters.crossing_index_a", "parameters.crossing_index_b")
PLANE_FRAME = "observation plane frame (u = plane local x, v = plane local y, n = plane normal; see observations[].plane)"
BEAM_FRAME = "the beam's own transverse Gaussian axes (beam.gaussian.e1 / e2: x = e1, y = e2), not design-frame axes"
GAUSSIAN = "paraxial Gaussian beam (complex q per transverse axis, ABCD / Coddington) carried with the chief ray"
GAUSSIAN_VALIDITY = ("withheld (OUT_OF_MODEL_SCOPE) when the Gaussian state is invalid: high NA, strong surface, large AOI on a curved "
                     "surface, severe truncation, twisted astigmatism, or pupil rays spreading more than gaussian_max_ray_spread_ratio "
                     "times the aberration-free bound (GAUSSIAN_ABERRATION_DOMINATED). A valid Gaussian state with a WARNING "
                     "(a truncation, a divergence near the limit) gives the target that status and the state's code; a metric "
                     "that does not read the Gaussian state takes neither")
CHIEF = "real chief ray of the beam path (exact 3D ray trace)"
PUPIL = "the beam's deterministic pupil sample rays (equal-power Gaussian rings), traced exactly"
JONES = "Jones calculus of the fully polarized chief-ray field in its transported transverse basis"
PARTIAL = ("for a partially polarized source, the light on the beam's route: the incoherent sum of its two polarization "
           "components' Stokes parameters (ObservationResult.polarization_mixtures); either component's path id selects the route")
GROUP_DELAY = ("group delay of the chief ray from its source: group optical path sum n_g L (n_g = n - lambda dn/dlambda at the "
               "line's vacuum wavelength) / c; the delay of a wave packet's envelope (dispersive, not the phase delay); pulse shape "
               "and dispersion broadening not modelled")
DELAY_VALIDITY = ("UNKNOWN (GROUP_DELAY_UNDEFINED), with the delay of the modelled path as an ungraded value (value_basis: ESTIMATE, "
                  "UNSTATED_GROUP_DELAY_ZERO at the passes), where a pass on the way has no stated group delay (an element without a thickness, "
                  "a coating without a stated phase) or a medium's group index is not available; a pass whose coating states a "
                  "group-delay bound widens the value to an interval the target is graded on")
FIRST_ORDER = "first-order ray-transfer matrix from the source plane, by central differences of differential rays traced with the chief ray"
TWO_BEAM = "analytic two-beam interference: plane-wave phases of the chief rays with paraxial Gaussian envelopes on the plane"
TWO_BEAM_ORDER = ("a = the beam selected by parameters.path_id_a, else by source_id (when source_id_b names another source); "
                  "otherwise a is the beam with the lexicographically smaller path_id (interference[].path_id_a)")

#: Behind a promoted reference ray (``validation/metrics.py``, ``REFERENCE_RAY_METRICS``): what a chief-ray metric reads.
_PROMOTED = (" Behind a clear aperture that took the chief ray (reference_ray_promoted_at) it is that of the passing pupil ray that "
             "carries the beam's reference, and a target on it is out of model scope (CLIPPED): its value is that ray's "
             "(value_basis: ESTIMATE, SINGLE_RAY_FOR_BEAM) and is not graded.")
_CENTROID = (" spot_centroid_u_mm / spot_centroid_v_mm give the centroid of the pupil samples that reach the plane: the beam's position "
             "within the samples' resolution (behind a promotion where few samples pass, a fraction of the beam's rms width).")


def _obs(name: str, unit: str, definition: str, *, sign: str = "", frame: str = "", basis: str = CHIEF, validity: str = "",
         aggregation: str = "single_beam") -> MetricDescriptor:
    return MetricDescriptor(name=name, unit=unit, kind="observation", definition=definition, sign_convention=sign, reference_frame=frame,
                            selectors=OBS_SELECTORS, aggregation=aggregation, model_basis=basis, validity=validity)  # type: ignore[arg-type]


def _gauss(name: str, unit: str, definition: str, *, sign: str = "") -> MetricDescriptor:
    return _obs(name, unit, definition, sign=sign, frame=BEAM_FRAME, basis=GAUSSIAN, validity=GAUSSIAN_VALIDITY)


#: Every two-beam metric compares two beams of one wavelength.
SAME_WAVELENGTH = ("both beams must have the same wavelength; beams of two sources that arrive at different wavelengths give "
                   "NOT_APPLICABLE (WAVELENGTH_MISMATCH, naming both)")


def _pair(name: str, unit: str, definition: str, *, sign: str = "", operands: str = "", validity: str = "") -> MetricDescriptor:
    return MetricDescriptor(name=name, unit=unit, kind="pair", definition=definition, sign_convention=sign, operands=operands,
                            selectors=PAIR_SELECTORS, aggregation="beam_pair", model_basis=TWO_BEAM,
                            validity="; ".join(text for text in (validity, SAME_WAVELENGTH) if text))


GEOMETRIC_PAIR_SELECTORS = ("observation_id", "source_id", "source_id_b", "wavelength_nm", "wavelength_nm_b", "parameters.path_id_a",
                            "parameters.path_id_b", "parameters.crossing_index_a", "parameters.crossing_index_b")
GEOMETRIC_PAIR = ("geometric co-alignment of two beams: chief rays and paraxial Gaussian intensity envelopes on the plane; no phase, "
                  "no coherence, no polarization (not an interference quantity)")
GEOMETRIC_OPERANDS = ("a = the beam of source_id at wavelength_nm (parameters.path_id_a / crossing_index_a); b = the beam of "
                      "source_id_b (default source_id) at wavelength_nm_b (default wavelength_nm) (parameters.path_id_b / "
                      "crossing_index_b); each must select exactly one beam, not the same one")
GEOMETRIC_VALIDITY = "beams of any wavelengths (two lines of one source or of two sources); never WAVELENGTH_MISMATCH"


def _geometric(name: str, unit: str, definition: str, *, gaussian: bool) -> MetricDescriptor:
    validity = GEOMETRIC_VALIDITY + ("; needs valid Gaussian states of both beams (else OUT_OF_MODEL_SCOPE)" if gaussian else "")
    return MetricDescriptor(name=name, unit=unit, kind="pair", definition=definition, operands=GEOMETRIC_OPERANDS,
                            reference_frame=PLANE_FRAME, selectors=GEOMETRIC_PAIR_SELECTORS, aggregation="beam_pair",
                            model_basis=GEOMETRIC_PAIR, validity=validity)


COHERENCE_VALIDITY = ("reads the source coherence (linewidth_nm or coherence_length_mm; a source that declares none is taken "
                      "as monochromatic, the value stated with its value_basis entry at a degree of coherence of 1, or at the "
                      "other source's coherence length where the other source of the pair declares one, and a target judged "
                      "from 0 up to that value; where the two paths' group-delay dispersions differ, an ESTIMATE value_basis "
                      "entry, coherence_factor_max the largest any shorter coherence length of the first source gives, and "
                      "a target judged from 0 up to the value at coherence_factor_max) and the beams' relative group delay; where the delay is not "
                      "determined (an element without a thickness, a coating without a stated phase or group-delay bound) the "
                      "value is stated at no added delay at those passes with an ESTIMATE value_basis entry and decides no "
                      "target; within the coatings' group-delay bounds the value is stated at the traced delay and a target "
                      "is judged on the interval they leave, whatever its width (where only one source of the pair declares "
                      "its coherence: an ESTIMATE value_basis entry, coherence_factor_max 1, no "
                      "coherence_factor_min, and a target judged from 0 up to the value at coherence_factor_max); UNKNOWN (COHERENCE_UNDEFINED) where a medium "
                      "states no group index at the wavelength, where a pulsed source states no coherence, or where the delay "
                      "lies near a multiple of a pulsed source's period length c / f_rep (its pulses may be mutually "
                      "coherent)")
COHERENCE_LENGTH_VALIDITY = ("needs the source coherence (linewidth_nm or coherence_length_mm); a source that declares none has "
                             "no coherence length (UNKNOWN)")
OVERLAP_VALIDITY = ("needs valid Gaussian states of both beams; two beams that do not overlap on the plane (BEAMS_NOT_OVERLAPPING) "
                    "are compared on the values computed for that overlap")
#: The fringe-geometry metrics of two beams that do not overlap on the plane.
NO_OVERLAP_GEOMETRY = ("where the two beams do not overlap on the plane (BEAMS_NOT_OVERLAPPING: spatial_overlap_ratio at or below "
                       "interference_min_overlap) no fringe forms: the target reports the period of the two beams' wavevectors and is "
                       "NOT_APPLICABLE, not graded")

#: What a power on a plane with no traced arrival is, and what a plane through a surface's hit point reads.
PLANE_POWER_VALIDITY = ("a plane with no traced arrival reads 0 W only where it is KNOWN_ZERO (every launched watt ended somewhere known and "
                        "no light the trace does not follow reaches it above the line's negligible level, negligible_power_fraction of its input); where light the trace followed or light of unknown fate may reach "
                        "it the plane has no power value (untraced_power_W and power_upper_bound_W stand beside it; BOUNDED from 0 where only "
                        "followed light may arrive, else UNKNOWN), and the target is "
                        "graded on [0, bound of the light followed to the plane] with any selector that selects no arrival: a maximum at or above "
                        "the bound is MET, a minimum above it UNMET, else INDETERMINATE (INDETERMINATE where part of that light has an unknown fate "
                        "or no finite bound); where only light left out as negligible may add, the target is decided when "
                        "both ends of [0, bound] agree. A plane without an active area counts light within the scene's extent; a monitor beyond it needs "
                        "its active area stated. A plane through a physical surface's chief-ray hit point reads the power arriving at "
                        "that surface before the interaction (arrivals[].surface_coincidence, code SURFACE_COINCIDENT_ARRIVAL)")
#: What the exposure bounds do not count, with the magnitudes and the rule that follows (``EXPOSURE_MODEL``, ``power_rating_utilization``).
EXPOSURE_NOT_COUNTED = ("the internal folds of a plane-faced solid (a cube's diagonal and faces, a penta prism's mirrors, total internal "
                        "reflection there) counted: its certified closed walks add to the open paths' count, and a solid without a certified "
                        "sum of them is a closed return; not counted: light that total internal reflection holds in a curved lens body: none "
                        "at the reference designs' feeds (internal angle at most 0.9 deg against a critical angle of 41.4 deg), 2.6 % of the "
                        "power at a 75 deg entry cone and 13-16.9 % near grazing, with no bound on its consecutive total internal "
                        "reflections; light that refraction turns back onto a refracting surface it already passed "
                        "before its next reflection: no such path in a search of 14 fixture runs and 540 two-singlet configurations, which did "
                        "not cover plane-faced prisms and retroreflectors, ball and hemispherical refractors, trains of three or more elements "
                        "and catalog products")

DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    _obs("power_W", "W", "Power the plane receives from the matching beams, the sum of their powers with no interference term (counted "
         "arrivals: inside the active area, or with at least capture_min_power_fraction of their Gaussian power inside it): on an "
         "absorbing plane the detected power (each beam's power on the active area, the power ledger's DETECTED; "
         "arrivals[].detected_power_W), on a transparent plane the crossing beams' power. Where two or more beams of one wavelength "
         "arrive, observations[].power_basis is INCOHERENT_SUM and the target carries POWER_INCOHERENT_SUM: coherent_power_W states the "
         "power with their fields added.", aggregation="sum_over_matching_beams",
         basis="chief-ray packet power (Fresnel / coating / absorption / Gaussian aperture truncation)", validity=PLANE_POWER_VALIDITY),
    _obs("coherent_power_W", "W", "Power through the plane of all beams of one wavelength summed coherently (interference "
         "included; observations[].superpositions): on an absorbing plane the detected power (each beam's power on the active "
         "area, its field scaled to it in the interference term); equals power_W for a single beam.", aggregation="sum_over_matching_beams",
         basis="coherent superposition of paraxial Gaussian envelopes with plane-wave phases (observations[].superpositions), whole plane",
         validity="needs valid Gaussian states and declared coherence for beams of one coherence group; select the wavelength with "
                  "wavelength_nm when several arrive. Where a coating pass the beams do not share states no phase the power has no value, "
                  "only its range over every phase (total_power_min_W / total_power_max_W, phase_statement, PHASE_UNDEFINED). The target reads "
                  "that range widened by the light the trace follows that may reach the plane: a limit inside that interval is "
                  "INDETERMINATE, a limit outside it decides; stating the coating's phases gives the value. "
                  + PLANE_POWER_VALIDITY),
    _gauss("beam_radius_x_mm", "mm", "1/e^2 intensity radius of the Gaussian beam at the plane along its x axis (measured perpendicular to the beam)."),
    _gauss("beam_radius_y_mm", "mm", "1/e^2 intensity radius along the beam's y axis."),
    _gauss("beam_radius_mm", "mm", "Geometric mean sqrt(w_x w_y) of the two 1/e^2 radii at the plane."),
    _gauss("waist_radius_x_mm", "mm", "1/e^2 radius of the Gaussian waist the beam has (or had) along x, in its current medium."),
    _gauss("waist_radius_y_mm", "mm", "1/e^2 radius of the Gaussian waist along y."),
    _gauss("waist_distance_x_mm", "mm", "Signed axial distance from the x waist to the plane, along the beam: Re(q_x).",
           sign="plane position minus waist position along the propagation direction: positive = the waist lies upstream (before the "
                "plane; the beam is diverging), negative = the waist lies downstream (the beam is still converging), 0 = waist on the "
                "plane. Waist position = chief-ray hit point - waist_distance_x_mm * propagation direction."),
    _gauss("waist_distance_y_mm", "mm", "Signed axial distance from the y waist to the plane: Re(q_y).", sign="as waist_distance_x_mm"),
    _gauss("rayleigh_range_x_mm", "mm", "Rayleigh range Im(q_x) = pi n w0x^2 / (M^2 lambda) in the current medium."),
    _gauss("rayleigh_range_y_mm", "mm", "Rayleigh range Im(q_y)."),
    _gauss("divergence_half_angle_x_rad", "rad", "Far-field half-angle divergence w0x / z_Rx (in the current medium)."),
    _gauss("divergence_half_angle_y_rad", "rad", "Far-field half-angle divergence w0y / z_Ry."),
    _obs("aoi_deg", "deg", "Angle between the chief ray and the plane normal, in [0, 90]." + _PROMOTED, frame=PLANE_FRAME),
    _obs("optical_path_length_mm", "mm", "Optical path length of the chief ray from the source to the plane: sum of n x geometric length "
         "over every segment (including ancestor paths of a branch). "
         "No value behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating, whose model states no optical path."),
    _obs("geometric_path_length_mm", "mm", "Geometric length of the chief ray from the source to the plane."),
    _obs("phase_rad", "rad", "Plane-wave phase of the chief-ray field: k0 x OPL + arg(Ex), or + arg(Ey) where |Ey|^2 exceeds |Ex|^2 by more "
         "than rel 1e-9, wrapped and reported in steps of "
         "1e-6 rad in (-pi, pi] (-pi reads +3.141593) (the digits the optical path length determines). "
         "The Jones phase refers to the beam's own transported basis (beam.basis), whose orientation depends on the path: "
         "the phases of two beams are not directly comparable; use the pair metric phase_difference_rad for interference. "
         "No value behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating, whose model states no optical path.",
         sign="field convention E ~ exp[i(k.r - w t)]; the Gaussian Gouy phase is not included", basis=CHIEF + "; " + JONES),
    _obs("position_u_mm", "mm", "Chief-ray hit point coordinate along the plane's u axis." + _PROMOTED + _CENTROID, frame=PLANE_FRAME),
    _obs("position_v_mm", "mm", "Chief-ray hit point coordinate along the plane's v axis." + _PROMOTED + _CENTROID, frame=PLANE_FRAME),
    _obs("wavelength_nm", "nm", "Vacuum wavelength of the beam."),
    _obs("spot_rms_mm", "mm", "RMS radius of the pupil-ray hit points about their centroid on the plane (geometric spot; each ray carries "
         "equal power).", frame=PLANE_FRAME, basis=PUPIL, validity="null when no pupil ray reaches the plane"),
    _obs("spot_centroid_u_mm", "mm", "Centroid of the pupil-ray hit points along u.", frame=PLANE_FRAME, basis=PUPIL,
         validity="null when no pupil ray reaches the plane"),
    _obs("spot_centroid_v_mm", "mm", "Centroid of the pupil-ray hit points along v.", frame=PLANE_FRAME, basis=PUPIL,
         validity="null when no pupil ray reaches the plane"),
    _obs("capture_margin_mm", "mm", "Distance from the 1/e^2 beam ellipse (projected on the plane) to the active-area edge along the radial "
         "direction.", sign="positive = the 1/e^2 ellipse is inside the active area; negative = it crosses the edge", frame=PLANE_FRAME,
         basis=GAUSSIAN, validity="needs an active area (aperture) and a valid Gaussian state; WARNING with the state's code where that "
                                  "state carries one"),
    _obs("active_area_power_fraction", "1", "Fraction of the beam's Gaussian power inside the active area.", frame=PLANE_FRAME, basis=GAUSSIAN,
         validity="needs an active area and a valid Gaussian state; WARNING with the state's code where that state carries one"),
    _obs("acceptance_margin_deg", "deg", "Acceptance half-angle minus the chief-ray AOI." + _PROMOTED, sign="positive = inside the acceptance cone",
         frame=PLANE_FRAME, validity="needs acceptance_half_angle_deg"),
    _obs("direction_cosine_u", "1", "Component of the chief-ray direction along the plane's u axis." + _PROMOTED, frame=PLANE_FRAME),
    _obs("direction_cosine_v", "1", "Component of the chief-ray direction along the plane's v axis." + _PROMOTED, frame=PLANE_FRAME),
    _obs("stokes_s1", "1", "Normalized Stokes S1/S0 = (|Ex|^2 - |Ey|^2)/S0 in the beam's transverse basis (e1, e2).", basis=JONES + "; " + PARTIAL,
         frame="the beam's transported polarization basis (beam.basis)"),
    _obs("stokes_s2", "1", "Normalized Stokes S2/S0 = 2 Re(conj(Ex) Ey)/S0.", basis=JONES + "; " + PARTIAL,
         frame="the beam's transported polarization basis (beam.basis)"),
    _obs("stokes_s3", "1", "Normalized Stokes S3/S0 = 2 Im(conj(Ex) Ey)/S0.", sign="-1 = right circular (clockwise viewed facing the source), "
         "+1 = left circular", basis=JONES + "; " + PARTIAL, frame="the beam's transported polarization basis (beam.basis)"),
    _obs("polarization_ellipticity_deg", "deg", "Ellipticity angle chi = asin(s3)/2 (of the polarized part of partially polarized light).",
         sign="0 linear, -45 right circular, +45 left circular", basis=JONES + "; " + PARTIAL,
         validity="NOT_APPLICABLE for unpolarized light (no ellipse)"),
    _obs("polarization_azimuth_deg", "deg", "Angle of the polarization ellipse's major axis projected on the plane, from u toward v, in [0, 180) "
         "(of the polarized part of partially polarized light).",
         frame=PLANE_FRAME, basis=JONES + "; " + PARTIAL, validity="null for circular light or a major axis along the plane normal; "
         "NOT_APPLICABLE for unpolarized light"),
    _obs("degree_of_polarization", "1", "Degree of polarization sqrt(S1^2 + S2^2 + S3^2) / S0 of the light: 1 for a fully polarized beam; "
         "for a partially polarized source, of the sum of its two polarization components on the route.", basis=JONES + "; " + PARTIAL,
         validity="UNKNOWN where a component's power or polarization on the route is not known, or the path budget left the other "
                  "component of a route untraced"),
    _obs("group_delay_ps", "ps", "Group delay of the beam from its source to the plane: the arrival time of a wave packet's envelope, "
         "group optical path / c (ArrivalResult.group_path_length_mm / c), t = 0 at emission from the source's origin plane.",
         sign="positive: later than the emission", basis=GROUP_DELAY, validity=DELAY_VALIDITY),
    _obs("phase_delay_ps", "ps", "Phase delay of the beam's carrier from its source to the plane: optical path length / c (sum n L / c; "
         "coating phases not included). In a dispersive medium it differs from group_delay_ps by sum (n - n_g) L / c.",
         sign="positive: later than the emission", basis=CHIEF,
         validity="no value behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating, whose model states no "
                  "optical path"),
    _obs("image_parity", "1", "+1 when the image keeps the source's handedness, -1 when it is mirrored: (-1)^(number of reflections on the path).",
         sign="+1 not mirrored, -1 mirrored (both ends viewed along the propagation direction)", basis="reflection count of the path lineage"),
    _obs("image_rotation_deg", "deg", "Rotation of the source frame (x toward y) onto the plane frame seen along the propagation, from the A block "
         "of the first-order transfer, in (-180, 180]: 0 upright, 180 inverted when the plane's (u, v) axes are chosen parallel to the "
         "source's (x, y) axes (for example both with local +Y up); otherwise it includes the angle between the two frames.",
         sign="positive from the source x axis toward its y axis",
         frame="source frame (x, y) -> plane (u, v) when the beam travels along +n, else (u, -v)", basis=FIRST_ORDER,
         validity="null when the image is mirrored (see image_parity), the A block is singular (e.g. a focus of a collimated beam) or a "
                  "differential ray did not reach the plane; it is the image orientation when the plane is conjugate to the source plane"),
    _obs("transverse_magnification", "1", "sqrt(|det A|) of the first-order transfer's position block (mm per mm). Positions are measured "
         "in the plane, so on a plane tilted to the beam it includes the 1/cos stretch along the tilt.", basis=FIRST_ORDER,
         validity="as image_rotation_deg (singular A gives null); the image magnification when the plane is conjugate to the source plane. "
                  "On a plane near a focus of a collimated beam A is close to 0 (not singular): the value is then a small position "
                  "scaling of the source plane, not an image magnification"),
    _pair("beam_crossing_angle_deg", "deg", "Angle between the two chief-ray directions."),
    _pair("half_angle_deg", "deg", "Half the crossing angle."),
    _pair("fringe_period_nm", "nm", "In-plane fringe period lambda / (n |d_a - d_b projected on the plane|).",
          validity="null for (nearly) collinear beams (NO_FRINGES); " + NO_OVERLAP_GEOMETRY),
    _pair("fringe_spacing_3d_nm", "nm", "Fringe-plane spacing lambda / (2 n sin(alpha/2)) perpendicular to the fringe planes.",
          validity=NO_OVERLAP_GEOMETRY),
    _pair("optical_path_difference_mm", "mm", "OPL(a) - OPL(b) of the two chief rays at the plane.",
          sign="positive = beam a has the longer optical path", operands=TWO_BEAM_ORDER),
    _pair("optical_path_difference_abs_mm", "mm", "|OPL(a) - OPL(b)|: the magnitude of the optical path difference (use it for "
          "'|OPD| below a limit' targets).", sign="never negative"),
    _pair("phase_difference_rad", "rad", "phi_a - phi_b: k0 (OPL_a - OPL_b) minus arg(e_a* . e_b), wrapped and reported in steps of 1e-6 rad in (-pi, pi] (-pi reads +3.141593); e_a, e_b are the "
          "beams' 3D complex field vectors (design frame), so the result includes the phase of the polarization overlap (for example "
          "the pi of two linear fields pointing in opposite directions).",
          sign="phi_a - phi_b; it equals k0 (OPL_a - OPL_b) wrapped only when the two 3D field vectors are equal up to a positive "
               "factor (overlap phase 0)", operands=TWO_BEAM_ORDER,
          validity="UNKNOWN (PHASE_UNDEFINED) where a pass one beam makes and the other does not has no stated phase at the "
                   "wavelength (an element without a thickness, a coating without a stated phase); passes both beams make alike "
                   "cancel. Where the two polarizations are orthogonal (polarization_overlap at most energy_conservation_tol) "
                   "their overlap has no phase: the value is k0 (OPL_a - OPL_b) wrapped, an ESTIMATE whose value_basis names "
                   "the overlap below that limit"),
    _pair("coherence_length_mm", "mm", "Coherence length of the pair (the smaller of the two sources').", validity=COHERENCE_LENGTH_VALIDITY),
    _pair("coherence_margin", "1", "1 - |OPD| / L_c.", sign="positive = within the coherence length", validity=COHERENCE_VALIDITY),
    _pair("coherence_factor", "1", "|gamma| = exp(-(OPD / L_c)^2) (Gaussian-spectrum model).", validity=COHERENCE_VALIDITY),
    _pair("spatial_overlap_ratio", "1", "Normalized overlap integral of the two Gaussian envelopes on the plane.", validity=OVERLAP_VALIDITY),
    _pair("polarization_overlap", "1", "|e_a* . e_b| of the two normalized 3D field vectors."),
    _pair("intensity_balance", "1", "min(I_a, I_b) / max(I_a, I_b) at the envelope-product maximum (peak_position_uv_mm: the centre of "
          "the overlapping envelopes, not a fringe maximum).", validity=OVERLAP_VALIDITY),
    _pair("fringe_visibility", "1", "Power-weighted (integrated) fringe visibility of the two beams: their balance x spatial overlap "
          "x polarization overlap x coherence factor over their envelopes; 0 for mutually incoherent beams (different source "
          "components), whose fringe averages out. Light of the same wavelength the trace does not "
          "follow that may cross the plane bounds the contrast over the plane (the target is graded on that range).",
          validity=COHERENCE_VALIDITY + "; " + OVERLAP_VALIDITY),
    _pair("fringe_visibility_peak", "1", "Local fringe visibility at the envelope-product maximum, (I_max - I_min) / (I_max + I_min) "
          "of the light there; light of the same wavelength the trace does not follow that may reach the plane bounds it "
          "(the target is graded on that range).", validity=COHERENCE_VALIDITY + "; " + OVERLAP_VALIDITY),
    _pair("mean_irradiance_W_per_mm2", "W/mm^2", "I_a + I_b at the envelope-product maximum, as beam irradiance (power per unit area "
          "perpendicular to each beam). The power flux through the plane per unit plane area (the irradiance map, 'results "
          "irradiance') is smaller by the cosine of the angle of incidence. Light of the same wavelength the trace does not "
          "follow that may reach the plane adds to it: a target is graded on the range it leaves.", validity=OVERLAP_VALIDITY),
    _pair("peak_irradiance_W_per_mm2", "W/mm^2", "Peak beam irradiance including the interference term (perpendicular to the beams, "
          "as mean_irradiance_W_per_mm2; bounded by the light the trace does not follow likewise).",
          validity=COHERENCE_VALIDITY + "; " + OVERLAP_VALIDITY),
    _pair("overlap_effective_area_mm2", "mm^2", "Effective area 2 pi / sqrt(det(S_a + S_b)) of the overlapping envelopes.", validity=OVERLAP_VALIDITY),
    _pair("estimated_exposure_time_s", "s", "Dose / mean irradiance with parameters.dose_mj_per_cm2.", validity="needs parameters.dose_mj_per_cm2 > 0: a target without it, or with a dose that is not positive, is refused as input"),
    _pair("time_delay_ps", "ps", "Relative group delay of the two beams' envelopes at the plane, t_a - t_b = (group path a - group path b) "
          "/ c (interference[].group_path_difference_mm / c): the delay of a pump-probe pair, an interferometer arm, a delay line.",
          sign="positive: beam a arrives later", operands=TWO_BEAM_ORDER,
          validity="UNKNOWN (GROUP_DELAY_UNDEFINED) where the beams cross different passes without a stated group delay (the value is "
                   "then the relative delay of the two modelled paths, an ungraded ESTIMATE in value_basis), a medium's "
                   "group index is not available, or the beams come from different sources (no common time origin); coatings the "
                   "beams meet differently whose group delay is bounded give an interval the target is graded on; passes both "
                   "beams make alike cancel"),
    _geometric("centroid_separation_mm", "mm", "Distance |c_a - c_b| between the two chief-ray hit points on the plane.", gaussian=False),
    _geometric("normalized_centroid_separation", "1", "|c_a - c_b| / sqrt(r_a^2 + r_b^2), r the 1/e^2 footprint radius of each beam "
               "on the plane along the separation direction; 0 for coincident centres.", gaussian=True),
    _geometric("angular_separation_rad", "rad", "Angle between the two chief-ray directions, atan2(|d_a x d_b|, d_a . d_b).", gaussian=False),
    _geometric("geometric_overlap_ratio", "1", "Overlap of the two intensity footprints on the plane, integral of sqrt(I_a I_b) over "
               "the plane / sqrt(P_a P_b): 1 for identical footprints, exp(-d^2 / (2 w^2)) for two round beams of radius w whose centres "
               "are d apart at normal incidence.", gaussian=True),
    _geometric("waist_position_separation_mm", "mm", "How far apart along the beams the two Gaussian waists lie: |(W_a - W_b) . m| with "
               "W the waist position (hit point minus the mean of the x / y waist distances along the beam) and m the unit mean "
               "direction of the two beams.", gaussian=True),
    MetricDescriptor(name="power_rating_utilization", unit="1", kind="element",
                     definition="The largest exposure / damage threshold over the element's rated surfaces and the ratings "
                                "that apply to the light there (exposures.elements[].max_utilization): above 1 the element's rating "
                                "is exceeded. The threshold is the one the rated value establishes for the peak exposure: a "
                                "J/cm2 or W/cm2 value stated as a peak (value_definition PEAK) is the threshold, one stated as "
                                "the total over the area of the test beam's 1/e^2 diameter (TOTAL_PER_1E2_AREA) gives twice it, "
                                "and one whose definition is not stated gives none (its conditions are not verified, and it "
                                "bounds the utilization from neither side); a W/cm value is its own threshold; a typical value "
                                "bounds no threshold. A rating applies as stated when every condition it states is "
                                "verified; a test beam "
                                "at least as large as the beam here makes the damage threshold at least the rated value, so the "
                                "rating bounds the utilization from above only; a smaller test beam makes it at most the rated "
                                "value, so the rating bounds it from below only. Such a rating, and one whose conditions are not "
                                "verified (its threshold may be at most its value), leaves the surface no upper bound unless a "
                                "rating of the same unit that applies as stated or from above has a lowest threshold no larger "
                                "than its highest (W/cm values compared as P/d of the 1/e^2 diameter). Where a smaller-test-beam "
                                "rating has only larger ones the ratings disagree: the lower bound stands and nothing bounds it "
                                "from above. A target is graded on the interval; its value is then the largest "
                                "exposure / threshold of the ratings that apply (the exposure exact, or its lower bound), with "
                                "no value when none of those exposures is known or bounded from below.",
                     operands="the element is the target's device_id", selectors=("device_id",), aggregation="max_over_surfaces",
                     model_basis="per-surface exposure: chief-ray packet power and paraxial Gaussian beam at each surface (peak "
                                 "irradiance 2P/(pi w_x w_y), P/d, peak fluence 2E/(pi w_x w_y)); beams that cannot interfere add, "
                                 "mutually coherent beams (one coherence group and line, from either side) combine by their "
                                 "amplitudes with open phases (a range); light left out as negligible adds a field of at most "
                                 "the amplitude of its diffraction limit 2 pi n^2 P / lambda^2 (P = (sum of the square roots "
                                 "of its pieces' powers)^2) times sqrt(2) m sum_{j=0}^{N} a_j, at least 2 (m meetings after "
                                 "each number of reflections, from either side: 1 on a lens or a refracting surface between "
                                 "different media, on any other surface as often per straight leg as a straight line meets it "
                                 "-- once a plane, twice a sphere or a conic, an asphere by the degree of its sag; a_j the most "
                                 "field of its pieces after j reflections, from different sequences of the N surfaces that may "
                                 "reflect it, added in phase); from below, a traced beam's mean over a window on its area, "
                                 "which light of unknown place lowers only by what its power can take away there; compared with "
                                 "the element's stated ratings without scaling between conditions; the utilization runs from the "
                                 "ratings that apply to every rating that may apply; closed returns through three or more "
                                 "surfaces of separate elements (a ring, a folded or coupled cavity, a periscope's windows "
                                 "through its folds) and stacks of parallel planes are screened: a stack keeps an all-orders "
                                 "series of its returns where one converges, and a possible return without one leaves every "
                                 "surface it may reach without an upper bound (OUT_OF_MODEL_SCOPE); a pair whose gain bound is 1 "
                                 "and a coating use outside its record leave a surface without one too (EXPOSURE_BOUND_WITHHELD); "
                                 + EXPOSURE_NOT_COUNTED + "; no thermal or damage model",
                     validity="UNKNOWN when the element has no rating (POWER_RATING_MISSING), when no rating applies to the "
                              "light (POWER_RATING_CONDITIONS_MISMATCH: a condition differs with no established relationship), "
                              "when a rating states a condition the light cannot be checked against -- a pulse count (the Project "
                              "states no pulses per site), a test beam diameter between the readings of an unstated definition, a "
                              "polarization stated only as text (POWER_RATING_CONDITIONS_NOT_VERIFIED) --, when a surface's "
                              "ratings bound it from one side only or disagree and that side does not decide (a smaller test beam "
                              "never PASSes, a larger one never FAILs), or when its exposure range straddles the limit (a beam "
                              "without a valid Gaussian state, coherent beams with open phases, an incomplete trace, light left "
                              "out as negligible or of unknown fate of known power bounded by its diffraction limit and its "
                              "returns between surfaces facing each other: FAINT_LIGHT_BOUNDED, FAINT_LIGHT_RECIRCULATES; light at "
                              "the interaction limit: no upper bound); never PASS while a rating that "
                              "may apply is not met, while a FAIL decided by a rating that applies stands"),
)
#: The retained-DOF limit metrics (``validation/mechanical.py``): stated inputs compared with what the design needs.
DOF_SELECTORS = ("device_id", "parameters.axis")
DOF_BASIS = ("the limits stated on the retained DOF (its travel, drift and repeatability as the design declares them); "
             "not computed from the trace")
DOF_VALIDITY = ("UNKNOWN (MECHANICAL_LIMIT_NOT_DECLARED, the target INDETERMINATE) when the DOF states no travel, or not the "
                "drift term or repeatability the target reads: an adjustment without stated limits is not verified")
DOF_OPERANDS = "the DOF is the target's device_id (a component) and parameters.axis (Tx, Ty, Tz for _mm; Rx, Ry, Rz for _deg)"


def _dof(name: str, unit: str, definition: str, sign: str = "") -> MetricDescriptor:
    return MetricDescriptor(name=name, unit=unit, kind="element", definition=definition, sign_convention=sign, operands=DOF_OPERANDS,
                            reference_frame="the component's local frame at its nominal pose (retained DOF axes)",
                            selectors=DOF_SELECTORS, aggregation="declared_limit", model_basis=DOF_BASIS, validity=DOF_VALIDITY)


TRAVEL_DEFINITION = ("How far the DOF's stated travel reaches beyond the adjustment the design needs, [parameters.required_min, "
                     "parameters.required_max] relative to the nominal pose: min(travel_max - required_max, required_min - "
                     "travel_min), either end alone using that side only. A target minimum 0 states that the needed adjustment "
                     "stays inside the travel.")
TRAVEL_SIGN = "zero or more: the needed adjustment lies inside the travel; negative: it exceeds the travel by that much"
DRIFT_DEFINITION = ("The DOF's drift over a stated span: drift_per_kelvin x parameters.temperature_span_K + drift_per_hour x "
                    "parameters.duration_h (the terms given, magnitudes added). A target maximum is the drift the design tolerates.")
REPEATABILITY_DEFINITION = "The DOF's stated repeatability (+/-). A target maximum is the repeatability the design needs."
DOF_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    _dof("dof_travel_margin_mm", "mm", TRAVEL_DEFINITION, TRAVEL_SIGN),
    _dof("dof_travel_margin_deg", "deg", TRAVEL_DEFINITION, TRAVEL_SIGN),
    _dof("dof_drift_mm", "mm", DRIFT_DEFINITION),
    _dof("dof_drift_deg", "deg", DRIFT_DEFINITION),
    _dof("dof_repeatability_mm", "mm", REPEATABILITY_DEFINITION),
    _dof("dof_repeatability_deg", "deg", REPEATABILITY_DEFINITION),
)
#: The detection metrics (``validation/detection.py``): the detector of the target's plane reads every source and wavelength
#: the plane counts.
DETECTION_BASIS = ("photodetector on the observation plane (Observation.detection): responsivity QE q lambda / (h c), white shot "
                   "noise 2qI, the stated dark-noise floor (NEP, or dark current shot noise and load-resistance thermal noise) in the "
                   "stated bandwidth; lock-in: rms fundamental of the modulated current over the noise in the output filter's "
                   "noise-equivalent bandwidth")
DETECTION_VALIDITY = ("UNKNOWN (DETECTION_QUANTITY_UNAVAILABLE, with the reason) when the detector does not state what the "
                      "quantity needs (PARAMETER_NOT_STATED), the light holds a wavelength outside the quantum-efficiency table, "
                      "the plane has no traced light of known power while light may arrive, its power is out of model scope or "
                      "its bounds were not computed (PLANE_POWER_NOT_KNOWN), the peak photocurrent is above the "
                      "detector's stated saturation current (PHOTOCURRENT_ABOVE_LINEAR_RANGE), the modulation lies above the "
                      "detector's bandwidth (MODULATION_ABOVE_DETECTOR_BANDWIDTH: above the -3 dB frequency of a first-order "
                      "response, or 2/pi of a noise-equivalent bandwidth), or the lock-in's filter averages over fewer than 10 "
                      "modulation periods per time constant (LOCK_IN_AVERAGING_TOO_SHORT). On a lit plane whose power is BOUNDED "
                      "or UNKNOWN the value is computed from the traced light and value_basis['value'] says what it is: where the "
                      "light that may add is bounded and the linear range holds for it, the target is graded on the interval the "
                      "quantity lies in; else a LOWER_BOUND decides a minimum it meets and a maximum it exceeds, an UPPER_BOUND "
                      "the reverse, and an ESTIMATE leaves the target INDETERMINATE")


def _detection(name: str, unit: str, definition: str) -> MetricDescriptor:
    return MetricDescriptor(name=name, unit=unit, kind="observation", definition=definition, selectors=("observation_id",),
                            aggregation="sum_over_matching_beams", model_basis=DETECTION_BASIS, validity=DETECTION_VALIDITY,
                            operands="every source and wavelength the plane counts (source_id and wavelength_nm are not set)")


DETECTION_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    _detection("detector_bandwidth_Hz", "Hz", "The electrical bandwidth the plane's detector states (electrical.bandwidth_Hz: the -3 dB "
               "frequency of a first-order response, or the noise-equivalent bandwidth, as bandwidth_definition says). A target "
               "minimum states the bandwidth the measurement needs."),
    _detection("detector_snr", "1", "Signal-to-noise ratio (an amplitude ratio, not dB) of the detector's DC photocurrent: the "
               "photocurrent of the light the plane counts (a chopped source in its open state) over the root-sum-square of its "
               "shot noise and the dark-noise floor in the noise-equivalent bandwidth. Not stated for a SQUARE chop above the "
               "detector's bandwidth, whose output never reaches the open state."),
    _detection("lock_in_snr", "1", "Signal-to-noise ratio of the lock-in output X: the rms fundamental of the modulated photocurrent "
               "(SQUARE: (2/pi) sin(pi D) times the open-state current; SINE: the depth times the mean) over the noise in the "
               "output filter's noise-equivalent bandwidth (1/(4 tau) for one RC section), reference in phase."),
)
#: The second-harmonic metrics (``validation/nonlinear.py``): one crystal, the selected fundamental line.
SHG_SELECTORS = ("device_id", "source_id", "wavelength_nm")
SHG_OPERANDS = ("the crystal is the target's device_id; source_id / wavelength_nm select the fundamental line (default: every line that "
                "enters it); every traced pass of that light into the crystal counts")
SHG_BASIS = ("undepleted-pump second-harmonic generation of a TEM00 Gaussian beam in a uniaxial crystal slab (paraxial coupled waves): "
             "P_2 = K f P_a P_b L^2 H / (pi w0x w0y), K = 2 omega^2 d_eff^2 / (n_a n_b n_2 eps0 c^3), H the Gaussian overlap integral "
             "with the phase mismatch of the traced waves (angle, temperature, quasi-phase-matching grating), walk-off and, for "
             "boyd_kleinman focusing, the beam's focusing (Boyd and Kleinman 1968); sinc^2(dk L / 2) for a plane wave")
SHG_VALIDITY = ("OUT_OF_MODEL_SCOPE (NONLINEAR_CONVERSION_OUT_OF_SCOPE) when a selected pass is outside the model: the estimated "
                "pump depletion above nonlinear_depletion_rel_tol, the near-field limit off by more than nonlinear_focusing_rel_tol, a "
                "fundamental that is not one polarized continuous-wave TEM00 line with a valid Gaussian state (a pulsed source; a line "
                "whose width changes the phase matching by more than nonlinear_linewidth_rel_tol), a harmonic already travelling "
                "with the light (cascaded crystals, a second pass); UNKNOWN when no traced beam of the "
                "selection enters the crystal (TARGET_SELECTION_EMPTY) or part of its light has an unknown fate")
NONLINEAR_DESCRIPTORS: tuple[MetricDescriptor, ...] = tuple(
    MetricDescriptor(name=name, unit=unit, kind="element", definition=definition, operands=SHG_OPERANDS, selectors=SHG_SELECTORS,
                     aggregation="sum_over_matching_beams", model_basis=SHG_BASIS, validity=SHG_VALIDITY)
    for name, unit, definition in (
        ("shg_conversion_efficiency", "1", "The crystal's second-harmonic conversion efficiency: the harmonic power it generates "
         "(inside the crystal, at its exit face: after the crystal's absorption, if any) divided by the fundamental power entering it, over every traced pass of the "
         "selected light."),
        ("nonlinear_conversion_efficiency", "1", "The same quantity as shg_conversion_efficiency (the nonlinear conversion the "
         "Engine models is second-harmonic generation)."),
        ("shg_power_W", "W", "The second-harmonic power the crystal generates from the selected light (inside the crystal, at its "
         "exit face: after the crystal's absorption, if any), summed over every traced pass."),
    ))
STAGE_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(name="layout_violation_count", unit="1", kind="layout", definition="Number of FAIL layout issues.", aggregation="stage",
                     model_basis="2.5D layout checks"),
    MetricDescriptor(name="layout_min_clearance_mm", unit="mm", kind="layout", definition="Smallest footprint clearance.", aggregation="stage",
                     model_basis="2.5D layout checks"),
    MetricDescriptor(name="recoverability_fraction", unit="1", kind="control", definition="Fraction of disturbance cases recovered.",
                     aggregation="stage", model_basis="Levenberg-Marquardt control analysis"),
    MetricDescriptor(name="jacobian_condition_number", unit="1", kind="control", definition="Condition number of the sensor Jacobian.",
                     aggregation="stage", model_basis="finite-difference sensitivity"),
    MetricDescriptor(name="monte_carlo_recovered_fraction", unit="1", kind="control", definition="Recovered fraction of Monte Carlo samples.",
                     aggregation="stage", model_basis="deterministic Monte Carlo"),
    MetricDescriptor(name="numerical_rank", unit="1", kind="control", definition="Numerical rank of the sensor Jacobian.", aggregation="stage",
                     model_basis="finite-difference sensitivity"),
)
DESCRIPTOR_BY_NAME = {d.name: d for d in DESCRIPTORS + DOF_DESCRIPTORS + NONLINEAR_DESCRIPTORS + DETECTION_DESCRIPTORS + STAGE_DESCRIPTORS}


GAUSSIAN_FIELDS = ("beam_radius_x_mm", "beam_radius_y_mm", "waist_radius_x_mm", "waist_radius_y_mm", "waist_distance_x_mm",
                   "waist_distance_y_mm", "rayleigh_range_x_mm", "rayleigh_range_y_mm")
PAIR_FIELDS = ("beam_crossing_angle_deg", "half_angle_deg", "fringe_period_nm", "fringe_spacing_3d_nm", "optical_path_difference_mm",
               "phase_difference_rad", "coherence_length_mm", "coherence_factor", "coherence_margin", "spatial_overlap_ratio",
               "overlap_effective_area_mm2", "polarization_overlap", "intensity_balance", "fringe_visibility", "fringe_visibility_peak",
               "mean_irradiance_W_per_mm2", "peak_irradiance_W_per_mm2")
#: Where each metric's value appears among a run's results (``MetricDescriptor.result_fields``).
RESULT_FIELDS: dict[str, tuple[str, ...]] = {
    **{name: (f"arrivals.{name}", f"monitor:arrivals[].beam.gaussian.{name}") for name in GAUSSIAN_FIELDS},
    "divergence_half_angle_x_rad": ("arrivals.far_field_half_angle_x_rad", "monitor:arrivals[].beam.gaussian.far_field_half_angle_x_rad"),
    "divergence_half_angle_y_rad": ("arrivals.far_field_half_angle_y_rad", "monitor:arrivals[].beam.gaussian.far_field_half_angle_y_rad"),
    "power_W": ("arrivals.power_W", "arrivals.detected_power_W", "monitors.total_power_W", "monitor:arrivals[].beam.power_W"),
    "coherent_power_W": ("superpositions.total_power_W", "superpositions.total_power_min_W", "superpositions.total_power_max_W",
                         "monitor:superpositions[]"),
    "aoi_deg": ("arrivals.aoi_deg", "monitor:arrivals[].aoi_deg"),
    "optical_path_length_mm": ("arrivals.optical_path_length_mm", "monitor:arrivals[].beam.optical_path_length_mm"),
    "geometric_path_length_mm": ("arrivals.geometric_length_mm", "monitor:arrivals[].beam.geometric_length_mm"),
    "phase_rad": ("arrivals.phase_rad", "monitor:arrivals[].beam.phase_rad"),
    "wavelength_nm": ("arrivals.wavelength_nm", "monitor:arrivals[].beam.wavelength_nm"),
    "position_u_mm": ("arrivals.u_mm", "monitor:arrivals[].uv_mm[0]"),
    "position_v_mm": ("arrivals.v_mm", "monitor:arrivals[].uv_mm[1]"),
    "direction_cosine_u": ("arrivals.direction_u", "monitor:arrivals[].direction_uvn[0]"),
    "direction_cosine_v": ("arrivals.direction_v", "monitor:arrivals[].direction_uvn[1]"),
    "spot_centroid_u_mm": ("arrivals.pupil_centroid_u_mm", "monitor:arrivals[].pupil_centroid_uv_mm[0]"),
    "spot_centroid_v_mm": ("arrivals.pupil_centroid_v_mm", "monitor:arrivals[].pupil_centroid_uv_mm[1]"),
    "spot_rms_mm": ("arrivals.pupil_rms_radius_mm", "monitor:arrivals[].pupil_rms_radius_mm"),
    "stokes_s1": ("arrivals.stokes_s1", "monitor:arrivals[].polarization_state.s1"),
    "stokes_s2": ("arrivals.stokes_s2", "monitor:arrivals[].polarization_state.s2"),
    "stokes_s3": ("arrivals.stokes_s3", "monitor:arrivals[].polarization_state.s3"),
    "polarization_ellipticity_deg": ("arrivals.polarization_ellipticity_deg", "monitor:arrivals[].polarization_state.ellipticity_deg"),
    "polarization_azimuth_deg": ("arrivals.polarization_azimuth_deg", "monitor:arrivals[].polarization_state.azimuth_in_plane_deg"),
    "degree_of_polarization": ("monitor:arrivals[].polarization_state.degree_of_polarization",
                               "monitor:polarization_mixtures[].degree_of_polarization"),
    "group_delay_ps": ("monitor:arrivals[].group_path_length_mm",),
    "phase_delay_ps": ("arrivals.optical_path_length_mm", "monitor:arrivals[].beam.optical_path_length_mm"),
    "time_delay_ps": ("monitor:interference[].group_path_difference_mm", "monitor:interference[].group_delay_bound_mm"),
    "acceptance_margin_deg": ("arrivals.acceptance_margin_deg", "monitor:arrivals[].acceptance_margin_deg"),
    "active_area_power_fraction": ("arrivals.active_area_power_fraction", "monitor:arrivals[].active_area_power_fraction"),
    "capture_margin_mm": ("arrivals.capture_margin_mm", "monitor:arrivals[].capture_margin_mm"),
    "transverse_magnification": ("arrivals.fo_transverse_magnification", "monitor:arrivals[].first_order.transverse_magnification"),
    "image_rotation_deg": ("arrivals.fo_image_rotation_deg", "monitor:arrivals[].first_order.image_rotation_deg"),
    **{name: (f"monitor:interference[].{name}",) for name in PAIR_FIELDS},
    "optical_path_difference_abs_mm": ("monitor:interference[].optical_path_difference_mm",),
    # (a plane whose detection states the detector's electrical response; `quantities[<name>]` is the entry of that name)
    **{name: (f"monitor:detection.quantities[{name}].value",) for name in ("detector_bandwidth_Hz", "detector_snr", "lock_in_snr")},
    "shg_conversion_efficiency": ("nonlinear.conversion_efficiency",),
    "nonlinear_conversion_efficiency": ("nonlinear.conversion_efficiency",),
    "shg_power_W": ("nonlinear.harmonic_power_W",),
}


def metric_descriptors(*, include_stage_metrics: bool = False) -> tuple[MetricDescriptor, ...]:
    """Every public target metric's ``MetricDescriptor`` (optical metrics; with ``include_stage_metrics`` also the layout /
    control metrics of Engine requests with mechanical envelopes and actuators), sorted by name, each with where its value
    appears among a run's results and the evaluation capabilities a target on it reads (``metrics.METRIC_REQUIRES``)."""

    from .metrics import METRIC_REQUIRES  # (the metric registry imports the Target contract, which reads this module)

    chosen = (DESCRIPTORS + DOF_DESCRIPTORS + NONLINEAR_DESCRIPTORS + DETECTION_DESCRIPTORS
              + (STAGE_DESCRIPTORS if include_stage_metrics else ()))
    return tuple(sorted((d.model_copy(update={"result_fields": RESULT_FIELDS.get(d.name, ()), "requires": METRIC_REQUIRES[d.name]})
                         for d in chosen), key=lambda d: d.name))
