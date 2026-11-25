"""Per-surface beam exposure and its comparison with the elements' power ratings (``ValidationResult.exposures``).

Input: the tracer's ``SurfaceHit`` records (``optics/rays/hits.py``: every traced packet's arrival at a physical surface,
with its power, both Gaussian radii, Jones vector, chief-ray optical path and coherence there), in a run with power ratings
the probe traces' hits of the light the trace does not follow and the light left out as negligible
(``validation/untraced.py``: ``surface_reach``), the sources' pulse trains and the components' ``power_ratings``. Nothing
is traced again.

Per hit (``ExposureHit``), with ``P`` the incident power, ``w_x`` / ``w_y`` the 1/e^2 radii and ``theta`` the chief ray's
angle of incidence on the local surface normal:

* peak irradiance in the beam's transverse plane ``I0 = 2 P / (pi w_x w_y)``, on the surface ``I0 cos(theta)``;
* linear power density ``P / d`` with ``d = 2 min(w_x, w_y)`` (the narrower 1/e^2 diameter: the larger value);
* for a pulsed source (``SourceSpec.pulse``) the pulse energy ``E = P / f_rep`` and the peak fluence ``2 E / (pi w_x w_y)``;
* the 1/e^2 footprint on the surface: the ellipse ``x^T S x = 1`` of ``S = b_x b_x^T / w_x^2 + b_y b_y^T / w_y^2``
  (beam axes projected onto the tangent plane), semi-axes ``1 / sqrt(eigenvalues)``.

"Peak" is spatial: ``P`` is the average power, so for a pulsed source the peak irradiance is the spatial peak of the
average irradiance; a pulse is described by its energy and peak fluence.

A hit whose Gaussian state is not valid (invalidated upstream, behind a central obstruction, or aberration dominated:
the pupil rays spread more than ``gaussian_max_ray_spread_ratio`` times the aberration-free bound, the observation-plane
rule) is ``UNKNOWN``: its beam-size quantities are ``None``, never 0. A hit whose beam crossed a tabulated thin lens
(``TabulatedThinLensInteraction``) is in that response model's scope only when the lens's ``supported_metrics`` list
``power_rating_utilization``, or ``power_W`` together with a beam-size metric (``beam_radius_mm``, or both
``beam_radius_x_mm`` and ``beam_radius_y_mm``): the exposure behind the lens is the ideal focusing's Gaussian spot.
Otherwise the hit is ``OUT_OF_MODEL_SCOPE`` (``METRIC_NOT_VALIDATED_FOR_ELEMENT``) with no beam-size quantities, and a
``power_rating_utilization`` target of an element such a beam reaches is out of model scope (``element_scope``). A hit of a
beam whose power is out of model scope (``SurfaceHit.power_scope``: behind a crystal whose second-harmonic generation is
out of the model's conditions -- e.g. a pulsed source, whose conversion follows the peak power --, a variable attenuator or
transmission map that varies across the beam, an element or absorbing plane that part of the beam meets beside its chief
ray) is ``OUT_OF_MODEL_SCOPE`` the same way (``NONLINEAR_CONVERSION_OUT_OF_SCOPE`` behind such a crystal, otherwise
``OUT_OF_MODEL_SCOPE``): its power is not the light's, so neither its exposure nor a bound of it is stated, and a
``power_rating_utilization`` target of an element such a beam reaches is out of model scope (``power_scope``).

The diffraction limit. Light of power ``p`` at vacuum wavelength ``lambda`` in a medium of index ``n`` that crosses a plane
from one side cannot reach an irradiance above ``I_max(p) = 2 pi n^2 p / lambda^2`` anywhere on it: writing its field as a
sum of plane waves of wavenumber ``k = 2 pi n / lambda`` (its angular spectrum ``a``), the power through the plane is the
integral of ``|a|^2 cos(angle)`` over the transverse wavevectors and the Cauchy-Schwarz inequality gives
``|E(0)|^2 <= p / (4 pi^2) x integral of 1 / cos(angle) over the disk |k_t| <= k = p k^2 / (2 pi)``. It is the peak
``2 p / (pi w^2)`` of a Gaussian beam of waist ``w = lambda / (pi n)`` (far-field half angle 1 rad; every beam the
Gaussian model describes has at most ``gaussian_scope_divergence_rad``, so at most 0.09 of it), and the same waist gives
the bound of a fluence (``2 E / (pi w^2)``) and of ``P / d`` (``d`` = ``2 w`` or ``w sqrt(2 ln 2)``) -- the values of
``_Quantity.limit``. Light of a spectrum adds the limits of its spectral parts: the limit scales with the mean of
``(lambda / lambda')^2`` over it (``spectral_factor``: the Gaussian spectrum of a declared coherence length, or a pulse's
transform-limited one; 1.0025 for a 10 fs pulse at 800 nm). Light that crosses a plane or meets a surface more than
once, from either side, has a field there of at most the sum of its crossings' amplitudes.

The exposure is the intensity of the summed fields of the light arriving at the surface from either side, in irradiance
units. Light on one surface combines in classes (``_classes``): contributions of one coherence group (the source's
``coherence_group``; one per source by default) and one spectral line (vacuum wavelengths within
``interference_wavelength_tol_nm``) are mutually coherent, whichever side of the surface they arrive from (beams crossing
the surface in opposite directions form a standing wave); light of different classes has no interference term in the time
average (different frequencies, mutually incoherent sources), so its values add. Within a class the fields add with
phases the exposure model leaves open, each pair with its degree of coherence ``gamma_ij``. With a declared coherence
length ``L_c`` (a finite bandwidth) two fields are coherent according to the delay of their envelopes: the Gaussian
spectrum of ``L_c`` after their relative group delay ``tau`` -- the difference of their group optical paths ``sum n_g L``,
not of their phase paths -- and group-delay dispersion (``optics/interference/coherence.py``: ``(1 + b^2)^(-1/4)
exp(-(tau / L_c)^2 / (1 + b^2))``, ``exp(-(tau / L_c)^2)`` without dispersion; the observation planes' model,
``validation/superposition.py``). A pulsed source's pulses may be mutually coherent (a mode-locked train): two of its
fields are coherent again near every whole multiple of its period length ``c / f_rep``, so ``tau`` counts as the smaller
of itself and its distance to the nearest non-zero multiple (``coherence.nearest_delay``). ``tau`` is the chief rays'
value only along them -- across the surface it changes with
the beams' tilt, offset and wavefront curvature, beams that cross at an angle have no relative delay somewhere in their
overlap, and a beam that crossed a dispersive medium has its envelope delay change across it beyond its wavefront's.
``gamma_ij`` is therefore the largest value the pair's cross term can take anywhere (``_coherence``): within a transverse
radius ``sqrt(K) w`` of the first beam's chief ray the delay differs from the chief rays' by at most ``r |n_i d_i - n_j
d_j| + n_j |(c_i - c_j) . d_j| + n_i r^2 / (2 |R_i|) + n_j (r + s)^2 / (2 |R_j|)`` (``r`` = that radius over the cosine of
its angle of incidence, ``c`` the chief rays' hits ``s`` apart, ``R`` the wavefronts' radii of curvature; on a plane
surface only the part of ``n_i d_i - n_j d_j`` along the plane counts, so beams meeting it from opposite sides at normal
incidence keep their standing wave's delay) plus each beam's envelope spread over the ray family that holds those points
(``optics/rays/envelope.py``: through a plate it changes with the rays' angles only; the dispersion's range widens the
spectrum's coherence likewise), outside it the beams' profiles carry less than ``exp(-K)`` of their peak amplitudes, and
the smallest over a grid of ``K`` of the two is taken (1 where a length is not declared, a path difference is not defined
-- behind an ideal or tabulated thin lens --, a beam's Gaussian state is not known or it crossed a dispersive medium other
than a plate -- a lens, a prism, a cube --; 0 below half a unit in the last place of any sum it enters). Collinear beams
keep about the chief rays' value; beams crossing at an angle get 1. At one point ``p`` the same bound with ``r = |p - c_i|``
holds. For a value ``v`` (any
of the quantities above, each beam's value at the point considered) the class's combined value ``V`` satisfies

    V <= sum_i v_i + 2 sum_{i<j} gamma_ij sqrt(v_i v_j)  (<= (sum_i sqrt(v_i))^2, the in-phase maximum)
    V >= max( max(0, sqrt(v_k) - sum_{j != k} sqrt(v_j))^2,  sum_i v_i - 2 sum_{i<j} gamma_ij sqrt(v_i v_j) )

(Minkowski's inequality for the time-averaged field, which holds whatever the spectrum and phases, and the pairwise worst
case of the cross terms in the coherence model). Two equal in-phase beams of one source therefore reach 4 v, not 2 v; in
antiphase 0. The upper bound of a surface's exposure is the sum over the classes of their bound at the beams' peak values
(as if every peak coincided); the lower bound the largest, over the traced beams' centres ``p``, of the sum over the
classes of their lower bound with each beam's value at ``p`` (its Gaussian profile there, ``exp(-2 rho^2)`` with ``rho``
the distance from its chief ray in units of its 1/e^2 radii, in its transverse frame) and the degree of coherence at
``p``. A single beam, or beams that cannot interfere and whose peaks coincide, give equal bounds: the exposure is then
exact. For ``W/cm`` each beam's ``P / d`` is weighted by its profile likewise. Light the trace does not follow (untraced
hits, beams beside the surface) only raises the upper bound and, coherent with the traced beams of its class, lowers the
lower bound (it never raises it); an untraced beam whose size is not known (no valid Gaussian state) counts with the
diffraction limit of its power, anywhere on the surface. Light of unknown place -- such a beam, and the light left out
as negligible (below) -- may be at that bound at a beam's centre, but it cannot be strong over the whole area the beam
covers: in peak irradiance and peak fluence the lower bound is also, per class with such light, the window bound of each
of its traced beams of one Gaussian mode on a plane or a sphere (``_window``, ``validation/window_bound.py``): the beam's
mean against a window ``exp(-s rho^2)`` over its area, less the overlap of the class's other beams with that window and
``H (sum_i sqrt(P_i) + c f sqrt(P*))`` -- the most light of power ``P`` crossing the surface from one side, whatever its
directions, can take away from that mean, ``H`` the window's overlap with plane waves weighted by ``1 / cos`` --, the
largest over the window's widths (the largest value on the surface is at least the mean). In ``P / d`` (a beam's power
over its narrower transverse diameter: not a field, no phase used) the window counts that light as Gaussian beams of its
power, no more focused than the diffraction limit and crossing the surface from one side, each taking away at most its
``P / d`` over the strip of the window it covers (``window_bound.linear_lower``, any traced beam on a plane or a
sphere): a thin sheet meeting the surface at a shallow angle covers a strip about ``w^2 / w_0`` wide, so ``P / d``
gains far less from the window than the irradiance. A class light of unknown fate of its group and line may join has no
lower bound above 0.

Light left out as negligible. The trace and the probes do not follow branches of light at or below the line's level
(``negligible_power_fraction`` x the line input), and paths stop at the power floor; that light still goes somewhere, and a
faint branch focused onto a surface can reach an irradiance far above its share of the power. So does light of unknown
fate whose power is known (the trace's or the probes' budget, a path ended out of model scope, a line not launched, the
light a tabulated thin lens does not deliver into its focused beam): in a run that follows the untraced light it counts
with the left-out light (``_unfinished``). It is left out in pieces -- each branch not followed, each path ended at the
floor, each ledger entry or probe flow of unknown fate -- of one line, mutually coherent: where they meet their fields add
in amplitude, so a line counts with ``p* = (sum sqrt(p_k))^2`` (``SurfaceReach.left_out_root``, ``_FaintLine.root``;
``ExposureReport.faint_light``: ``power_W`` the pieces' total, ``in_phase_power_W`` that ``p*``), the lines of one class
adding their sums. That light enters every surface's classes of its line as a field of unknown phase, place and
polarization of amplitude at most ``A = c f sqrt(I_max(p*))`` (``_Faint``; the medium's index the larger of the
surface's two media). ``c`` is the most field its open paths' crossings bring onto a surface or across a plane
(``validation/resonance.py``, ``ReturnGains.crossing_field``): between two reflections a piece meets each lens and each
refracting surface between media of different index at most once (light they turn back onto one of them it already
passed, before it is reflected again, is outside the model) and travels in at most ``L = T + 1`` straight legs (``T``
the lenses and refracting surfaces that may turn it), each meeting any other surface at most ``K`` times (a plane once, a
sphere or a conic twice, an asphere by the degree of its sag along a line), so it meets the surface at most ``m`` times
after each number of reflections (1 for a lens or a refracting surface between different media, ``K L`` for any other
surface). After ``j`` reflections it is made of pieces from different sequences of ``j`` of the ``N`` surfaces that may
reflect it (distinct: no surface reflects it twice on an open path), mutually coherent, whose fields add: at most
``a_j`` times one crossing's, the smaller of the sum over those ordered sequences of the products of ``m_s sqrt(R_s)``
(``R_s`` a surface's reflectance bound, ``m_s`` its meetings) and the square root of their number (the products of
``m_s``) times the power they share (the sum of the products of ``m_s R_s``, at most 1), so ``c = sqrt(2) m sum_{j=0}^{N}
a_j`` (``sqrt(2)``: the light split between the two sides each time), at least 2 (the light once from either side) --
2.26 on the faces of four lenses with 0.25 % coatings. ``f`` is the factor by which two surfaces facing
each other may return it and send it on toward any surface (``validation/resonance.py``: ``f = 1 + sum g / (1 - g)``
over the scene's pairs, ``g`` a pair's round-trip gain bound; the faces of a window, a lens or a cube, a linear cavity;
``faint_light_return_factor``, ``FAINT_LIGHT_RECIRCULATES``): it raises each class's upper bound to ``sum v + 2 sum gamma
sqrt(v_i v_j) + 2 A sum sqrt(v_i) + A^2`` and lowers the bound at the beams' centres by the same amplitude (floored at
0), the window bound by what its power ``(c f)^2 P*`` can take away; its share of
the polarization condition below reads the same ``A``. A rated surface no followed light reaches is bounded by it alone
(between 0 and ``A^2``), never a known zero; ``faint_light_peak_W_per_cm2`` reports its share and
``FAINT_LIGHT_BOUNDED`` names it. Where the scene has a pair whose gain bound is 1 (two surfaces that reflect
everything, a coating met outside its data, an uncoated face met from inside up to its critical angle), nothing bounds
the light it may hold and send on: no upper bound on any surface, and a coating use the left-out light makes outside its
record (``validation/coating_use.py``) leaves its surface without one, with the surfaces of a pair that refusal leaves at a gain bound
of 1 (every other surface and every plane keeps the record's stated value); each such surface has the run-level notice
``EXPOSURE_BOUND_WITHHELD`` (``NOT_APPLICABLE``: no bound, comparison, target or status moves). Light at the per-path interaction limit (the trace's or a probe's) may be stored in a
cavity: it bounds nothing anywhere (no upper bound). Where the left-out light alone keeps a comparison from being decided (the rating
lies between the bounds with it and would not without it, or its share of the light decides a condition below), the run
follows those lines' light deeper (``ExposureOutcome.deeper_levels``: probe passes in ``validation/pipeline.py`` down to
a level that would make it small enough, at most ``MAX_DEEPER_PASSES``), so that the probes' known spots replace the
diffraction limit; what cannot be followed (below the power floor, beyond the probes' budget) keeps its bound.

The surface values (``SurfaceExposure``: incident power, peak irradiances, linear power density, peak fluence) are the
traced beams' combined upper bounds (their sum where no two can interfere). The incident power is a flux through the
surface: beams arriving from opposite sides add their powers (counter-propagating waves have no cross term in the
time-averaged Poynting flux), and the cross term of two coherent beams from one side at angles of incidence ``a_i``,
``a_j`` is bounded by ``2 gamma sqrt(P_i P_j) (cos a_i + cos a_j) / (2 sqrt(cos a_i cos a_j))`` (Cauchy-Schwarz over the
surface's area). ``coherent_groups`` lists the classes of two or more contributions with some nonzero degree of
coherence; one with a traced beam names ``COHERENT_BEAMS_BOUNDED`` and makes the surface's exposure a range
(``complete`` false).

When part of the launched power has an unknown fate (trace budget, interaction limit, a path ended out of model scope,
the light a tabulated thin lens does not deliver into its focused beam) the trace is incomplete (``complete`` false,
``INCOMPLETE_TRACE`` and the causes). In a run with power ratings the light the trace does not follow (untraced
reflections, a reflector coating's transmission, light passing an aperture or an element its chief ray misses) is followed
through the optics by probe traces down to the level of its line, as for an observation plane, each with the pupil rays of
two whole rings of the beam it leaves: where that light meets a surface it is listed in ``untraced_hits`` (every arrival,
whatever its power) and its Gaussian state is checked against those rays as a traced beam's is (an aberration-dominated
probe arrival counts with the diffraction limit of its power). Unknown-fate light of known power, the trace's or the
probes', is bounded with the left-out light (above); light at the interaction limit removes every upper bound, and the
classes of its lines have no lower bound either (it may be coherent with them). Without ratings no probe runs
(``untraced_light_included`` false), the values cover the traced beams only, and any unknown fate removes the upper
bound.

Rating comparison (``RatingComparison``; no scaling between conditions, no thermal or damage model). Every condition a
rating can state is recorded against the light on a surface (traced, untraced and beside it;
``RatingComparison.conditions``, ``validation/rating_conditions.py``): verified, known to differ (with what that says
about the damage threshold of this light), not verifiable (with why) or not stated. The regime (a ``CW`` rating
continuous sources, a ``PULSED`` rating pulsed ones), every beam's vacuum wavelength within ``rating_wavelength_tol_rel``
(2 %) of the rating's, for a pulsed rating every pulsed source's pulse duration within ``rating_pulse_duration_tol_rel``
(10 %) and, when the rating states one, its repetition rate within ``rating_repetition_rate_tol_rel`` (10 %) are verified
or differ with no established relationship. Followed light (traced, untraced, beside the surface) of lines the rating is
not for that carries at most ``rating_unlocated_light_share`` of the most light the surface can receive, in peak
irradiance and every quantity that gives a comparison its direction (``_line_share_quantities``: the rule for light left
out as negligible and for minor beams), counts in the exposure but not in these conditions (``_minor_lines``; a beam
without a value in one of them is never left out); with the left-out light of such lines it is not verifiable beyond
that share (``_faint_mismatch``). The test beam diameter is compared with every beam's 1/e^2 size
(``_beam_sizes``: both transverse diameters and the footprint's largest extent on the surface): verified within
``rating_beam_diameter_tol_rel`` (5 %) of a stated definition, a threshold at least the rated value for beams no larger
than the test beam, at most the rated value for beams larger under every reading of its definition, not verifiable
between the readings of a definition the rating does not state; light of unknown size bounded by its diffraction limit
(``_unsized_share``: untraced light of unknown size, light left out as negligible) withholds a verified or at-least
direction when it may carry more than ``rating_unlocated_light_share`` of the most light the surface can receive, and
the weakest beams that carry, with the followed light of unknown size, at most that share are left out of the comparison
(``_minor_beams``: a faint ghost of another size; the light left out as negligible is then checked with them, so beyond
that share it leaves the size not verifiable, never a ghost decisive) -- both shares in every quantity that gives the
size its direction (``_size_quantities``: the
rating's own, ``_share_quantities``, peak irradiance, ``P / d`` in both readings of an unstated definition or peak
fluence, since a micro-spot at a low repetition rate may carry little of the irradiance and most of the fluence; for a CW
rating also the power and ``P / d``, a broad beam's heating; for a pulsed one the peak fluence and the pulse energy; a
beam beside the surface with its tail bound in each), and a beam without a value in one of them is never left out. A
stated pulse count (read as the number of pulses each tested site received) is compared with the pulses one site of the
surface receives (``_pulses_per_site``): every crossing of pulsed light there brings its source's ``total_pulses``, except
the weakest crossings that together carry at most ``rating_unlocated_light_share`` of the most light the surface can
receive in every quantity of ``_line_share_quantities`` (the rule for minor beams and lines); a site receives at least the
fewest one counted crossing brings and at most all of them, unbounded where light left out as negligible may carry more
than that share or light may be stored (``rating_conditions.pulse_count_condition``). On a surface only light left out
as negligible reaches, that light's lines are the light the regime,
wavelength and pulse conditions compare (each with its pulse train; ``rating_conditions.scalar_conditions``); where light
may be stored between surfaces facing each other (no upper bound), a line the rating is not for -- of the left-out light,
or of the light that may be stored -- leaves each condition it differs in not verifiable whatever its power
(``_unbounded_mismatch``). The rating's test polarization
(``PowerRating.polarization``) holds for the light as a whole (``_polarization``): every beam within
``rating_polarization_fraction_tol`` (1 %) of one common state of it (s or p in its own plane of incidence; one linear
axis; one circular handedness), so that any incoherent mixture of the beams (their powers add) and any sum of beams in
one state is within it too. Mutually coherent beams in different states (``_interfering``: linked by a nonzero degree of
coherence, fields not parallel to rounding) add their fields with phases the model leaves open, and a beam's own share
does not bound their sum's: where their parts in the state cancel, the rest may be all the light (two equal beams at +2
and -2 deg to one axis sum to circular light for one phase). Their sum is checked for every phase, in field amplitudes
``a`` of the peak irradiance (and peak fluence for a ``J/cm2`` rating; ``_coherent_bound``): its part outside the state
is at most ``U = sum sqrt(f_i) a_i`` and its part in it at least ``W = sqrt(1 - f_j) a_j - sum_{i != j} sqrt(1 - f_i)
a_i`` (``f`` a beam's own share outside the state, ``j`` a traced beam), so at most ``U^2 / (U^2 + W^2)`` of it is
outside the state; in the state of one traced beam ``j`` it keeps at least ``B = a_j - sum a_i |e_j^* . e_i|`` and the
rest is at most ``D = sum a_i sqrt(1 - |e_j^* . e_i|^2)``, so at most ``(B sqrt(f_j) + D)^2 / (B^2 + D^2)`` (when ``B >
sqrt(f_j) D``) -- the smaller bound counts for each of those beams, in place of its own share; it holds at every point
only for beams of one amplitude profile on the surface (``_one_profile``; on a curved surface the same profile in space),
so beams displaced from each other, of different sizes or meeting the surface from different directions with a part
outside the state are not verified (their amplitude ratio, and their combined state, varies
across the surface), and a violation at a traced beam's centre for every phase is checked first. Beyond the tolerance the
condition is not verifiable (``COHERENT_STATE_UNDETERMINED``), never verified; within it (a beam with weak coherent
ghosts in other states) it holds and the note gives that bound. S / P are taken along the s vector of one traced beam
of the group, so beams in other planes of incidence count with their parts across it. Beams of another state that carry
at most that share of the most light the surface can receive, in peak irradiance and every quantity that gives the
comparison its direction, count in the exposure but not in the state (``_line_share_quantities``), and the shares below
then count them entirely outside it; and the light left out as negligible,
of unknown polarization, cannot carry more than that share of the most the light can reach (``sum_c (sqrt(f_c) S_c +
A_c)^2 <= tol sum_c (S_c + A_c)^2`` with ``S_c`` the class's summed amplitudes, ``f_c`` its beams' largest share outside
the state -- a coherent group's bound for its beams -- and ``A_c`` its left-out amplitude; in peak irradiance and in the
rating's own quantity, the larger share counting). The polarization differs only where the light at a traced beam's
centre violates it for certain (the least
share a mixture there can have in the orthogonal state, over the classes' ranges, above the tolerance). Left-out light
of lines the rating is not for counts likewise (``_faint_mismatch``; in peak irradiance and in the rating's own
quantity, the larger share counting): beyond that share of the most the light can reach, each condition such a line
differs in -- its regime, else its wavelength, pulse duration and repetition rate (``rating_conditions.line_differences``) -- is not
verifiable. A polarization stated only as text or a text that says more than the rating's ``polarization`` (an
orientation, a handedness: not a plain form of it, ``rating_polarization``), ``S`` / ``P`` at normal incidence, coherent
beams of different polarization states whose sum may leave the state, beams in different states whose mixture is
neither, light of unknown state that may reach the surface (unknown fate, left-out light beyond its share) leave the
polarization not verifiable.

A peak fluence is the energy of one pulse per area. A pulse holds every line of its source that crosses the surface
along one path (the model keeps a pulse's duration: its lines travel together); pulses of different sources, of one
source along different paths, or of one class's members (passes whose delays differ by more than a pulse) need not
arrive together. The lower bound of a peak fluence is that of one pulse: for each crossing (a source, its sequence of
interactions, its pass) the sum over the line classes of the bound of the class's light in the pulse that holds that
crossing's member ``k`` -- ``max(v_k - 2 sqrt(v_k) (sum_{j != k} gamma_kj sqrt(v_j) + A), (sqrt(v_k) - sum_{j != k}
sqrt(v_j) - A)^2)``, the class's other light at its worst -- and the largest over the crossings; the upper bound keeps
every pulse together (``_lower``, ``_class_range``).

The rated value is a damage threshold for the peak exposure only as its definition establishes one
(``rating_conditions.rated_threshold``, ``RatingComparison.rated_threshold_low`` / ``_high``): a ``J/cm2`` or ``W/cm2``
value stated as the peak (``value_definition`` ``PEAK``) is the threshold, one stated as the total over the area of the
test beam's 1/e^2 diameter (``TOTAL_PER_1E2_AREA``) half of it; one whose definition is not stated establishes none (the
``value_definition`` condition is not verifiable: the comparison is ``CONDITIONS_NOT_VERIFIED``, its exposure still
reported); a ``W/cm`` value is its own threshold; a ``TYPICAL`` value bounds none. How deep a run follows the light left
out as negligible is judged as if the rating stated no pulse count, and for a value of unstated definition against the
value and twice it (``_compare``): the exposure's bounds do not depend on the count, its protocol or the sources'
totals; a stated definition is judged against its own threshold, so the depth and the exposure's bounds may differ from
those of the same value unstated (valid either way).

The conditions give the rating's applicability (``rating_conditions.rating_applicability``): ``NOT_APPLICABLE`` (a
condition differs with no established relationship, or two in opposite directions: ``CONDITIONS_MISMATCH``),
``NOT_VERIFIED`` (``CONDITIONS_NOT_VERIFIED``, its exposure -- exact, or its bounds -- still reported),
``THRESHOLD_AT_LEAST_RATED`` (``WITHIN_RATING`` when the exposure's upper bound is at most ``rated_threshold_low``, else
``UNKNOWN``), ``THRESHOLD_AT_MOST_RATED`` (``EXCEEDS_RATING`` when its lower bound is above ``rated_threshold_high``, else
``UNKNOWN``) or ``AS_RATED`` (both). The exposure in the rating's unit: ``W/cm2`` the transverse peak irradiance, ``W/cm`` ``P / d`` with ``d`` of
the rating's diameter definition (1/e^2: ``2 w``; FWHM: ``w sqrt(2 ln 2)``; not stated: the range of both,
``_exposure_bounds``), ``J/cm2`` the peak fluence, bounded as above (exact when the bounds agree); it is computed unless
the regime, wavelength or pulse differs. Each surface that exceeds a rating raises one ``POWER_RATING_EXCEEDED`` notice
with the numbers of the rating whose damage threshold the exposure exceeds most (the threshold named where it is not
the rated value): its status is ``NOT_APPLICABLE``, so it does not change the run's status (an exceedance is reported,
never a gate); only a ``power_rating_utilization`` target turns the comparison into a target result.

A surface's utilization (exposure / damage threshold) lies between the largest lower bound of its ``AS_RATED`` and
at-most comparisons (a decided exceedance stands) and, when it has an ``AS_RATED`` or at-least comparison, the largest
upper bound of those and of the comparisons not verified (they may apply as stated and be the stricter ones;
``_surface_bounds``, ``_surface_status``). An at-most comparison -- and one not verified, which may have a threshold at
most its value -- leaves the surface no upper bound (its threshold may lie below the exposure) unless a comparison of the
same unit that applies as stated or with a threshold at least the rated value has a lowest threshold no larger than its
highest under every reading (``W/cm`` values compared in P / d of the 1/e^2 diameter, ``_linear_scale``): the threshold
then lies between the two, and that comparison bounds it (``_at_most_open``); comparisons of different units are criteria
of their own. Where an at-most comparison has only such comparisons whose lowest may be larger the ratings disagree, as
where the lower bound lies above the upper bound: the lower bound stands and there is no upper bound. An element's rated
surfaces are those at least one of its ratings applies to (a rating's ``surface_ids``, or every surface); its status and utilization
come from them alone, and its other surfaces are named in ``unrated_surface_ids``. Its utilization lies between the
largest of its surfaces' lower bounds and the largest of their upper bounds; it has no upper bound while a rated surface
has none. A ``power_rating_utilization`` target's message names what makes the utilization an interval
(``measure_utilization``): ratings that bound it from one side only (their test beam smaller or larger than the light),
ratings that disagree, conditions not verifiable, the two readings of an unstated ``W/cm`` diameter definition, the
exposure's own range; its value is then the largest exposure / damage threshold of the ratings that apply (the exposure
exact, or its lower bound), and it has no value when none of those exposures is known or bounded from below.

The arithmetic is plain IEEE ``+ - x /`` and ``sqrt`` on Python floats (no fused multiply-add) with the libm-free
exponential: the reported values have identical bits in every runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from ..config import NumericalConfig
from ..contracts.basis import Assumed, Omitted, ValidityCheck, ValueBasis
from ..contracts.components import PowerRating, PulseTrain, rating_polarization
from ..contracts.project import ProjectSnapshot, Target
from ..contracts.results import (
    CoherentExposureGroup,
    ElementExposure,
    ExposureHit,
    ExposureReport,
    FaintLight,
    PowerBudget,
    RatingComparison,
    RatingCondition,
    SurfaceExposure,
    UtilizationOpening,
)
from ..contracts.basis import Omitted, ValueBasis
from ..errors import ErrorCode, Status, ValidationIssue, make_issue
from ..errors.status import AssumedInput, OmittedPart, ValueKind
from ..math.exact_linalg import symmetric_eigenvalues_2x2
from ..math.vectors import dot, transverse_basis
from ..optics.interference.coherence import delays_agree, largest_coherence, nearest_delay, period_length_mm
from ..optics.rays.envelope import COHERENCE_K, COHERENCE_SQRT_K, NO_SPREAD, Spread, spread_at
from ..optics.rays.hits import SurfaceHit
from ..optics.polarization.partial import component_group, partially_polarized
from .observations import gaussian_ray_spread
from ..math.exact_trig import exp as exact_exp
from ..contracts.optics import MeshSurface
from ..scene.lower import coherence_length_from_linewidth_mm, facet_patch_ids
from ..units import NM_TO_MM
from ..optics.rays.tracer import SINK_INTERACTION_LIMIT, UNKNOWN_FATE_SINKS
from .power import UNKNOWN_FATE_CAUSES, bounded_cause, line_fate
from .rating_conditions import (
    APPLYING,
    AS_RATED,
    AT_LEAST,
    AT_MOST,
    COHERENT_STATE_UNDETERMINED,
    DEFAULT_TOLERANCES,
    DIFFERS,
    FWHM_PER_E2_DIAMETER,
    LIGHT_STATE_UNKNOWN,
    NO_RELATIONSHIP,
    NOT_APPLICABLE,
    NOT_STATED,
    NOT_VERIFIABLE,
    NOT_VERIFIED,
    PEAK_PER_TOTAL_OVER_E2_AREA,
    RATING_INFORMATION_MISSING,
    REASON_TEXT,
    VERIFIED,
    BeamSize,
    Light,
    PulsesPerSite,
    RatingTolerances,
    beam_diameter_decision,
    comparison_status,
    line_differences,
    pulse_count_condition,
    rated_threshold,
    rating_applicability,
    scalar_conditions,
    total_pulses,
    value_definition_condition,
)
from .coating_use import NO_AIR_EQUIVALENT, NOT_BOUNDED, OUTSIDE_WAVELENGTH, CoatingUse
from .descriptors import EXPOSURE_NOT_COUNTED
from .resonance import (
    COATING_DATA_RANGE,
    COATING_EVALUATION_ISSUE,
    INDEX_UNKNOWN,
    REFUSED_COATING_USE,
    ReturnGains,
    ReturnPair,
)
from .untraced import STORED_KINDS, SurfaceReach
from .window_bound import Anchor, WindowBeam, linear_lower, window_lower

#: The most surface pairs or closed returns a withheld-bound notice names; where there are more (the facing facets of a
#: solid of many faces), it names the first ones, in order, and counts them all (``return_pair_count``,
#: ``closed_return_count``), so a notice stays the same size whatever the number of pairs.
NAMED_RETURNS = 12


def closed_return_kinds(gains, wavelengths) -> dict[tuple[str, ...], str]:
    """What each closed return of ``gains`` (a mapping by wavelength) at ``wavelengths`` is (``ReturnGains.closed_return_kinds``:
    ``path``, ``parallel_stack``, ``solid_folds``)."""
    return {cycle: kind for w in wavelengths if gains and w in gains
            for cycle, kind in zip(gains[w].closed_cycles, gains[w].closed_return_kinds)}


def closed_return_text(kinds: dict[tuple[str, ...], str], path_join: str = " -> "):
    """The formatter of one closed return in words (``named_returns``): a closed path's surfaces in order (``path_join``),
    a stack's parallel surfaces or a solid's faces as a group -- no path runs between them in that order."""

    def text(cycle: tuple[str, ...]) -> str:
        kind = kinds.get(cycle, "path")
        if kind == "parallel_stack":
            return f"the parallel surfaces {', '.join(cycle)}"
        if kind == "solid_folds":
            return f"the faces {', '.join(cycle)} of one solid"
        return path_join.join(cycle)

    return text


def kinds_context(cycles, kinds: dict[tuple[str, ...], str]) -> dict:
    """``closed_return_kinds`` (``"; "`` between them, as ``closed_return_surfaces``) where a named return is not a path."""
    found = [kinds.get(cycle, "path") for cycle in cycles]
    return {"closed_return_kinds": "; ".join(found)} if any(kind != "path" for kind in found) else {}


def closed_return_facts(gains, wavelengths) -> tuple[str, dict]:
    """What a withheld-bound notice states about the closed returns at ``wavelengths`` besides their names: the clause of
    its message that says where any other closed return can run, and the context -- the surfaces of every group the trail
    search screens (``closed_return_candidates``, each
    group's ids joined by ", "; at most ``NAMED_RETURNS``, with ``closed_return_candidate_count``: closed returns of the
    screened kind run only among the surfaces of one group, whichever one the search names), the round gain bound of the
    trail it found (``closed_return_gain``: the product of its surfaces' reflectance bounds; the largest over the
    wavelengths) and, per solid whose folds have no certified sum, why (``solid_fold_reasons``, ``<owner>: <reason>``)."""
    groups = sorted({group for w in wavelengths for group in gains[w].search.candidates})
    found = [gains[w].search.cycle_gain for w in wavelengths if gains[w].search.cycle_gain is not None]
    reasons = sorted({f"{fold.owner_id}: {fold.uncertified}" for w in wavelengths for fold in gains[w].solid_folds
                      if getattr(fold, "uncertified", None) is not None})
    context: dict = {}
    if groups:
        context["closed_return_candidates"] = [", ".join(group) for group in groups[:NAMED_RETURNS]]
        context["closed_return_candidate_count"] = len(groups)
    if found:
        context["closed_return_gain"] = max(found)
    if reasons:
        context["solid_fold_reasons"] = reasons
    # (a fact of the message: no step inside the tool obtains this bound, so the notice has no action)
    if groups:
        among = "; ".join(context["closed_return_candidates"]) + ("; ..." if len(groups) > NAMED_RETURNS else "")
        where = (f"the search names the shortest closed return it finds, and any closed return between separate elements runs "
                 f"only among the surfaces {among}")
    else:
        where = "the search names the shortest closed return it finds, and other closed returns may remain"
    return where, context


def named_returns(items, each, noun: str) -> tuple[list, str, dict]:
    """``(named items, text, context)`` of the sorted surface pairs or closed returns ``items``: all of them, ``each``
    formatting one and ``"; "`` between them, or, beyond ``NAMED_RETURNS``, the first ``NAMED_RETURNS`` within
    ``"<count> <noun> (among them ...)"`` with the count in ``context`` (``return_pair_count`` / ``closed_return_count``)."""
    items = list(items)
    shown = items[:NAMED_RETURNS]
    text = "; ".join(each(item) for item in shown)
    if len(items) <= NAMED_RETURNS:
        return shown, text, {}
    key = "return_pair_count" if noun.startswith("pair") else "closed_return_count"
    return shown, f"{len(items)} {noun} (among them {text})", {key: len(items)}


#: |d x n| at or below which a beam meets a surface at normal incidence: no plane of incidence, s and p not defined.
NORMAL_INCIDENCE_SIN = 1e-9
#: A coherent class's lower bound at most this share of its members' summed values is taken as 0 (rounding of a cancellation).
CANCELLATION_ROUNDING = 1e-12
#: Two fields whose share in each other's polarization state (``_overlap2``) is within this of 1 are in one state (the
#: rounding of their Jones vectors; their sum differs from it by at most the rounding of a cancellation).
PARALLEL_ROUNDING = CANCELLATION_ROUNDING
#: The smallest |cos(AOI)| a flux factor of two coherent beams uses (a grazing beam carries no flux through the surface).
GRAZING_COS = 1e-9
#: The smallest |cos(AOI)| at which a beam's footprint on a surface is stated: the footprint's long axis is w / cos(AOI),
#: and cos(AOI) carries the rounding of the beam's direction and the surface's normal (about 1e-15, which follows the last
#: bits of the poses); below 1e-5 (an AOI within 6e-4 deg of grazing) that is more than 1e-10 of the footprint, and the
#: footprint is not known (``None``, as at grazing incidence).
FOOTPRINT_COS_RESOLUTION = 1e-5
#: A pair whose chief rays' degree of coherence is at least this is taken as fully coherent (an upper bound).
COHERENT_SHORTCUT = 0.999
#: Radii (in units of the squared 1/e^2 radius: outside sqrt(K) w the profile amplitude is below exp(-K)) over which a pair's
#: degree of coherence is bounded (``_coherence``; ``optics/rays/envelope.py``: ``COHERENCE_K``, the same sizes as the
#: spread of a beam's envelope delay).
_COHERENCE_TAIL = tuple(float(v) for v in exact_exp(np.array([-k for k in COHERENCE_K])))
#: ``(OPD / L_c)^2`` at or below which the chief rays' degree of coherence exp(-(OPD / L_c)^2) is at least
#: ``COHERENT_SHORTCUT`` (0.999): -ln(0.999).
COHERENT_SHORTCUT_RATIO2 = 0.0010005003335835335
#: A degree of coherence below this changes no sum it enters (its term is below half a unit in the last place): 0.
GAMMA_NEGLIGIBLE = 2.0 ** -55
#: A relative delay of at least this many coherence lengths gives exp(-ratio^2) below GAMMA_NEGLIGIBLE (with the largest
#: radius's profile tail exp(-64)).
INCOHERENT_RATIO = 6.3
#: A deeper probe pass follows the lines concerned down to this factor below the level the decision needs (headroom for
#: the branches that multiply below it).
DEEPER_HEADROOM = 4.0
MM2_PER_CM2 = 100.0
MM_PER_CM = 10.0
PERCENT = 100.0
#: FWHM diameter of a Gaussian intensity profile in units of its 1/e^2 radius: sqrt(2 ln 2).
FWHM_PER_RADIUS = 1.1774100225154747
EXPOSURE_MODEL = (
    "chief-ray packet power and paraxial Gaussian beam at each surface hit: peak irradiance 2P/(pi w_x w_y) in the transverse "
    "plane (x cos(AOI) on the surface; P the average power, so for a pulsed source the spatial peak of the average "
    "irradiance), P/d, peak fluence 2E/(pi w_x w_y) with E = P / repetition rate (one pulse's); light of different lines or "
    "coherence groups adds, in a peak fluence the lines of one source along one path (one pulse), while pulses of different "
    "sources or paths need not arrive together and the lower bound is the strongest pulse's; mutually coherent beams (one coherence group and line, from either side: a standing wave) add their fields "
    "with open phases: between max(0, sqrt(v_k) - sum sqrt(v_j))^2 and sum v + 2 sum gamma sqrt(v_i v_j) (gamma the largest "
    "degree of coherence over the surface from the declared coherence length, else 1; a pulse train coherent again at "
    "multiples of its period length); upper bounds at the beams' peaks, lower bounds at the traced beams' centres and a "
    "traced beam's mean over a window on its area, which light of unknown place lowers only by what its power can take "
    "away there (in P/d, as Gaussian beams crossing the surface from one side, by the strip each covers); incident power a "
    "flux (no cross term between the two sides); a J/cm2 or W/cm2 rated value is a threshold for the peak exposure as its "
    "stated definition gives it (a peak value: the value; the total over the 1/e^2 area of a Gaussian test beam: twice it) "
    "and no threshold when the rating does not state it, a typical value no threshold; with power ratings, light the trace does not follow is followed to the surfaces and bounds the exposure, and light left out as "
    "negligible or of unknown fate of known power adds a field of at most the amplitude of its diffraction limit "
    "2 pi n^2 P / lambda^2 (P = (sum of the square roots of its pieces' powers)^2: they are mutually coherent) times the most "
    "its open paths' crossings bring onto a surface or across a plane, sqrt(2) m sum_{j=0}^{N} a_j, at least 2 (m the "
    "meetings after each number of reflections, from either side: 1 on a lens or a refracting surface between different "
    "media, on any other surface as often per straight leg between two reflections as a straight line meets it -- once a "
    "plane, twice a sphere or a conic, an asphere by the degree of its sag; a_j the most field of its pieces after j "
    "reflections, from different sequences of the N surfaces that may reflect it, added in phase: the smaller of the sum "
    "over those sequences of the products of m_s sqrt(R_s) and the square root of their number times the power they share), "
    "and times the factor of the returns between two surfaces facing each other, which leave them toward every surface "
    "(followed deeper where that decides a comparison; light at the interaction limit, and a pair whose gain bound is 1, "
    "bound nothing: a pair's surface and every other then has no upper bound and the run names each with "
    "EXPOSURE_BOUND_WITHHELD, which changes no bound, target or status; a coating whose record states an air-equivalent "
    "angle range is read, from a medium denser than air, at the air-equivalent angles asin(n sin(theta)) of the left-out light's "
    "own bounded extent on its way to its next reflection, and the surface and a pair that refusal leaves at a gain bound of 1 "
    "have no exposure upper bound where that extent leaves the record's range or has no real air-equivalent angle, a plane's "
    "power bound keeping the record's stated value); closed returns through three or more surfaces of separate elements (a ring, a folded or coupled "
    "cavity, a periscope's windows through its folds) and stacks of parallel planes are screened: a stack keeps an "
    "all-orders series of its returns where one converges, and a possible return without one leaves every surface it may "
    f"reach without an upper bound (EXPOSURE_BOUND_WITHHELD); {EXPOSURE_NOT_COUNTED}; coherence from the group delay; no thermal "
    "or damage model")
#: Severity of a rating status (most severe first when aggregating an element).
RATING_SEVERITY = {"EXCEEDS_RATING": 5, "UNKNOWN": 4, "CONDITIONS_NOT_VERIFIED": 3, "CONDITIONS_MISMATCH": 2, "RATING_MISSING": 1,
                   "WITHIN_RATING": 0}
POWER_RATING_UTILIZATION = "power_rating_utilization"
#: Target metrics of a response model's ``supported_metrics`` that validate the beam size behind it: one of these sets.
BEAM_SIZE_METRIC_SETS = (frozenset({"beam_radius_mm"}), frozenset({"beam_radius_x_mm", "beam_radius_y_mm"}))
#: The power metric a response model's ``supported_metrics`` validate the power behind it with (its ``transmittance``).
POWER_METRIC = "power_W"
#: Kinds of light on a surface: the traced beams, untraced light the probes follow there, beams beside the surface.
TRACED, UNTRACED, BESIDE = "traced", "untraced", "beside"
_POLARIZATION_TEXT = {"S": "s-polarized", "P": "p-polarized", "LINEAR": "linear", "CIRCULAR": "circular", "UNPOLARIZED": "unpolarized"}

#: (source_id, spectral_index) of a source line.
LineKey = tuple[str, int]
#: (coherence group, vacuum wavelength nm) of a source line.
LineTag = tuple[str, float]


@dataclass(frozen=True)
class _HitValues:
    hit: ExposureHit
    pulsed: bool
    pulse: PulseTrain | None
    w_min: float | None  # the narrower 1/e^2 radius (mm), None when unknown
    kind: str | None = None  # untraced light: the kind of untraced branch it started as
    raw: SurfaceHit | None = None
    light: str = TRACED
    # untraced light whose size is not known: its values are the diffraction limit of its power (medium index ``n_max``)
    limited: bool = False
    n_max: float = 1.0
    # a beam beside the surface: the share of its peak values that reaches it, exp(-2 d^2 / w_max^2) (1 otherwise)
    tail: float = 1.0
    # a traced beam: (the surface's radius, None on a plane; the distance from its hit to the clear aperture's edge; on a
    # sphere the largest difference of two normals in its clear aperture) where the window bound applies
    # (``validation/window_bound.py``: ``surface_room``), else None
    room: tuple[float | None, float, float | None] | None = None

    @property
    def located(self) -> bool:
        """A beam meeting the surface with a known Gaussian profile there (its value at any point is known)."""
        return self.light != BESIDE and self.w_min is not None and not self.limited

    @property
    def ref(self) -> str:
        return f"{self.light}:{self.hit.path_id}:{self.hit.pass_index}"


def exposure_validated(validated: frozenset[str]) -> bool:
    """Whether a response model validated for the target metrics ``validated`` covers the exposure behind it: the
    utilization itself, or the power together with the beam size (module docstring)."""
    return POWER_RATING_UTILIZATION in validated or (
        POWER_METRIC in validated and any(group <= validated for group in BEAM_SIZE_METRIC_SETS))


def _unvalidated(h: SurfaceHit, metric_scope: dict[str, frozenset[str]]) -> tuple[str, ...]:
    """The response-model surfaces on the hit's way that do not validate the exposure behind them."""
    crossed: list[str] = []
    for step in h.lineage:
        surface = step.rsplit(":", 1)[0]
        if surface in metric_scope and surface not in crossed and not exposure_validated(metric_scope[surface]):
            crossed.append(surface)
    return tuple(crossed)


def _footprint(h: SurfaceHit, wx: float, wy: float, cfg: NumericalConfig) -> tuple[float, float] | None:
    n = h.normal / math.sqrt(float(dot(h.normal, h.normal)))
    u, v = transverse_basis(n[None, :], cfg.zero_vector_tol)
    u, v = u[0], v[0]
    bxu, bxv = float(dot(h.gstate.bx, u)), float(dot(h.gstate.bx, v))
    byu, byv = float(dot(h.gstate.by, u)), float(dot(h.gstate.by, v))
    wx2, wy2 = wx * wx, wy * wy
    a = bxu * bxu / wx2 + byu * byu / wy2
    b = bxu * bxv / wx2 + byu * byv / wy2
    c = bxv * bxv / wx2 + byv * byv / wy2
    # (the determinant a c - b^2 cancels to cos^2(AOI) of its terms: it is taken from the projected axes, |bx_u by_v -
    # bx_v by_u| = cos(AOI), whose rounding is that of the beam's direction)
    obliquity = bxu * byv - bxv * byu
    if abs(obliquity) < FOOTPRINT_COS_RESOLUTION:
        return None  # grazing: the footprint is unbounded (or longer than the beam's direction resolves)
    _small, big = symmetric_eigenvalues_2x2(a, b, c)
    small = obliquity * obliquity / (wx2 * wy2 * big)
    return (1.0 / math.sqrt(small), 1.0 / math.sqrt(big))


def _hit_values(h: SurfaceHit, pass_index: int, pulse: PulseTrain | None, cfg: NumericalConfig,
                metric_scope: dict[str, frozenset[str]], kind: str | None = None, light: str = TRACED,
                n_max: float | None = None, converters: frozenset[str] = frozenset()) -> _HitValues:
    """One contribution's values. ``n_max`` (untraced light, ``kind`` given): the largest index of the surface's media at
    its wavelength -- an untraced beam whose size is not known (status ``UNKNOWN``) is then bounded by the diffraction limit
    of its power (``_HitValues.limited``). ``converters``: the crystal surfaces whose second-harmonic generation is out of
    model scope (the reason code of a hit behind one, ``_power_scope_reason``)."""
    values = _hit_values_of(h, pass_index, pulse, cfg, metric_scope, kind, light, converters)
    if kind is not None and n_max is not None and values.hit.status is Status.UNKNOWN and values.w_min is None:
        return replace(values, limited=True, n_max=n_max)
    return values


def _power_scope_reason(h: SurfaceHit, converters: frozenset[str]) -> str:
    """The reason code of a hit whose power is out of model scope (``SurfaceHit.power_scope``)."""
    if any(surface in converters for surface in h.power_scope):
        return ErrorCode.NONLINEAR_CONVERSION_OUT_OF_SCOPE.value
    return ErrorCode.OUT_OF_MODEL_SCOPE.value


def _hit_values_of(h: SurfaceHit, pass_index: int, pulse: PulseTrain | None, cfg: NumericalConfig,
                   metric_scope: dict[str, frozenset[str]], kind: str | None, light: str,
                   converters: frozenset[str] = frozenset()) -> _HitValues:
    base = dict(path_id=h.path_id, source_id=h.source_id, spectral_index=h.spectral_index, wavelength_nm=h.wavelength_nm,
                pass_index=pass_index, incident_power_W=h.power_W, aoi_deg=h.aoi_deg,
                pulse_energy_J=None if pulse is None or h.power_scope else h.power_W / pulse.repetition_rate_hz)
    if h.power_taken_upper_W is not None:
        # (surfaces took a bounded share of the beam before the hit: the incident power is the beam's, an upper value)
        base["value_basis"] = {"incident_power_W": ValueBasis(kind=ValueKind.UPPER_BOUND, omitted=(Omitted(
            part=OmittedPart.BEAM_PART_ELSEWHERE, at=tuple(h.power_taken_at), upper=h.power_taken_upper_W,
            code=ErrorCode.POWER_TAKEN_BEFORE_PLANE.value),))}
    g = h.gstate
    reason: str | None = None
    status = Status.UNKNOWN
    if h.miss_distance_mm is not None:
        return _beside_values(h, base, pulse, metric_scope, kind, converters)
    if _unvalidated(h, metric_scope):
        reason, status = ErrorCode.METRIC_NOT_VALIDATED_FOR_ELEMENT.value, Status.OUT_OF_MODEL_SCOPE
    elif h.power_scope:  # (the beam's power is not the light's: no exposure, no bound of it)
        reason, status = _power_scope_reason(h, converters), Status.OUT_OF_MODEL_SCOPE
    elif h.chief_obstructed:
        reason = ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION.value
    elif not g.valid:
        reason = (g.reason or ErrorCode.OUT_OF_MODEL_SCOPE).value
    else:
        spread = gaussian_ray_spread(h, cfg)  # type: ignore[arg-type]  # SurfaceHit carries the ArrivalRecord fields it reads
        if spread is not None and spread.ratio > cfg.gaussian_max_ray_spread_ratio:
            reason = ErrorCode.GAUSSIAN_ABERRATION_DOMINATED.value
    if reason is not None:
        return _HitValues(ExposureHit(**base, status=status, reason=reason), pulse is not None, pulse, None, kind, h, light)
    wx, wy = g.radius("x"), g.radius("y")
    peak = 2.0 * h.power_W / (math.pi * wx * wy) * MM2_PER_CM2
    fluence = None if pulse is None else 2.0 * (h.power_W / pulse.repetition_rate_hz) / (math.pi * wx * wy) * MM2_PER_CM2
    hit = ExposureHit(**base, beam_radius_x_mm=wx, beam_radius_y_mm=wy, footprint_semi_axes_mm=_footprint(h, wx, wy, cfg),
                      peak_irradiance_W_per_cm2=peak, peak_irradiance_surface_W_per_cm2=peak * h.cos_aoi,
                      linear_power_density_W_per_cm=h.power_W / (2.0 * min(wx, wy)) * MM_PER_CM, peak_fluence_J_per_cm2=fluence)
    return _HitValues(hit, pulse is not None, pulse, min(wx, wy), kind, h, light)


def _beside_values(h: SurfaceHit, base: dict, pulse: PulseTrain | None, metric_scope: dict[str, frozenset[str]],
                   kind: str | None, converters: frozenset[str] = frozenset()) -> _HitValues:
    """A beam whose chief ray misses the surface while its footprint reaches it (``SurfaceHit.miss_distance_mm``): the
    Gaussian profile at the support's nearest point, ``exp(-2 d^2 / w_max^2)`` of the peak values, bounds what reaches the
    surface from above (no ``P / d``). Without a valid Gaussian state (or behind a response model not validated for the beam
    size) nothing bounds it."""
    g, d = h.gstate, h.miss_distance_mm
    assert d is not None
    base = {**base, "miss_distance_mm": d if d != math.inf else None}
    if _unvalidated(h, metric_scope):
        return _HitValues(ExposureHit(**base, status=Status.OUT_OF_MODEL_SCOPE, reason=ErrorCode.METRIC_NOT_VALIDATED_FOR_ELEMENT.value),
                          pulse is not None, pulse, None, kind, h, BESIDE)
    if h.power_scope:
        return _HitValues(ExposureHit(**base, status=Status.OUT_OF_MODEL_SCOPE, reason=_power_scope_reason(h, converters)),
                          pulse is not None, pulse, None, kind, h, BESIDE)
    if not g.valid or h.chief_obstructed or d == math.inf or h.cos_aoi < FOOTPRINT_COS_RESOLUTION:
        # (no valid Gaussian state: the beam's extent and profile beyond its pupil rays are not known -- named by its reason;
        # a beam grazing the surface's plane crosses it where its direction does not resolve, 1 / cos(AOI) away: its size
        # there is not known either)
        reason = (g.reason.value if not g.valid and g.reason is not None else ErrorCode.INCOMPLETE_ENVELOPE.value)
        return _HitValues(ExposureHit(**base, status=Status.UNKNOWN, reason=reason), pulse is not None, pulse, None, kind, h, BESIDE)
    wx, wy = g.radius("x"), g.radius("y")
    w_max = max(wx, wy)
    factor = float(exact_exp(-2.0 * d * d / (w_max * w_max)))  # libm-free: the bound is reported
    peak = 2.0 * h.power_W / (math.pi * wx * wy) * MM2_PER_CM2 * factor
    fluence = None if pulse is None else 2.0 * (h.power_W / pulse.repetition_rate_hz) / (math.pi * wx * wy) * MM2_PER_CM2 * factor
    hit = ExposureHit(**base, beam_radius_x_mm=wx, beam_radius_y_mm=wy, peak_irradiance_W_per_cm2=peak,
                      peak_irradiance_surface_W_per_cm2=peak * h.cos_aoi, peak_fluence_J_per_cm2=fluence,
                      reason=ErrorCode.INCOMPLETE_ENVELOPE.value)
    return _HitValues(hit, pulse is not None, pulse, None, kind, h, BESIDE, tail=factor)


# ------------------------------------------------------------------------------------------- quantities


@dataclass(frozen=True)
class _Quantity:
    """A value of the light on a surface: ``peak`` (transverse peak irradiance, W/cm2), ``surface`` (peak irradiance on the
    surface), ``fluence`` (peak fluence, J/cm2), ``linear`` (P / d, W/cm; ``definition`` ``FWHM``, else 1/e^2), ``power``
    (incident power, W) or ``energy`` (pulse energy, J). A beam beside the surface counts with the share of its power and
    pulse energy its tail bound gives (``_HitValues.tail``); light of unknown size or state has no power or energy here
    (its diffraction limit's power when it is bounded by it). ``weighed``: a ``P / d`` the beam-size comparison weighs
    beams in (``_size_quantities``), where a beam beside the surface counts with its tail bound too -- that share of its
    power over its own narrower diameter (an exposure's ``P / d`` has no value for it)."""

    kind: str
    definition: str | None = None
    weighed: bool = False

    def __call__(self, v: _HitValues) -> float | None:
        if v.limited:
            energy = None if v.pulse is None else v.hit.incident_power_W / v.pulse.repetition_rate_hz
            length = v.raw.coherence_length_mm if v.raw is not None else None
            return self.limit(v.hit.incident_power_W, energy, v.n_max, v.hit.wavelength_nm,
                              spectral_factor(v.hit.wavelength_nm, length, v.pulse))
        if self.kind == "peak":
            return v.hit.peak_irradiance_W_per_cm2
        if self.kind == "surface":
            return v.hit.peak_irradiance_surface_W_per_cm2
        if self.kind == "fluence":
            return v.hit.peak_fluence_J_per_cm2
        if self.kind in ("power", "energy"):
            if v.hit.peak_irradiance_W_per_cm2 is None:
                return None  # (no valid Gaussian state: neither its size nor what of it reaches the surface is known)
            amount = v.hit.incident_power_W if self.kind == "power" else v.hit.pulse_energy_J
            return None if amount is None else amount * v.tail
        if v.light == BESIDE and self.weighed and v.hit.peak_irradiance_W_per_cm2 is not None:
            # (the share of its power its tail bound gives, exp(-2 d^2 / w_max^2), over its narrower diameter)
            w = min(v.hit.beam_radius_x_mm, v.hit.beam_radius_y_mm)  # type: ignore[type-var]
            across = FWHM_PER_RADIUS * w if self.definition == "FWHM" else 2.0 * w
            return v.hit.incident_power_W * v.tail / across * MM_PER_CM
        if v.w_min is None or v.light == BESIDE:
            return None
        if self.definition == "FWHM":
            return v.hit.incident_power_W / (FWHM_PER_RADIUS * v.w_min) * MM_PER_CM
        return v.hit.linear_power_density_W_per_cm

    def limit(self, power_W: float, energy_J: float | None, n: float, wavelength_nm: float, spectral: float = 1.0) -> float | None:
        """The diffraction limit of light of ``power_W`` (pulse energy ``energy_J``) in a medium of index ``n``: its value
        focused to the smallest waist ``w = lambda / (pi n)`` (module docstring), with ``spectral`` the mean of
        ``(lambda / lambda')^2`` over its spectrum (``spectral_factor``: the limit of each spectral part summed); ``None`` for
        a fluence of continuous light; a power or a pulse energy is not concentrated by focusing (its own value)."""
        if self.kind == "power":
            return power_W
        if self.kind == "energy":
            return energy_J
        w = limit_waist(wavelength_nm, n, spectral)
        if self.kind == "fluence":
            return None if energy_J is None else 2.0 * energy_J / (math.pi * w * w) * MM2_PER_CM2
        if self.kind == "linear":
            if self.definition == "FWHM":
                return power_W / (FWHM_PER_RADIUS * w) * MM_PER_CM
            return power_W / (2.0 * w) * MM_PER_CM
        return 2.0 * power_W / (math.pi * w * w) * MM2_PER_CM2


def limit_waist(wavelength_nm: float, n: float, spectral: float = 1.0) -> float:
    """The diffraction-limited waist ``lambda / (pi n)`` (mm) of light at vacuum wavelength ``wavelength_nm`` in a medium
    of index ``n``, over the square root of its spectral factor (``spectral_factor``)."""
    return wavelength_nm * NM_TO_MM / (math.pi * n * math.sqrt(spectral))


#: ln 2 (the Gaussian pulse's spectral width).
LN2 = 0.6931471805599453
#: The speed of light in vacuum (mm/s).
C_MM_PER_S = 2.99792458e11


def spectral_factor(wavelength_nm: float, coherence_length_mm: float | None, pulse: PulseTrain | None) -> float:
    """The mean of ``(k / k0)^2`` over a line's spectrum (``k0 = 2 pi / lambda``), which scales its diffraction limit: the
    Gaussian spectrum of a declared coherence length ``L_c`` (``exp(-((k - k0) L_c / 2)^2)`` in power, the coherence model's
    ``exp(-(delay / L_c)^2)``) gives ``1 + lambda^2 / (2 pi^2 L_c^2)``; a pulse's transform-limited Gaussian spectrum of
    intensity FWHM ``tau`` gives ``1 + ln2 lambda^2 / (2 pi^2 c^2 tau^2)`` (a 10 fs pulse at 800 nm: 1.0025); the larger of
    the two, 1 for a line of neither."""
    lam = wavelength_nm * NM_TO_MM
    factor = 1.0
    if coherence_length_mm is not None and coherence_length_mm > 0.0:
        factor = max(factor, 1.0 + lam * lam / (2.0 * math.pi * math.pi * coherence_length_mm * coherence_length_mm))
    if pulse is not None:
        reach = C_MM_PER_S * pulse.pulse_duration_s
        factor = max(factor, 1.0 + LN2 * lam * lam / (2.0 * math.pi * math.pi * reach * reach))
    return factor


Quantity = _Quantity
_peak = _Quantity("peak")
_peak_surface = _Quantity("surface")
_fluence = _Quantity("fluence")
_power = _Quantity("power")
_energy = _Quantity("energy")
#: A share's quantity by its kind (messages).
_SHARE_NAME = {"peak": "peak irradiance", "surface": "peak irradiance on the surface", "fluence": "peak fluence",
               "linear": "P / d", "power": "power", "energy": "pulse energy"}


def _linear(definition: str | None) -> _Quantity:
    return _Quantity("linear", "FWHM" if definition == "FWHM" else None)


def _quantity(rating: PowerRating) -> _Quantity:
    if rating.unit == "W/cm2":
        return _peak
    if rating.unit == "J/cm2":
        return _fluence
    return _linear(rating.beam_diameter_definition)


# ------------------------------------------------------------------------------------------- coherent combination


def _dot3(a, b) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _side(h: SurfaceHit) -> int:
    """+1 when the chief ray travels along the surface's normal (d . n > 0), else -1."""
    return 1 if _dot3(h.direction, h.normal) > 0.0 else -1


@dataclass(frozen=True)
class _Pair:
    """The degree of coherence of two contributions of one class: ``gamma``, its largest value anywhere on the surface
    where both carry light (``_coherence``), and the terms that bound it at one point (``at``). ``fixed``: ``gamma`` holds
    everywhere (1: not declared, not known, or about 1 at the chief rays)."""

    gamma: float
    fixed: bool = True
    tau: float = 0.0  # |chief-ray optical path difference| (mm)
    length: float = 1.0  # the smaller coherence length (mm)
    slope: float = 0.0  # |n_i d_i - n_j d_j| (on a plane its part along the plane): the delay's change per mm
    offset: float = 0.0  # n_j |(c_i - c_j) . d_j|
    curvature_i: float = 0.0  # n_i max(1 / |R_i|) / 2
    curvature_j: float = 0.0
    separation: float = 0.0  # |c_i - c_j| (mm)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)  # c_i
    # the spread of each beam's envelope delay (``optics/rays/envelope.py``), its narrower 1/e^2 radius (mm), and the
    # chief rays' relative group-delay dispersion (mm^2)
    envelope_i: Spread = NO_SPREAD
    envelope_j: Spread = NO_SPREAD
    radius_i: float = 1.0
    radius_j: float = 1.0
    dispersion: float = 0.0

    def spread(self, r: float) -> float:
        """The largest change of the relative delay (mm) within ``r`` of the first chief ray's hit."""
        rj = r + self.separation
        return self.slope * r + self.offset + self.curvature_i * r * r + self.curvature_j * rj * rj

    @property
    def dispersed(self) -> bool:
        """Whether a beam's envelope delay spreads across it or the two paths' dispersion differs."""
        return self.dispersion != 0.0 or self.envelope_i is not NO_SPREAD or self.envelope_j is not NO_SPREAD

    def envelope(self, r: float) -> tuple[float, float, float, float]:
        """Within ``r`` of the first chief ray's hit: the largest change of the two envelope delays' difference beyond the
        wavefronts' (mm), the range (low, high) of the relative group-delay dispersion (mm^2), from the spreads of each
        beam's ray family that holds the points there, and the amplitude the points beyond a beam's largest kept family
        may still carry there (``exp(-K)`` of its peak, ``K`` the largest family: the cross term of a pair is bounded by it
        there whatever the delay; 0 when both families are kept)."""
        largest = COHERENCE_K[-1]
        ka = (r / self.radius_i) * (r / self.radius_i)
        rj = r + self.separation
        kb = (rj / self.radius_j) * (rj / self.radius_j)
        a = spread_at(self.envelope_i, min(ka, largest))
        b = spread_at(self.envelope_j, min(kb, largest))
        assert a is not None and b is not None
        beyond = _COHERENCE_TAIL[-1] if ka > largest or kb > largest else 0.0
        return a[0] + b[0], self.dispersion - a[1] - b[1], self.dispersion + a[1] + b[1], beyond

    def at(self, p) -> float:
        """A bound of the degree of coherence at the point ``p`` of the surface (as a factor of the two beams' amplitudes
        there: 0 where the pair's cross term is below the rounding of any sum everywhere, ``gamma`` 0)."""
        if self.fixed or self.gamma == 0.0:
            return self.gamma
        dx, dy, dz = float(p[0]) - self.origin[0], float(p[1]) - self.origin[1], float(p[2]) - self.origin[2]
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        gap = self.tau - self.spread(r)
        low = high = beyond = 0.0
        if self.dispersed:
            spread, low, high, beyond = self.envelope(r)
            gap -= spread
        if gap <= 0.0:
            gap = 0.0
            if not self.dispersed:
                return 1.0
        ratio = gap / self.length
        value = float(exact_exp(-(ratio * ratio))) if not self.dispersed else \
            min(1.0, largest_coherence(ratio, low, high, self.length) + beyond)
        return 0.0 if value < GAMMA_NEGLIGIBLE else value


@dataclass(frozen=True)
class _Wave:
    """What a contribution's degree of coherence with another one reads (``_coherence``): its declared coherence length
    (``None``: not declared), chief-ray optical path (``None``: not defined), whether its Gaussian state is known, the
    medium's index, its direction, hit point and surface normal (tuples), whether the surface is a plane, ``n max(1 /
    |R|) / 2`` of its wavefront and its larger 1/e^2 radius over the cosine of its angle of incidence (mm)."""

    length: float | None
    opl: float | None
    known: bool
    n: float = 1.0
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    flat: bool = False
    curvature: float = 0.0
    width: float = 0.0
    envelope: Spread = NO_SPREAD
    radius: float = 1.0  # the narrower 1/e^2 radius (mm)
    dispersion: float = 0.0  # the chief ray's group-delay dispersion (mm^2)
    delays: tuple[str, ...] = ()  # the passes of undetermined group delay on its way (``SurfaceHit.delays``)
    period: float | None = None  # a pulsed source's period length c / f_rep (mm): its pulses may be mutually coherent


def _wave(v: _HitValues, continuous: frozenset[str] = frozenset()) -> _Wave:
    """What a contribution's coherence reads; a line of a source of continuous spectrum (``continuous``: their source ids) as
    if no coherence length were declared, fully coherent with its partners: its line's width is the quadrature's bin, not a
    property of the light, so the line model does not bound the coherence of its share of the spectrum."""
    h = v.raw
    assert h is not None
    if h.source_id in continuous:
        return _Wave(None, h.opl_mm if h.opl_defined else None, False, delays=h.delays)
    # the relative delay of two fields' envelopes: their chief rays' group optical paths (sum n_g x length; in media without
    # dispersion the optical paths) and how much each envelope delay spreads across its beam (``optics/rays/envelope.py``):
    # light that crossed a dispersive medium other than a plate (behind a lens, a prism) has no bounded spread (known False)
    group = h.group_opl_mm if h.group_opl_mm is not None or h.dispersive else h.opl_mm
    opl = group if h.opl_defined else None
    g = h.gstate
    period = None if v.pulse is None else period_length_mm(v.pulse.repetition_rate_hz)
    if h.coherence_length_mm is None or opl is None or not g.valid or h.chief_obstructed or h.envelope_spread is None:
        return _Wave(h.coherence_length_mm, opl, False, delays=h.delays, period=period)
    curvature = 0.0
    for q in (g.qx, g.qy):
        c = abs(q.real) / (q.real * q.real + q.imag * q.imag)
        if c > curvature:
            curvature = c
    return _Wave(h.coherence_length_mm, opl, True, g.n, (float(h.direction[0]), float(h.direction[1]), float(h.direction[2])),
                 (float(h.position[0]), float(h.position[1]), float(h.position[2])),
                 (float(h.normal[0]), float(h.normal[1]), float(h.normal[2])), h.surface_flat, 0.5 * g.n * curvature,
                 max(g.radius("x"), g.radius("y")) / max(h.cos_aoi, GRAZING_COS), h.envelope_spread,
                 min(g.radius("x"), g.radius("y")), h.gdd_mm2, h.delays, period)


_FULL = _Pair(1.0)
_NONE = _Pair(0.0)


def _coherence(a: _Wave, b: _Wave) -> _Pair:
    """The pair's degree of coherence (module docstring): 1 where a coherence length is not declared, a path difference not
    defined (the two beams crossed different passes of undetermined group delay, ``delays_agree``) or a beam's Gaussian
    state not known; else the smallest over the radii ``sqrt(K) w`` (``COHERENCE_K``, ``w`` the first beam's larger 1/e^2
    radius) of the largest value inside -- from the chief rays' delay less the most it changes there (``_Pair.spread``) --
    plus the profiles' amplitude ``exp(-K)`` outside."""
    if a.length is None or b.length is None or a.opl is None or b.opl is None or not delays_agree(a.delays, b.delays):
        return _FULL
    length = min(a.length, b.length)
    # (a pulse train's pulses may be mutually coherent: the delay counts by its distance to the nearest multiple of the
    # period length too, optics/interference/coherence.py)
    tau = nearest_delay(a.opl - b.opl, tuple(p for p in (a.period, b.period) if p is not None))
    ratio = tau / length
    if ratio * ratio <= COHERENT_SHORTCUT_RATIO2 or not a.known or not b.known:
        return _FULL
    na, nb = a.n, b.n
    vx = na * a.direction[0] - nb * b.direction[0]
    vy = na * a.direction[1] - nb * b.direction[1]
    vz = na * a.direction[2] - nb * b.direction[2]
    if a.flat and b.flat:
        # on a plane the points lie along it: only the part of n_i d_i - n_j d_j along the plane changes the delay
        nx, ny, nz = a.normal
        along = (vx * nx + vy * ny + vz * nz) / (nx * nx + ny * ny + nz * nz)
        vx, vy, vz = vx - along * nx, vy - along * ny, vz - along * nz
    cx = a.position[0] - b.position[0]
    cy = a.position[1] - b.position[1]
    cz = a.position[2] - b.position[2]
    pair = _Pair(1.0, False, tau, length, math.sqrt(vx * vx + vy * vy + vz * vz),
                 nb * abs(cx * b.direction[0] + cy * b.direction[1] + cz * b.direction[2]), a.curvature, b.curvature,
                 math.sqrt(cx * cx + cy * cy + cz * cz), a.position, a.envelope, b.envelope, a.radius, b.radius,
                 abs(a.dispersion - b.dispersion))
    dispersed = pair.dispersed
    widest = a.width * COHERENCE_SQRT_K[-1]
    if not dispersed and (tau - pair.spread(widest)) / length >= INCOHERENT_RATIO:
        return _NONE  # (even the largest radius leaves every term below GAMMA_NEGLIGIBLE)
    if dispersed:
        # (the same with the envelopes' spread there and the widest dispersion: the spectrum's coherence is at most
        # exp(-ratio^2 / (1 + b^2)) for every dispersion up to b; a beam's light beyond its largest kept family adds at most
        # exp(-64), below the rounding too)
        spread, _, high, _ = pair.envelope(widest)
        ratio = (tau - pair.spread(widest) - spread) / length
        b = 2.0 * max(high, 0.0) / (length * length)
        if ratio > 0.0 and ratio * ratio >= INCOHERENT_RATIO * INCOHERENT_RATIO * (1.0 + b * b):
            return _NONE
    gaps = []
    lows, highs, beyonds = [], [], []
    for root_k in COHERENCE_SQRT_K:
        r = a.width * root_k
        gap = tau - pair.spread(r)
        if dispersed:
            spread, low, high, beyond = pair.envelope(r)
            gap -= spread
            lows.append(max(low, 0.0))
            highs.append(max(high, 0.0))
            beyonds.append(beyond)
        gaps.append(0.0 if gap <= 0.0 else gap / length)
    if dispersed:
        x = np.array(gaps)
        b_low = 2.0 * np.array(lows) / (length * length)
        b_high = 2.0 * np.array(highs) / (length * length)
        u = np.minimum(np.maximum(4.0 * x * x, 1.0 + b_low * b_low), 1.0 + b_high * b_high)
        # (``largest_coherence`` for each radius at once, with the light beyond the kept families)
        inside = np.minimum(exact_exp(-(x * x) / u) / np.sqrt(np.sqrt(u)) + np.array(beyonds), 1.0)
    else:
        inside = exact_exp(-(np.array(gaps) * np.array(gaps)))
    best = 1.0
    for value, tail in zip(inside, _COHERENCE_TAIL):
        bound = float(value) + tail
        if bound < best:
            best = bound
    return _Pair(0.0 if best < GAMMA_NEGLIGIBLE else best, False, pair.tau, pair.length, pair.slope, pair.offset,
                 pair.curvature_i, pair.curvature_j, pair.separation, pair.origin, pair.envelope_i, pair.envelope_j,
                 pair.radius_i, pair.radius_j, pair.dispersion)


@dataclass(frozen=True)
class _Faint:
    """Light of a class's lines left out as negligible (``SurfaceReach.left_out_W``): the power its pieces could carry
    together in phase (``(sum of the square roots of the pieces' powers)^2``: they are mutually coherent; the class's lines
    add their sums), the same of their pulse energies (pulsed lines; ``None`` without one), the larger index of the
    surface's media and the shortest of the lines' wavelengths, per line (key, wavelength, power, pulse train) with its
    sum of square roots (``roots``), the largest spectral factor of the lines (``spectral_factor``) and the factor by which
    the surfaces facing each other that return it to the surface multiply its field there (``validation/resonance.py``:
    ``ReturnGains.factor``; ``inf``: not bounded), and the most field its open paths' crossings bring to the surface, as a
    multiple of one crossing of its whole power (``ReturnGains.crossing_field`` of the surface; ``inf``: not known). Its
    field on the surface is at most the amplitude of the diffraction limit of that in-phase power times those two
    (``amplitude``)."""

    power_W: float
    energy_J: float | None
    n: float
    wavelength_nm: float
    lines: tuple[tuple[LineKey, float, float, PulseTrain | None], ...] = ()
    spectral: float = 1.0
    factor: float = 1.0
    roots: tuple[float, ...] = ()
    crossings: float = math.inf

    def amplitude(self, q: _Quantity) -> float:
        value = q.limit(self.power_W, self.energy_J, self.n, self.wavelength_nm, self.spectral)
        return 0.0 if value is None or value <= 0.0 else self.crossings * math.sqrt(value) * self.factor

    def root(self, q: _Quantity) -> float:
        """The most its field brings across a plane, as the square root of a power (``peak``; sqrt W) or a pulse energy
        (``fluence``; sqrt J): ``c f sqrt(P*)``, its in-phase power's crossings and returns (``window_bound``)."""
        amount = self.power_W if q.kind != "fluence" else self.energy_J
        return 0.0 if amount is None or amount <= 0.0 else self.crossings * math.sqrt(amount) * self.factor

    def part(self, keys: set[LineKey], q: _Quantity) -> float:
        """The amplitude of the lines ``keys`` among them alone."""
        root, energy_root = 0.0, None
        for (key, _, p, pulse), r in zip(self.lines, self.roots):
            if key in keys:
                root += r
                if pulse is not None:
                    energy_root = (energy_root or 0.0) + r / math.sqrt(pulse.repetition_rate_hz)
        energy = None if energy_root is None else energy_root * energy_root
        return (_Faint(root * root, energy, self.n, self.wavelength_nm, (), self.spectral, self.factor, (),
                       self.crossings).amplitude(q)
                if root > 0.0 else 0.0)


@dataclass(frozen=True)
class _Class:
    """Mutually coherent contributions on one surface (indices into the surface's contribution list, ascending), each
    pair's degree of coherence, whether light of unknown size or fate may join it (no lower bound above 0), and the light
    of its lines left out as negligible (``faint``). A class of left-out light alone has no members."""

    members: tuple[int, ...]
    gamma: dict[tuple[int, int], _Pair]
    open_below: bool
    faint: _Faint | None = None

    def pair(self, i: int, j: int) -> float:
        return self.gamma[(i, j) if i < j else (j, i)].gamma

    def pair_at(self, i: int, j: int, p) -> float:
        return self.gamma[(i, j) if i < j else (j, i)].at(p)

    def amplitude(self, q: _Quantity) -> float:
        return 0.0 if self.faint is None else self.faint.amplitude(q)

    @property
    def interferes(self) -> bool:
        return any(g.gamma > 0.0 for g in self.gamma.values())

    def interferes_with(self, members: set[int]) -> bool:
        """Whether one of ``members`` has a nonzero degree of coherence with another member of the class."""
        return any(g.gamma > 0.0 for (i, j), g in self.gamma.items() if i in members or j in members)


@dataclass(frozen=True)
class _FaintLine:
    """A source line with light left out as negligible: its key, coherence group, wavelength, total left-out power, pulse
    train (``None``: continuous), spectral factor (``spectral_factor``) and the sum of the square roots of its pieces'
    powers (``root``, sqrt(W); its square is the power they could carry together in phase)."""

    key: LineKey
    group: str
    wavelength_nm: float
    power_W: float
    pulse: PulseTrain | None
    spectral: float = 1.0
    root: float = 0.0
    # the other coherence groups its light may belong to: a partially polarized source's second polarization component
    # (``optics/polarization/partial.py``); the left-out light is counted whole with each, an upper bound
    also: tuple[str, ...] = ()


def _faint_of(lines: list[_FaintLine], n_of: Callable[[float], float],
              factor_of: Callable[[float], float] = lambda _: 1.0,  # type: ignore[assignment]
              crossings_of: Callable[[float], float] = lambda _: math.inf) -> _Faint | None:  # type: ignore[assignment]
    if not lines:
        return None
    root, energy_root, n, shortest, spectral, factor, crossings = 0.0, None, 0.0, math.inf, 1.0, 1.0, 0.0
    for line in lines:
        root += line.root
        if line.pulse is not None:
            energy_root = (energy_root or 0.0) + line.root / math.sqrt(line.pulse.repetition_rate_hz)
        n = max(n, n_of(line.wavelength_nm))
        shortest = min(shortest, line.wavelength_nm)
        spectral = max(spectral, line.spectral)
        factor = max(factor, factor_of(line.wavelength_nm))
        crossings = max(crossings, crossings_of(line.wavelength_nm))
    energy = None if energy_root is None else energy_root * energy_root
    return _Faint(root * root, energy, n, shortest, tuple((x.key, x.wavelength_nm, x.power_W, x.pulse) for x in lines), spectral,
                  factor, tuple(x.root for x in lines), crossings)


def _classes(contribs: list[_HitValues], cfg: NumericalConfig, unknown_lines: list[LineTag],
             faint_lines: list[_FaintLine] = (), n_of: Callable[[float], float] = lambda _: 1.0,  # type: ignore[assignment]
             factor_of: Callable[[float], float] = lambda _: 1.0,  # type: ignore[assignment]
             crossings_of: Callable[[float], float] = lambda _: math.inf,  # type: ignore[assignment]
             continuous: frozenset[str] = frozenset()) -> list[_Class]:
    """The coherence classes of a surface's contributions (module docstring), ordered by their first member: one per
    coherence group and line (vacuum wavelengths chained within ``interference_wavelength_tol_nm``), whichever side of the
    surface the light arrives from, then the classes of left-out light whose lines no contribution here belongs to.
    ``unknown_lines``: lines part of whose light has an unknown fate; ``faint_lines``: the lines with light left out as
    negligible (``n_of``: the larger index of the surface's media at a wavelength; ``factor_of``: the factor of the
    returns between surfaces facing each other at a wavelength, ``crossings_of``: the field its open paths' crossings
    bring at a wavelength, ``validation/resonance.py``)."""
    keyed: dict[str, list[int]] = {}
    for i, v in enumerate(contribs):
        assert v.raw is not None
        keyed.setdefault(v.raw.coherence_group, []).append(i)
    out: list[_Class] = []
    tol = cfg.interference_wavelength_tol_nm
    used: set[tuple[LineKey, str]] = set()
    for group, indices in keyed.items():
        lines: list[list[int]] = []
        for i in sorted(indices, key=lambda k: (contribs[k].hit.wavelength_nm, k)):
            if lines and abs(contribs[lines[-1][0]].hit.wavelength_nm - contribs[i].hit.wavelength_nm) <= tol:
                lines[-1].append(i)
            else:
                lines.append([i])
        for line in lines:
            members = tuple(sorted(line))
            waves = {i: _wave(contribs[i], continuous) for i in members}
            gamma = {(i, j): _coherence(waves[i], waves[j]) for a, i in enumerate(members) for j in members[a + 1:]}
            wavelengths = sorted({contribs[i].hit.wavelength_nm for i in members})

            def near(w: float, wavelengths: list[float] = wavelengths) -> bool:
                return any(abs(w - x) <= tol for x in wavelengths)

            open_below = any(g == group and near(w) for g, w in unknown_lines)
            mine = [x for x in faint_lines if group in (x.group, *x.also) and near(x.wavelength_nm) and (x.key, group) not in used]
            used.update((x.key, group) for x in mine)
            out.append(_Class(members, gamma, open_below, _faint_of(mine, n_of, factor_of, crossings_of)))
    out.sort(key=lambda c: c.members[0])
    # left-out light of lines no contribution here belongs to: classes of its own (one per group and chained lines)
    rest = sorted((x for x in faint_lines if not any((x.key, g) in used for g in (x.group, *x.also))),
                  key=lambda x: (x.group, x.wavelength_nm, x.key))
    chain: list[_FaintLine] = []
    for x in rest:
        if chain and (x.group != chain[-1].group or abs(x.wavelength_nm - chain[-1].wavelength_nm) > tol):
            out.append(_Class((), {}, False, _faint_of(chain, n_of, factor_of, crossings_of)))
            chain = []
        chain.append(x)
    if chain:
        out.append(_Class((), {}, False, _faint_of(chain, n_of, factor_of, crossings_of)))
    return out


def _profile(v: _HitValues, p) -> float:
    """The beam's Gaussian profile at the point ``p``: ``exp(-2 rho^2)``, ``rho`` the distance from its chief ray in units of
    its 1/e^2 radii in its transverse frame (1 at its centre)."""
    h = v.raw
    assert h is not None and v.hit.beam_radius_x_mm is not None and v.hit.beam_radius_y_mm is not None
    r = (float(p[0]) - float(h.position[0]), float(p[1]) - float(h.position[1]), float(p[2]) - float(h.position[2]))
    x = _dot3(r, h.gstate.bx) / v.hit.beam_radius_x_mm
    y = _dot3(r, h.gstate.by) / v.hit.beam_radius_y_mm
    rho2 = x * x + y * y
    return 1.0 if rho2 == 0.0 else float(exact_exp(-2.0 * rho2))


def _class_upper(cls: _Class, values: list[float], amp: float = 0.0) -> float:
    """``sum v + 2 sum gamma sqrt(v_i v_j)`` of a class's members' values (the members' order), with the left-out light's
    amplitude ``amp``: ``+ 2 amp sum sqrt(v_i) + amp^2``."""
    if len(values) == 1 and amp == 0.0:
        return values[0]
    roots = [math.sqrt(v) for v in values]
    total = 0.0
    for v in values:
        total += v
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            total += 2.0 * cls.pair(cls.members[a], cls.members[b]) * roots[a] * roots[b]
    if amp > 0.0:
        summed = 0.0
        for root in roots:
            summed += root
        total += 2.0 * amp * summed + amp * amp
    return total


def _upper(contribs: list[_HitValues], classes: list[_Class], q: Quantity, *, known_only: bool = False) -> float | None:
    """The combined upper bound of the contributions' peak values and the left-out light (``None`` when one is not known;
    ``known_only``: over the known ones, ``None`` when none is)."""
    total, any_known = 0.0, False
    for cls in classes:
        values = [q(contribs[i]) for i in cls.members]
        amp = cls.amplitude(q)
        if any(v is None for v in values):
            if not known_only:
                return None
            kept = tuple(i for i, v in zip(cls.members, values) if v is not None)
            if not kept and amp == 0.0:
                continue
            cls = _Class(kept, cls.gamma, cls.open_below, cls.faint)
            values = [v for v in values if v is not None]
        if not values and amp == 0.0:
            continue
        total += _class_upper(cls, values, amp)  # type: ignore[arg-type]
        any_known = True
    return total if any_known or not known_only else None


def _class_range(cls: _Class, contribs: list[_HitValues], q: Quantity, p, *, events: bool = False,
                 members: dict[int, float] | None = None) -> tuple[float, float] | None:
    """(lower, upper) of a class's combined value at the point ``p``: each member's value there through its Gaussian
    profile (a beam beside the surface, or untraced light of unknown size, at its bound anywhere), each pair's degree of
    coherence at ``p``, the left-out light at its amplitude; ``None`` when a member's value is not known. The lower bound is
    the traced members' value less every cross term with a traced member at its worst, or the strongest traced member less
    all the others' amplitudes (module docstring): the other light's own intensity, its cross terms among itself included,
    is >= 0 and left out. 0 for a class without a traced member or one light of unknown size or fate may join. ``events``
    (a peak fluence): the members may arrive in separate pulses (passes whose delays differ by more than a pulse), and the
    sum of the members' values holds only for light that arrives in one pulse; the pulse that holds a traced member ``k``
    is at least ``v_k - 2 a_k (sum_{j != k} gamma_kj a_j + A)`` (its cross terms with whatever arrives with it at their
    worst, the rest's own intensity >= 0) and at least ``(a_k - sum_{j != k} a_j - A)^2`` -- whichever members arrive
    with it --, the largest over the traced members (``members``, when given: each traced member's bound, 0 where none
    holds)."""
    traced: list[tuple[int, float]] = []  # (index, value at p)
    amplitude: dict[int, float] = {}  # every member's amplitude (sqrt value) at p
    amp = cls.amplitude(q)
    upper = 0.0
    for i in cls.members:
        v = contribs[i]
        value = q(v)
        if value is None:
            return None
        here = value * _profile(v, p) if v.located else value
        amplitude[i] = math.sqrt(here)
        upper += here
        if v.light == TRACED:
            traced.append((i, here))
    summed = 0.0
    for a in range(len(cls.members)):
        i = cls.members[a]
        summed += amplitude[i]
        for b in range(a + 1, len(cls.members)):
            j = cls.members[b]
            upper += 2.0 * cls.pair_at(i, j, p) * amplitude[i] * amplitude[j]
    if amp > 0.0:
        upper += 2.0 * amp * summed + amp * amp
    if cls.open_below or not traced:
        return 0.0, upper
    if len(cls.members) == 1 and amp == 0.0:
        if members is not None:
            members[traced[0][0]] = traced[0][1]
        return traced[0][1], upper
    is_traced = {i for i, _ in traced}
    pairwise = 0.0
    for _, here in traced:
        pairwise += here
    for a in range(len(cls.members)):
        for b in range(a + 1, len(cls.members)):
            i, j = cls.members[a], cls.members[b]
            if i in is_traced or j in is_traced:
                pairwise -= 2.0 * cls.pair_at(i, j, p) * amplitude[i] * amplitude[j]
    if amp > 0.0:
        for i, _ in traced:
            pairwise -= 2.0 * amp * amplitude[i]
    k = max(traced, key=lambda item: item[1])[0]
    rest = amplitude[k]
    scale = 0.0
    for i in cls.members:
        scale += amplitude[i] * amplitude[i]
        if i != k:
            rest -= amplitude[i]
    rest -= amp
    least = rest * rest if rest > 0.0 else 0.0
    if not events:
        least = max(pairwise, least)
    else:
        for i, here in traced:
            cross = amp
            alone = amplitude[i] - amp
            for j in cls.members:
                if j != i:
                    cross += cls.pair_at(i, j, p) * amplitude[j]
                    alone -= amplitude[j]
            own = max(here - 2.0 * amplitude[i] * cross, alone * alone if alone > 0.0 else 0.0)
            least = max(least, own)
            if members is not None:
                members[i] = own if own > CANCELLATION_ROUNDING * scale else 0.0
    # (a cancellation left at the rounding level of the members' values is 0: the smaller bound is the safe one)
    return (least if least > CANCELLATION_ROUNDING * scale else 0.0), upper


def _centres(contribs: list[_HitValues], q: Quantity) -> list[int]:
    """The traced beams with a known value and Gaussian profile on the surface (where lower bounds are taken)."""
    return [i for i, v in enumerate(contribs) if v.light == TRACED and v.located and q(v) is not None]


def _crossing(v: _HitValues) -> tuple[str, tuple[str, ...], int]:
    """A crossing of a source's light, whatever its line: the source, the interactions from it up to the surface and the
    pass. The lines of one pulse that cross the surface along one path share it."""
    assert v.raw is not None
    return v.hit.source_id, v.raw.lineage, v.hit.pass_index


def _lower(contribs: list[_HitValues], classes: list[_Class], q: Quantity) -> float:
    """The combined lower bound: the largest, over the traced beams' centres, of the classes' lower bounds there, and of
    the window bound of each class with light of unknown place (``_window``; module docstring); 0 when no traced beam has
    a known value. Values of the time-averaged light (irradiance, ``P / d``) add over the classes; a peak fluence is the
    energy of one pulse per area: a pulse holds the lines of its source along one path (``_crossing``), each line's light in
    it at least its class's bound for that crossing's member (``_class_range`` ``events``; within a class the largest of
    its members on that crossing), and pulses of different crossings need not arrive together, so it is bounded by the
    largest crossing's sum."""
    pulses = q.kind == "fluence"
    best = 0.0
    for c in _centres(contribs, q):
        p = contribs[c].raw.position  # type: ignore[union-attr]
        total = 0.0
        pulse: dict[tuple[str, tuple[str, ...], int], float] = {}
        for cls in classes:
            found: dict[int, float] = {}
            bounds = _class_range(cls, contribs, q, p, events=pulses, members=found if pulses else None)
            if bounds is None:
                continue
            if not pulses:
                total += bounds[0]
                continue
            mine: dict[tuple[str, tuple[str, ...], int], float] = {}
            for i, value in found.items():
                key = _crossing(contribs[i])
                mine[key] = max(mine.get(key, 0.0), value)
            for key, value in mine.items():
                pulse[key] = pulse.get(key, 0.0) + value
        if pulses:
            for value in pulse.values():
                total = max(total, value)
        best = max(best, total)
    return max(best, _window(contribs, classes, q))


def _window_beam(v: _HitValues, q: Quantity) -> WindowBeam:
    h = v.raw
    assert h is not None and v.hit.beam_radius_x_mm is not None and v.hit.beam_radius_y_mm is not None
    value = q(v)
    assert value is not None
    return WindowBeam(value, (float(h.position[0]), float(h.position[1]), float(h.position[2])),
                      (float(h.gstate.bx[0]), float(h.gstate.bx[1]), float(h.gstate.bx[2])),
                      (float(h.gstate.by[0]), float(h.gstate.by[1]), float(h.gstate.by[2])),
                      v.hit.beam_radius_x_mm, v.hit.beam_radius_y_mm)


def _inverse_radius(q: complex) -> float:
    """``|Re 1/q|``: the wavefront's curvature of a Gaussian axis (1/mm)."""
    return abs(q.real) / (q.real * q.real + q.imag * q.imag)


def _window(contribs: list[_HitValues], classes: list[_Class], q: Quantity) -> float:
    """The largest window bound (``validation/window_bound.py``) of a class that light of unknown place -- untraced light
    bounded by its diffraction limit, light left out as negligible -- may reach, over its traced beams on a plane or a
    sphere: in peak irradiance and peak fluence for a beam of one Gaussian mode (``window_lower``), in ``P / d`` for any
    (``linear_lower``: that light as Gaussian beams crossing the surface from one side, by the strip each covers); 0
    otherwise."""
    if q.kind not in ("peak", "fluence", "linear"):
        return 0.0
    linear = q.kind == "linear"
    kappa = MM_PER_CM / (FWHM_PER_RADIUS if q.definition == "FWHM" else 2.0)
    best = 0.0
    for cls in classes:
        if cls.open_below or not cls.members:
            continue
        unknown = beside = 0.0
        waist = math.inf  # (P / d: the smallest diffraction-limited waist of the light of unknown place)
        located: list[int] = []
        known = True
        for i in cls.members:
            v = contribs[i]
            value = q(v)
            if value is None:
                known = False
                break
            if v.limited:
                amount = v.hit.pulse_energy_J if q.kind == "fluence" else v.hit.incident_power_W
                if amount is None:
                    known = False
                    break
                unknown += math.sqrt(amount)
                length = v.raw.coherence_length_mm if v.raw is not None else None
                waist = min(waist, limit_waist(v.hit.wavelength_nm, v.n_max, spectral_factor(v.hit.wavelength_nm, length, v.pulse)))
            elif v.light == BESIDE:
                beside += math.sqrt(value)
            elif v.located:
                located.append(i)
            else:
                known = False
                break
        if not known:
            continue
        if cls.faint is not None and cls.faint.root(q) > 0.0:
            unknown += cls.faint.root(q)
            waist = min(waist, limit_waist(cls.faint.wavelength_nm, cls.faint.n, cls.faint.spectral))
        if not unknown > 0.0:
            continue  # (no light of unknown place: the bound at the beams' centres is at least as large)
        for k in located:
            v = contribs[k]
            h = v.raw
            assert h is not None
            if v.light != TRACED or v.room is None:
                continue
            radius, room, spread = v.room
            normal = (float(h.normal[0]), float(h.normal[1]), float(h.normal[2]))
            others = [_window_beam(contribs[j], q) for j in located if j != k]
            if linear:
                best = max(best, linear_lower(_window_beam(v, q), normal, others, beside, unknown, kappa, waist, radius, room))
                continue
            if h.gstate.m2x != 1.0 or h.gstate.m2y != 1.0:
                continue
            anchor = Anchor(_window_beam(v, q), normal, h.cos_aoi, h.gstate.n, v.hit.wavelength_nm, _inverse_radius(h.gstate.qx),
                            _inverse_radius(h.gstate.qy))
            best = max(best, window_lower(anchor, others, beside, unknown, v.n_max, radius, room, spread))
    return best


def _bounds(contribs: list[_HitValues], classes: list[_Class], q: Quantity, open_above: bool) -> tuple[float, float | None]:
    """(lower, upper or ``None``) of a surface's exposure in one quantity."""
    upper = None if open_above else _upper(contribs, classes, q)
    lower = _lower(contribs, classes, q)
    if upper is not None and lower > upper:
        lower = upper  # (rounding)
    return lower, upper


def _without_faint(classes: list[_Class]) -> list[_Class]:
    """The classes without their left-out light (classes of left-out light alone dropped)."""
    return [replace(cls, faint=None) for cls in classes if cls.members]


def _amplitudes(contribs: list[_HitValues], classes: list[_Class], q: Quantity) -> list[tuple[float, float]] | None:
    """Per class (the summed amplitudes of its members' peak values, its left-out amplitude); ``None`` when a value is not
    known."""
    out: list[tuple[float, float]] = []
    for cls in classes:
        summed = 0.0
        for i in cls.members:
            value = q(contribs[i])
            if value is None:
                return None
            summed += math.sqrt(value)
        out.append((summed, cls.amplitude(q)))
    return out


def _faint_scale(contribs: list[_HitValues], classes: list[_Class], q: Quantity, within: float | None, lower: float,
                 upper: float | None, open_above: bool, exceeds: float | None = None) -> float | None:
    """When the left-out light alone keeps the comparison from being decided: the factor its amplitudes must shrink by for
    the decision (with half the margin to spare; ``validation/pipeline.py`` follows its lines deeper). ``None`` otherwise.
    ``within``: the exposure up to which the comparison is within the rating, ``exceeds`` the one above which it exceeds it
    (``rated_threshold``; ``exceeds`` omitted: ``within``; ``None``: not decided on that side)."""
    exceeds = within if exceeds is None else exceeds
    if open_above or not any(cls.faint is not None and cls.amplitude(q) > 0.0 for cls in classes):
        return None
    if (exceeds is not None and lower > exceeds) or (within is not None and upper is not None and upper <= within):
        return None  # decided
    bare = _without_faint(classes)
    lower0, upper0 = _bounds(contribs, bare, q, False)
    parts = _amplitudes(contribs, classes, q)
    if parts is None:
        return None
    total = 0.0
    for _, amp in parts:
        total += amp
    if total <= 0.0:
        return None
    if within is not None and upper0 is not None and upper0 <= within:  # within without it: sum_c (2 k A_c S_c + k^2 A_c^2) <= half the margin
        a = b = 0.0
        for summed, amp in parts:
            a += amp * amp
            b += 2.0 * amp * summed
        c = -0.5 * (within - upper0)
        return (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    if exceeds is not None and lower0 > exceeds:  # exceeded without it: (sqrt(lower) - k sum A_c)^2 above it with half the margin
        return 0.5 * (math.sqrt(lower0) - math.sqrt(exceeds)) / total
    return None


def _incident_power(values: list[_HitValues], classes: list[_Class]) -> float:
    """The traced beams' combined incident power, a flux through the surface: per class ``sum P + 2 sum gamma sqrt(P_i P_j)
    (cos a_i + cos a_j) / (2 sqrt(cos a_i cos a_j))`` over the pairs arriving from one side (beams crossing the surface in
    opposite directions have no cross term in the flux; module docstring)."""
    total = 0.0
    for cls in classes:
        powers = [values[i].hit.incident_power_W for i in cls.members]
        if not powers:
            continue
        if len(powers) == 1:
            total += powers[0]
            continue
        part = 0.0
        for p in powers:
            part += p
        for a in range(len(powers)):
            for b in range(a + 1, len(powers)):
                i, j = cls.members[a], cls.members[b]
                if _side(values[i].raw) != _side(values[j].raw):  # type: ignore[arg-type]
                    continue
                ci = max(values[i].raw.cos_aoi, GRAZING_COS)  # type: ignore[union-attr]
                cj = max(values[j].raw.cos_aoi, GRAZING_COS)  # type: ignore[union-attr]
                flux = (ci + cj) / (2.0 * math.sqrt(ci * cj))
                part += 2.0 * cls.pair(i, j) * math.sqrt(powers[a]) * math.sqrt(powers[b]) * flux
        total += part
    return total


# ------------------------------------------------------------------------------------------- rating conditions


def _off_share(contribs: list[_HitValues], classes: list[_Class], off: set[LineKey], q: Quantity,
               followed: frozenset[int] = frozenset(), tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[float, float | None] | None:
    """The share of the most light the surface can receive, in the quantity ``q``, that the left-out light of the lines
    ``off`` and the followed contributions ``followed`` (of lines the rating is not for, ``_minor_lines``) may carry, and
    the factor the left-out amplitudes must shrink by for the tolerance (with half the margin to spare; ``None``: that
    cannot help). A contribution without a value in ``q`` is left out of the most light (it would only add to it: the
    larger share), a followed one never (``None``); ``None`` too when that light has none in ``q``."""
    most = mismatched = known = 0.0
    a = b = fixed = 0.0  # (sum p_c^2, 2 sum m_c p_c, sum m_c^2: the followed and left-out amplitudes per class)
    for cls in classes:
        summed = mine = 0.0
        for i in cls.members:
            value = q(contribs[i])
            if value is None:
                if i in followed:
                    return None
                continue
            summed += math.sqrt(value)
            if i in followed:
                mine += math.sqrt(value)
        amp = cls.amplitude(q)
        most += (summed + amp) * (summed + amp)
        known += summed * summed
        part = 0.0
        if cls.faint is not None:
            keys = {key for key, _, _, _ in cls.faint.lines if key in off}
            if keys:
                part = cls.faint.part(keys, q)
        mismatched += (mine + part) * (mine + part)
        a += part * part
        b += 2.0 * mine * part
        fixed += mine * mine
    if not mismatched > 0.0 or not most > 0.0:
        return None
    if not followed:
        scale = 0.5 * math.sqrt(tol.unlocated_light_share * known / mismatched) if known > 0.0 else None
        return mismatched / most, scale
    # (the factor k on the left-out amplitudes with sum_c (m_c + k p_c)^2 at the tolerance of the known light, halved)
    c = fixed - tol.unlocated_light_share * known
    scale = 0.5 * (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a) if a > 0.0 and c < 0.0 else None
    return mismatched / most, scale


def _faint_mismatch(rating: PowerRating, contribs: list[_HitValues], classes: list[_Class],
                    surface_id: str, followed: frozenset[int] = frozenset(), tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[dict[str, str], float | None]:
    """Left-out light of lines whose regime, wavelength, pulse duration or repetition rate ``rating`` is not for (module
    docstring), with the followed light of such lines that the share rule leaves out of the conditions (``followed``,
    ``_minor_lines``): when its share of the most light the surface can receive -- in peak irradiance or in the rating's
    own quantity (``_share_quantities``; with followed light, in every quantity of ``_line_share_quantities``), whichever
    is larger -- exceeds the tolerance, a text per condition such light differs in (each leaves that condition not
    verifiable), and the factor its amplitudes must shrink by (``None``: that cannot help)."""
    off: set[LineKey] = set()
    names: dict[str, list[str]] = {}
    for cls in classes:
        if cls.faint is None:
            continue
        for key, wavelength, _, pulse in cls.faint.lines:
            if _name_differences(rating, key[0], wavelength, pulse, names, tol=tol):
                off.add(key)
    if not off:
        return {}, None
    quantities = _line_share_quantities(rating) if followed else (_peak, *_share_quantities(rating))
    found = [x for x in (_off_share(contribs, classes, off, q, followed, tol=tol) for q in quantities) if x is not None]
    over = [x for x in found if x[0] > tol.unlocated_light_share]
    if not over:
        return {}, None
    for i in sorted(followed):
        v = contribs[i]
        _name_differences(rating, v.hit.source_id, v.hit.wavelength_nm, v.pulse, names, tol=tol)
    share = max(x[0] for x in over)
    scale = None if any(x[1] is None for x in over) else min(x[1] for x in over if x[1] is not None)
    wanted = _wanted(rating)
    texts = {kind: f"{kind.replace('_', ' ')}: light of {', '.join(sorted(set(names[kind])))} that the trace leaves out as "
                   f"negligible may reach {surface_id} (light the rating is not for carries up to {PERCENT * share:.3g} % of "
                   f"the most light there; {wanted[kind]})" for kind in sorted(names)}
    return texts, scale


def _name_differences(rating: PowerRating, source_id: str, wavelength_nm: float, pulse: PulseTrain | None,
                      names: dict[str, list[str]], tol: RatingTolerances = DEFAULT_TOLERANCES) -> bool:
    """Add a line's name to ``names`` under each condition of ``rating`` it differs in (``line_differences``); whether it
    differs in one."""
    kinds = line_differences(rating, wavelength_nm, pulse, tol)
    for kind in kinds:
        detail = ("" if pulse is None or kind not in ("pulse_duration", "repetition_rate") else
                  f" ({pulse.pulse_duration_s:.4g} s pulses)" if kind == "pulse_duration" else
                  f" ({pulse.repetition_rate_hz:.4g} Hz)")
        names.setdefault(kind, []).append(f"{source_id} at {wavelength_nm:g} nm{detail}")
    return bool(kinds)


def _wanted(rating: PowerRating) -> dict[str, str]:
    """What ``rating`` is for, per condition a line may differ in (messages)."""
    return {"regime": f"a {rating.regime} rating", "wavelength": f"a rating at {rating.wavelength_nm:g} nm",
            "pulse_duration": f"a rating for {rating.pulse_duration_s or 0.0:.4g} s pulses",
            "repetition_rate": f"a rating at {rating.repetition_rate_hz or 0.0:.4g} Hz"}


def _unbounded_mismatch(rating: PowerRating, lines: tuple[Light, ...], surface_id: str, tol: RatingTolerances = DEFAULT_TOLERANCES) -> dict[str, str]:
    """On a surface only light left out as negligible reaches, where nothing bounds that light (light that may be stored
    between surfaces facing each other): a text per condition of ``rating`` one of ``lines`` -- the left-out light's lines
    and the lines whose light may be stored -- differs in. Each leaves that condition not verifiable, whatever the line's
    power (its share of the light there has no bound)."""
    names: dict[str, list[str]] = {}
    for x in lines:
        _name_differences(rating, x.source_id, x.wavelength_nm, x.pulse, names, tol=tol)
    wanted = _wanted(rating)
    return {kind: f"{kind.replace('_', ' ')}: light of {', '.join(sorted(set(names[kind])))} that the trace leaves out as "
                  f"negligible, or whose fate it does not know, may reach {surface_id}, and nothing bounds its share of the "
                  f"light there ({wanted[kind]})" for kind in sorted(names)}


@dataclass(frozen=True)
class _State:
    """A contribution's polarization at the surface: the field's squared norm, the share of its power in s (``None`` at
    normal incidence: no plane of incidence), its normalized circular Stokes parameter ``s3`` (|s3| = 1 circular, 0
    linear; > 0 one handedness about its direction of travel), the major axis of its polarization ellipse (a unit vector,
    ``None`` for a circular state) and ``field``: the complex field vector (real parts, imaginary parts) in global
    coordinates."""

    norm2: float
    s_share: float | None
    s3: float
    axis: tuple[float, float, float] | None
    field: tuple[tuple[float, float, float], tuple[float, float, float]]


def _state(h: SurfaceHit) -> _State | None:
    """The polarization of a contribution at its hit, from its Jones vector in its transverse basis (``None`` without
    field). The major axis from the Stokes parameters s1, s2 by half-angle square roots (no trigonometry)."""
    jx, jy = complex(h.jones[0]), complex(h.jones[1])
    xr, xi, yr, yi = jx.real, jx.imag, jy.real, jy.imag
    norm2 = xr * xr + xi * xi + yr * yr + yi * yi
    if not norm2 > 0.0:
        return None
    e1, e2 = h.e1, h.e2
    real = tuple(xr * float(e1[k]) + yr * float(e2[k]) for k in range(3))
    imag = tuple(xi * float(e1[k]) + yi * float(e2[k]) for k in range(3))
    s3 = 2.0 * (xr * yi - xi * yr) / norm2
    s1 = (xr * xr + xi * xi) - (yr * yr + yi * yi)
    s2 = 2.0 * (xr * yr + xi * yi)
    length = math.sqrt(s1 * s1 + s2 * s2)
    axis = None
    if length > 0.0:
        c2 = s1 / length  # cos(2 psi), psi the major axis's angle from e1
        c = math.sqrt(max(0.0, 0.5 * (1.0 + c2)))
        sn = math.sqrt(max(0.0, 0.5 * (1.0 - c2)))
        if s2 < 0.0:
            sn = -sn
        axis = (c * float(e1[0]) + sn * float(e2[0]), c * float(e1[1]) + sn * float(e2[1]), c * float(e1[2]) + sn * float(e2[2]))
    d, n = h.direction, h.normal
    sx = float(d[1]) * float(n[2]) - float(d[2]) * float(n[1])
    sy = float(d[2]) * float(n[0]) - float(d[0]) * float(n[2])
    sz = float(d[0]) * float(n[1]) - float(d[1]) * float(n[0])
    s_len = math.sqrt(sx * sx + sy * sy + sz * sz)
    s_share = None
    if s_len > NORMAL_INCIDENCE_SIN:
        sx, sy, sz = sx / s_len, sy / s_len, sz / s_len
        er = sx * real[0] + sy * real[1] + sz * real[2]
        ei = sx * imag[0] + sy * imag[1] + sz * imag[2]
        s_share = min((er * er + ei * ei) / norm2, 1.0)
    return _State(norm2, s_share, s3, axis, (real, imag))  # type: ignore[arg-type]


def _orthogonal_share(condition: str, state: _State) -> float | None:
    """The share of a beam's own power in the state orthogonal to ``condition`` (``None``: s / p not defined)."""
    if condition in ("S", "P"):
        if state.s_share is None:
            return None
        return 1.0 - state.s_share if condition == "S" else state.s_share
    c = min(abs(state.s3), 1.0)
    if condition == "LINEAR":  # the minor axis of the polarization ellipse: sin^2 chi = (1 - sqrt(1 - s3^2)) / 2
        return 0.5 * (1.0 - math.sqrt((1.0 - c) * (1.0 + c)))
    return 0.5 * (1.0 - c)  # CIRCULAR: the opposite circular state


def _outside_axis(state: _State, axis: tuple[float, float, float]) -> float:
    """The share of a beam's power outside the linear axis ``axis`` (a unit vector): 1 - |axis . E|^2 / |E|^2."""
    (ar, ai) = state.field
    re = axis[0] * ar[0] + axis[1] * ar[1] + axis[2] * ar[2]
    im = axis[0] * ai[0] + axis[1] * ai[1] + axis[2] * ai[2]
    return max(0.0, 1.0 - (re * re + im * im) / state.norm2)


def _overlap2(a: _State, b: _State) -> float:
    """``|a^* . b|^2 / (|a|^2 |b|^2)`` of two fields: the share of one's power in the other's polarization state (1: one
    state, up to phase and amplitude)."""
    (ar, ai), (br, bi) = a.field, b.field
    re = ar[0] * br[0] + ar[1] * br[1] + ar[2] * br[2] + ai[0] * bi[0] + ai[1] * bi[1] + ai[2] * bi[2]
    im = ar[0] * bi[0] + ar[1] * bi[1] + ar[2] * bi[2] - ai[0] * br[0] - ai[1] * br[1] - ai[2] * br[2]
    return min(1.0, (re * re + im * im) / (a.norm2 * b.norm2))


def _along(state: _State, u: tuple[float, float, float]) -> float:
    """``|u . E|^2 / |E|^2``: the share of a beam's power along the real unit vector ``u``."""
    (re, im) = state.field
    x = u[0] * re[0] + u[1] * re[1] + u[2] * re[2]
    y = u[0] * im[0] + u[1] * im[1] + u[2] * im[2]
    return min(1.0, (x * x + y * y) / state.norm2)


def _interfering(cls: _Class, states: list[_State | None]) -> list[tuple[tuple[int, ...], bool]]:
    """The groups of a class's beams of known state that interfere (linked by a nonzero degree of coherence; a group's
    beams are mutually incoherent with the class's other beams), two or more each, in the order of their first beam, and
    whether a group's beams are in one polarization state (``_overlap2`` within ``PARALLEL_ROUNDING`` of 1: their sum is
    that state whatever their phases)."""
    members = [i for i in cls.members if states[i] is not None]
    parent = {i: i for i in members}

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for a, i in enumerate(members):
        for j in members[a + 1:]:
            if cls.pair(i, j) > 0.0:
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in members:
        groups.setdefault(root(i), []).append(i)
    out: list[tuple[tuple[int, ...], bool]] = []
    for group in groups.values():
        if len(group) > 1:
            first = states[group[0]]
            one = all(_overlap2(first, states[i]) >= 1.0 - PARALLEL_ROUNDING for i in group[1:])  # type: ignore[arg-type]
            out.append((tuple(group), one))
    return out


def _surface_form(v: _HitValues, n: tuple[float, float, float]) -> tuple[float, float, float]:
    """A traced beam's Gaussian profile on the surface (``_profile`` at ``position + s u + t v``) as the quadratic form
    ``(q_uu, q_uv, q_vv)`` of ``rho^2 = q_uu s^2 + 2 q_uv s t + q_vv t^2`` in a tangent basis ``(u, v)`` of the unit normal
    ``n``: on a plane, two beams with the same centre and form have the same amplitude profile on the surface, whatever
    their directions (a beam and its reflection do)."""
    h = v.raw
    assert h is not None and v.hit.beam_radius_x_mm is not None and v.hit.beam_radius_y_mm is not None
    k = min(range(3), key=lambda i: (abs(n[i]), i))
    e = (1.0 if k == 0 else 0.0, 1.0 if k == 1 else 0.0, 1.0 if k == 2 else 0.0)
    c = n[k]
    ux, uy, uz = e[0] - c * n[0], e[1] - c * n[1], e[2] - c * n[2]
    length = math.sqrt(ux * ux + uy * uy + uz * uz)
    u = (ux / length, uy / length, uz / length)
    w = (n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0])
    bx, by = h.gstate.bx, h.gstate.by
    wx, wy = v.hit.beam_radius_x_mm, v.hit.beam_radius_y_mm
    a1, a2 = _dot3(u, bx) / wx, _dot3(w, bx) / wx
    b1, b2 = _dot3(u, by) / wy, _dot3(w, by) / wy
    return a1 * a1 + b1 * b1, a1 * a2 + b1 * b2, a2 * a2 + b2 * b2


def _space_form(v: _HitValues) -> tuple[float, ...]:
    """A traced beam's Gaussian profile in space (``_profile`` at ``position + r``) as the entries ``(Q_xx, Q_xy, Q_xz,
    Q_yy, Q_yz, Q_zz)`` of the symmetric ``Q`` of ``rho^2 = r^T Q r``: its radii, its axes and, as the direction ``Q``
    leaves out, its direction. Two beams with the same centre and ``Q`` have the same profile at every point."""
    h = v.raw
    assert h is not None and v.hit.beam_radius_x_mm is not None and v.hit.beam_radius_y_mm is not None
    bx, by = h.gstate.bx, h.gstate.by
    wx, wy = v.hit.beam_radius_x_mm, v.hit.beam_radius_y_mm
    a = (float(bx[0]) / wx, float(bx[1]) / wx, float(bx[2]) / wx)
    b = (float(by[0]) / wy, float(by[1]) / wy, float(by[2]) / wy)
    return tuple(a[i] * a[j] + b[i] * b[j] for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)))


def _one_profile(group: tuple[int, ...], contribs: list[_HitValues]) -> bool:
    """Whether the mutually coherent beams ``group`` have one amplitude profile on the surface: traced beams of known
    Gaussian profile with the same centre and, on a plane, the same profile in the surface's own coordinates
    (``_surface_form``: the footprint, which follows a beam's direction as well as its radii and axes), to rounding. A
    curved surface leaves its tangent plane, so there they need the same profile in space (``_space_form``: the same
    direction, radii and axes); a beam and its reflection on a curved mirror are not one profile. Only then do their
    amplitudes keep one ratio at every point, so a bound on the share of their sum outside a state at the centre holds
    everywhere; beams displaced from each other, of different sizes or meeting the surface from different directions
    change that ratio across it, and where one of them dominates the other's state does."""
    first = contribs[group[0]]
    for i in group:
        v = contribs[i]
        if v.light != TRACED or not v.located or v.raw is None:
            return False
    h0, w0x, w0y = first.raw, first.hit.beam_radius_x_mm, first.hit.beam_radius_y_mm
    assert h0 is not None and w0x is not None and w0y is not None
    nx, ny, nz = float(h0.normal[0]), float(h0.normal[1]), float(h0.normal[2])
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    n = (nx / nl, ny / nl, nz / nl)
    flat = all(bool(contribs[i].raw.surface_flat) for i in group)  # type: ignore[union-attr]
    q0 = _surface_form(first, n) if flat else _space_form(first)
    q_scale = max(abs(x) for x in q0)
    round_off = PARALLEL_ROUNDING * (w0x if w0x < w0y else w0y)
    for i in group[1:]:
        v = contribs[i]
        h = v.raw
        assert h is not None
        dx = float(h.position[0]) - float(h0.position[0])
        dy = float(h.position[1]) - float(h0.position[1])
        dz = float(h.position[2]) - float(h0.position[2])
        if math.sqrt(dx * dx + dy * dy + dz * dz) > round_off:
            return False
        q = _surface_form(v, n) if flat else _space_form(v)
        if any(abs(q[j] - q0[j]) > PARALLEL_ROUNDING * q_scale for j in range(len(q0))):
            return False
    return True


def _bound_in(group: tuple[int, ...], f: dict[int, float], amp: dict[int, float], states: list[_State | None],
              wanted_refs: list[int], state_refs: list[int]) -> float:
    """``_coherent_bound`` in one basis of the rating's state (``f``: each beam's own share outside it; ``amp``: each
    beam's field amplitude). Whatever the phases, the part of the sum outside the state is at most ``U = sum sqrt(f_i)
    a_i`` and its part in it at least ``W = sqrt(1 - f_j) a_j - sum_{i != j} sqrt(1 - f_i) a_i`` (``j`` among
    ``wanted_refs``): at most ``U^2 / (U^2 + W^2)`` of the sum is outside it. In the state of a beam ``j`` (among
    ``state_refs``) the sum has a part of at least ``B = a_j - sum_{i != j} a_i |e_j^* . e_i|`` and the rest at most ``D =
    sum_{i != j} a_i sqrt(1 - |e_j^* . e_i|^2)``, so at most ``(B sqrt(f_j) + D)^2 / (B^2 + D^2)`` of it is outside the
    rating's state (when ``B > sqrt(f_j) D``; the share of a field ``x e_j + r`` outside it is at most ``(|x| sqrt(f_j) +
    |r|)^2 / (|x|^2 + |r|^2)``, which falls with ``|x|`` and rises with ``|r|`` there). The smaller of the bounds; 1 when
    neither holds."""
    u = 0.0
    for i in group:
        u += math.sqrt(f[i]) * amp[i]
    if u == 0.0:
        return 0.0  # (no beam has a part outside the state: neither has their sum)
    best = 1.0
    w_best = 0.0
    for j in wanted_refs:
        w = math.sqrt(1.0 - f[j]) * amp[j]
        for i in group:
            if i != j:
                w -= math.sqrt(1.0 - f[i]) * amp[i]
        if w > w_best:
            w_best = w
    if w_best > 0.0:
        best = u * u / (u * u + w_best * w_best)
    for j in state_refs:
        b, d = amp[j], 0.0
        for i in group:
            if i == j:
                continue
            o = _overlap2(states[j], states[i])  # type: ignore[arg-type]
            if o >= 1.0 - PARALLEL_ROUNDING:
                o = 1.0
            b -= amp[i] * math.sqrt(o)
            d += amp[i] * math.sqrt(1.0 - o)
        s = math.sqrt(f[j])
        if d == 0.0:
            share = f[j]  # (every beam in the state of j: so is their sum)
        elif b > s * d:
            share = (b * s + d) * (b * s + d) / (b * b + d * d)
        else:
            continue
        if share < best:
            best = share
    return best


def _coherent_bound(condition: str, group: tuple[int, ...], contribs: list[_HitValues], states: list[_State | None],
                    shares: dict[int, float], q: Quantity) -> float:
    """The largest share of the combined light of the mutually coherent beams ``group`` (of different polarization
    states) outside the state of ``condition`` for any of their relative phases, in field amplitudes of the quantity ``q``
    (``_bound_in``): ``LINEAR`` / ``CIRCULAR`` in the common state of ``_holds`` (``shares``), ``S`` / ``P`` along the s
    vector of one traced beam of the group (its plane of incidence), the smallest over them. A beam's part in the state is
    taken at least as its own only for a traced beam of known Gaussian profile (its value at its centre is exact; the
    others' values are upper bounds). 1 when a beam has no value in ``q``."""
    amp: dict[int, float] = {}
    for i in group:
        value = q(contribs[i])
        if value is None:
            return 1.0
        amp[i] = math.sqrt(value)
    refs = [i for i in group if contribs[i].light == TRACED and contribs[i].located]
    if condition not in ("S", "P"):
        return _bound_in(group, {i: min(1.0, shares[i]) for i in group}, amp, states, refs, refs)
    best = 1.0
    for j in refs:
        s_vec = _s_unit(contribs[j].raw)  # type: ignore[arg-type]
        if s_vec is None:
            continue
        f: dict[int, float] = {}
        for i in group:
            along = _along(states[i], s_vec)  # type: ignore[arg-type]
            f[i] = along if condition == "P" else max(0.0, 1.0 - along)
        best = min(best, _bound_in(group, f, amp, states, refs, [j]))
    return best


def _field_quantities(rating: PowerRating) -> tuple[_Quantity, ...]:
    """The field quantities the combined state of mutually coherent beams is checked in: the peak irradiance, and the
    peak fluence for a ``J/cm2`` rating (their square roots are field amplitudes)."""
    return (_peak, _fluence) if rating.unit == "J/cm2" else (_peak,)


def _beam(v: _HitValues) -> str:
    return f"{v.hit.source_id} ({v.light} {v.hit.path_id}, pass {v.hit.pass_index})"


def _least_share(ranges: list[tuple[float, float, float]]) -> float | None:
    """The smallest ``sum P_c f_c / sum P_c`` over ``P_c`` in ``[lo_c, hi_c]`` for ``ranges`` (f_c, lo_c, hi_c): a
    linear-fractional function, smallest at a corner where the classes of the smaller shares take their largest amount
    and the others their least (``None`` when every corner holds no light)."""
    items = sorted(ranges)
    best: float | None = None
    for k in range(len(items) + 1):
        num, den = 0.0, 0.0
        for index, (f, lo, hi) in enumerate(items):
            w = hi if index < k else lo
            num += w * f
            den += w
        if den > 0.0:
            share = num / den
            if best is None or share < best:
                best = share
    return best


def _reference(contribs: list[_HitValues], index: list[int]) -> int:
    """The beam whose state the common state of a condition is taken from: the strongest traced beam (else the strongest)."""
    traced = [i for i in index if contribs[i].light == TRACED]
    return max(traced or index, key=lambda i: (contribs[i].hit.incident_power_W, -i))


def _outside_circular(ref: _State, direction, state: _State) -> float:
    """The share of a beam's power outside the circular state nearest the reference beam's (direction of travel
    ``direction``): ``1 - |c^* . E|^2 / |E|^2`` with ``c = (u + i s w) / sqrt(2)``, ``u`` the reference's major axis (its
    field's real part for a circular one), ``w = d x u`` and ``s`` its sense of rotation. Compared in the laboratory frame,
    so a beam travelling back along the reference with the same field (its reflection at normal incidence) is in it."""
    (ar, ai) = ref.field
    u = ref.axis
    if u is None:
        base = ar if ar[0] * ar[0] + ar[1] * ar[1] + ar[2] * ar[2] > 0.0 else ai
        length = math.sqrt(base[0] * base[0] + base[1] * base[1] + base[2] * base[2])
        u = (base[0] / length, base[1] / length, base[2] / length)
    d = (float(direction[0]), float(direction[1]), float(direction[2]))
    wx, wy, wz = d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]
    wl = math.sqrt(wx * wx + wy * wy + wz * wz)
    w = (wx / wl, wy / wl, wz / wl)

    def parts(st: _State) -> tuple[float, float, float, float]:
        (re, im) = st.field
        return (u[0] * re[0] + u[1] * re[1] + u[2] * re[2], u[0] * im[0] + u[1] * im[1] + u[2] * im[2],
                w[0] * re[0] + w[1] * re[1] + w[2] * re[2], w[0] * im[0] + w[1] * im[1] + w[2] * im[2])

    ua, ub, wa, wb = parts(ref)
    sense = 1.0 if ua * wb - ub * wa >= 0.0 else -1.0
    ua, ub, wa, wb = parts(state)
    inside = (ua * ua + ub * ub + wa * wa + wb * wb + 2.0 * sense * (ua * wb - ub * wa)) / (2.0 * state.norm2)
    return max(0.0, 1.0 - inside)


def _holds(condition: str, contribs: list[_HitValues],
           states: list[_State | None], tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[bool, float, int | None, dict[int, float]]:
    """Whether every beam is within the tolerance of one common state of ``condition``, the largest share outside it,
    that beam and every beam's share. Then so is any incoherent mixture of them (its share outside the state is the
    power-weighted mean of the beams' shares) and any sum of beams in one state; mutually coherent beams of different
    states add their fields, and their sum is checked as a whole (``_coherent_bound``). ``S`` / ``P``: s or p in each
    beam's own plane of incidence; ``LINEAR``: the major axis of the reference beam (``_reference``); ``CIRCULAR``: the
    circular state nearest the reference beam's (``_outside_circular``); ``UNPOLARIZED`` never holds."""
    index = [i for i, state in enumerate(states) if state is not None]
    if condition == "UNPOLARIZED":
        return False, 1.0, index[0] if index else None, {}
    if not index:
        return True, 0.0, None, {}
    shares: dict[int, float] = {}
    if condition in ("S", "P"):
        for i in index:
            share = _orthogonal_share(condition, states[i])  # type: ignore[arg-type]
            if share is None:
                return False, 0.0, None, {}  # normal incidence (named among the conditions not verified)
            shares[i] = share
    elif condition == "LINEAR":
        ref = _reference(contribs, index)
        axis = states[ref].axis  # type: ignore[union-attr]
        if axis is None:
            return False, 0.5, ref, {}
        shares = {i: _outside_axis(states[i], axis) for i in index}  # type: ignore[arg-type]
    else:  # CIRCULAR
        ref = _reference(contribs, index)
        direction = contribs[ref].raw.direction  # type: ignore[union-attr]
        shares = {i: _outside_circular(states[ref], direction, states[i]) for i in index}  # type: ignore[arg-type]
    worst = max(index, key=lambda i: (shares[i], -i))
    return shares[worst] <= tol.polarization_fraction, shares[worst], worst, shares


def _faint_share(contribs: list[_HitValues], classes: list[_Class], shares: dict[int, float],
                 q: Quantity = _peak, others: frozenset[int] = frozenset(), tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[float, float] | None:
    """The share of the most the light can reach that may be outside a condition's state when the left-out light, of
    unknown polarization, and the beams ``others`` (of another state) are entirely outside it: ``sum_c (sqrt(f_c) S_c +
    O_c + A_c)^2 / sum_c (S_c + O_c + A_c)^2`` (module docstring) in amplitudes of the quantity ``q``, ``S_c`` the class's
    other beams with ``f_c`` their largest share outside the state in ``shares`` (a beam's own; for mutually coherent beams
    of different states the largest their sum may have, ``_coherent_bound``), ``O_c`` the beams ``others``, and the factor the
    left-out amplitudes must shrink by for the tolerance (with half the margin to spare; NaN when the beams' own shares
    leave none). ``None`` without left-out light or ``others``. A beam without a value in ``q`` is left out where its
    class holds neither (more of a class within the tolerance only draws the share toward its own), and leaves the
    share unbounded (1, NaN) where it does: its part outside the state may add in phase to theirs."""
    if not (others or any(cls.amplitude(q) > 0.0 for cls in classes)):
        return None
    outside = most = known = known_outside = left = 0.0
    for cls in classes:
        amp = cls.amplitude(q)
        f = summed = apart = 0.0
        unvalued = False
        for i in cls.members:
            value = q(contribs[i])
            if value is None:
                unvalued = True
                continue
            summed += math.sqrt(value)
            if i in others:
                apart += math.sqrt(value)
            elif shares.get(i, 0.0) > f:
                f = shares.get(i, 0.0)
        if unvalued and (amp > 0.0 or apart > 0.0 or any(i in others for i in cls.members)):
            return 1.0, math.nan
        root = math.sqrt(f) * (summed - apart) + apart
        outside += (root + amp) * (root + amp)
        most += (summed + amp) * (summed + amp)
        known += summed * summed
        known_outside += root * root
        left += amp
    share = outside / most if most > 0.0 else 1.0
    spare = tol.unlocated_light_share * known - known_outside
    scale = (0.5 * (math.sqrt(tol.unlocated_light_share * known) - math.sqrt(known_outside)) / left
             if spare > 0.0 and left > 0.0 else None)
    return share, (scale if scale is not None and scale > 0.0 else math.nan)


def _largest_share(found: list[tuple[float, float] | None], tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[float, float] | None:
    """The largest of ``_faint_share``'s shares in several quantities, with the factor that brings every one above the
    tolerance within it (NaN when one of them cannot be)."""
    known = [x for x in found if x is not None]
    if not known:
        return None
    over = [scale for share, scale in known if share > tol.unlocated_light_share]
    scale = math.nan if not over or any(math.isnan(s) for s in over) else min(over)
    return max(share for share, _ in known), scale


def _s_unit(h: SurfaceHit) -> tuple[float, float, float] | None:
    """The unit s vector ``d x n / |d x n|`` of a beam's plane of incidence (``None`` at normal incidence)."""
    d, n = h.direction, h.normal
    sx = float(d[1]) * float(n[2]) - float(d[2]) * float(n[1])
    sy = float(d[2]) * float(n[0]) - float(d[0]) * float(n[2])
    sz = float(d[0]) * float(n[1]) - float(d[1]) * float(n[0])
    length = math.sqrt(sx * sx + sy * sy + sz * sz)
    return None if length <= NORMAL_INCIDENCE_SIN else (sx / length, sy / length, sz / length)


def _orthogonal_least(condition: str, cls: _Class, contribs: list[_HitValues], states: list[_State | None], p, tol: RatingTolerances = DEFAULT_TOLERANCES) -> float:
    """A lower bound of the intensity a class's light carries at ``p`` in the state orthogonal to ``S`` / ``P``, whatever
    the phases: the largest traced member's orthogonal amplitude less every other member's and the left-out light's (the
    orthogonal parts add as vectors). 0 when the members do not share one plane of incidence (no common s)."""
    first: tuple[float, float, float] | None = None
    amplitude: dict[int, float] = {}
    for i in cls.members:
        v, state = contribs[i], states[i]
        if state is None:
            amplitude[i] = 0.0
            continue
        s_vec = _s_unit(v.raw)  # type: ignore[arg-type]
        if s_vec is None:
            return 0.0
        if first is None:
            first = s_vec
        else:
            c = first[0] * s_vec[0] + first[1] * s_vec[1] + first[2] * s_vec[2]
            if c * c < 1.0 - tol.polarization_fraction:
                return 0.0
        value = _peak(v)
        assert value is not None
        here = value * _profile(v, p) if v.located else value
        share = _orthogonal_share(condition, state)
        amplitude[i] = math.sqrt((share or 0.0) * here)
    traced = [i for i in cls.members if contribs[i].light == TRACED]
    if not traced:
        return 0.0
    k = max(traced, key=lambda i: (amplitude[i], -i))
    rest = amplitude[k]
    for i in cls.members:
        if i != k:
            rest -= amplitude[i]
    rest -= cls.amplitude(_peak)
    return rest * rest if rest > 0.0 else 0.0


def _violated(condition: str, contribs: list[_HitValues], classes: list[_Class], states: list[_State | None],
              undetermined: set[int], tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[float, int] | None:
    """Where the light certainly violates ``condition``: the largest, over the traced beams' centres, of the smallest share
    there of the light's power in the state orthogonal to it (``UNPOLARIZED``: in the state of one class), when above the
    tolerance, with that beam; ``None`` otherwise. At a centre each class's amount lies between its lower and upper bound
    there (``_class_range`` of the peak irradiance, the left-out light included). ``S`` / ``P``: the orthogonal light's
    least intensity (``_orthogonal_least``) against the classes' largest. ``LINEAR`` / ``CIRCULAR``: a class's share is at
    least the smallest of its members' own shares (0 for a class with mutually coherent beams of different states,
    ``undetermined``: their combined state depends on their phases), and
    with left-out light of amplitude ``A`` (unknown state) at least ``((sqrt(f) e - A) / (e + A))^2`` with ``e`` the
    least amplitude of its own light there; a mixture's share outside a linear or circular state is at least the
    power-weighted mean of its parts' own (the largest eigenvalue of a sum is at most the sum of the largest eigenvalues),
    smallest at a corner of the classes' ranges (``_least_share``). ``UNPOLARIZED``: a class of one state carrying at
    least its lower bound in that state against the most the others can reach (with left-out light: ``(e - A)^2`` in its
    state against at most ``A^2`` of its own light in another)."""
    least: list[float] = []
    for cls in classes:
        if condition in ("UNPOLARIZED", "S", "P") or any(i in undetermined for i in cls.members) or not cls.members:
            least.append(0.0)
            continue
        own = [_orthogonal_share(condition, states[i]) for i in cls.members if states[i] is not None]  # type: ignore[arg-type]
        least.append(min((s if s is not None else 0.0) for s in own) if own else 0.0)
    found: tuple[float, int] | None = None
    for c in _centres(contribs, _peak):
        p = contribs[c].raw.position  # type: ignore[union-attr]
        ranges: list[tuple[float, float, float]] = []
        in_state: list[float | None] = []  # (UNPOLARIZED) the least intensity of a class's light in its own state
        for cls, f in zip(classes, least):
            bounds = _class_range(cls, contribs, _peak, p)
            if bounds is None:
                break
            amp = cls.amplitude(_peak)
            own_state: float | None = bounds[0] if cls.members else None
            if amp > 0.0 and cls.members:
                bare = _class_range(replace(cls, faint=None), contribs, _peak, p)
                e = math.sqrt(bare[0]) if bare is not None else 0.0
                gap = math.sqrt(f) * e - amp
                f = (gap / (e + amp)) * (gap / (e + amp)) if gap > 0.0 else 0.0
                rest = e - amp
                own_state = rest * rest if rest > 0.0 else 0.0
            ranges.append((f, bounds[0], bounds[1]))
            in_state.append(own_state)
        else:
            share: float | None = None
            if condition == "UNPOLARIZED":
                total_hi = 0.0
                for _, _, hi in ranges:
                    total_hi += hi
                for cls, (_, lo, hi), own_state in zip(classes, ranges, in_state):
                    if own_state is None or own_state <= 0.0 or any(i in undetermined for i in cls.members):
                        continue
                    # (left-out light in the class, of a state not known: at most its own intensity A^2 is in another state)
                    amp = cls.amplitude(_peak)
                    den = lo + (total_hi - hi) if amp == 0.0 else own_state + amp * amp + (total_hi - hi)
                    if den > 0.0 and (share is None or own_state / den > share):
                        share = own_state / den
                violated = share is not None and share > 0.5 + tol.polarization_fraction
            elif condition in ("S", "P"):
                num, den = 0.0, 0.0
                for cls, (_, _, hi) in zip(classes, ranges):
                    num += _orthogonal_least(condition, cls, contribs, states, p, tol=tol)
                    den += hi
                share = num / den if den > 0.0 else None
                violated = share is not None and share > tol.polarization_fraction
            else:
                share = _least_share(ranges)
                violated = share is not None and share > tol.polarization_fraction
            if violated and (found is None or share > found[0]):  # type: ignore[operator]
                found = (share, c)  # type: ignore[assignment]
    return found


@dataclass(frozen=True)
class _Checked:
    """A condition checked against the light on a surface: ``mismatches`` (it does not hold), ``unverified`` (it cannot be
    checked) with the ``reasons`` (``RatingCondition.reason``, one per text), ``notes``, and what the light left out as
    negligible alone leaves unverified (``faint``) with the factor its amplitudes must shrink by for the check (``scale``;
    ``None`` when that cannot help)."""

    mismatches: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    faint: tuple[str, ...] = ()
    scale: float | None = None
    reasons: tuple[str, ...] = ()


def _polarization(rating: PowerRating, contribs: list[_HitValues], classes: list[_Class], surface_id: str,
                  unknown_light: str | None, tol: RatingTolerances = DEFAULT_TOLERANCES) -> _Checked:
    """A rating's test polarization against the light on the surface as a whole (``RatingComparison``): it holds when
    every beam is within the tolerance of one common state of it (``_holds``) -- mutually coherent beams of different
    states as a sum, for every relative phase (``_coherent_bound``) -- and the left-out light's share stays within it too
    (``_faint_share``), does not hold where the light at a traced beam's centre violates it for certain (``_violated``),
    and is not verified otherwise -- or when ``unknown_light`` (light of unknown fate, whose state the trace does not
    know) may reach the surface."""
    if rating.polarization is None:
        if rating.polarization_condition:
            return _Checked(unverified=(f"polarization: rated for {rating.polarization_condition!r}, a polarization stated as "
                                        "text only (the comparison reads linear, circular, s, p or unpolarized)",),
                            reasons=(RATING_INFORMATION_MISSING,))
        return _Checked()
    condition = rating.polarization
    wanted = _POLARIZATION_TEXT[condition]
    states = [_state(v.raw) for v in contribs]  # type: ignore[arg-type]
    unverified: list[str] = []
    reasons: list[str] = []  # (one per unverified text)
    if rating.polarization_condition and rating_polarization(rating.polarization_condition) != condition:
        # (a text that is not a plain form of the field states more than it -- an orientation, a handedness, a geometry)
        unverified.append(f"polarization: rated for {rating.polarization_condition!r}, which states more than {wanted} light "
                          "(the comparison checks the light against linear, circular, s, p or unpolarized only)")
        reasons.append(RATING_INFORMATION_MISSING)
    if unknown_light is not None:
        unverified.append(f"polarization: rated for {wanted} light; {unknown_light} may reach {surface_id} in a state the "
                          "trace does not know")
        reasons.append(LIGHT_STATE_UNKNOWN)
    # mutually coherent beams of different states: their combined state depends on their phases
    mixed = [group for cls in classes for group, one in _interfering(cls, states) if not one]
    undetermined: set[int] = {i for group in mixed for i in group}
    if condition in ("S", "P"):
        normal = [contribs[i] for i, state in enumerate(states) if state is not None and state.s_share is None]
        if normal:
            unverified.append(f"polarization: rated for {wanted} light; {len(normal)} beam{'s' if len(normal) != 1 else ''} "
                              f"of {', '.join(sorted({v.hit.source_id for v in normal}))} meet{'s' if len(normal) == 1 else ''} "
                              f"{surface_id} at normal incidence, where s and p are not defined")
            reasons.append(RATING_INFORMATION_MISSING)
    held, worst, worst_index, shares = _holds(condition, contribs, states, tol=tol)
    # (mutually coherent beams of different states hold the condition only as a sum: the most of it that can be outside
    # the state whatever their relative phase, in each field quantity, stands for each of them -- the beam's own share
    # neither decides it nor is enough)
    bounds: list[tuple[tuple[int, ...], float]] = []
    spread: set[tuple[int, ...]] = set()  # (groups of beams of different profiles: their combined state varies on the surface)
    if shares and mixed:
        shares = dict(shares)
        for group in mixed:
            bound = max(_coherent_bound(condition, group, contribs, states, shares, q) for q in _field_quantities(rating))
            # (a bound above zero holds at every point only for beams of one amplitude profile: otherwise their amplitude
            # ratio, and with it their combined state, varies across the surface -- not verified; zero, no beam with a part
            # outside the state beyond rounding, holds wherever they overlap)
            if bound > PARALLEL_ROUNDING and not _one_profile(group, contribs):
                bound = 1.0
                spread.add(group)
            bounds.append((group, bound))
        for group, bound in bounds:
            for i in group:
                shares[i] = bound
        worst_index = max(shares, key=lambda i: (shares[i], -i))
        worst = shares[worst_index]
        held = worst <= tol.polarization_fraction
    single = {i: f for i, f in shares.items() if i not in undetermined}

    def coherent_texts(excluded: frozenset[int]) -> list[str]:
        out: list[str] = []
        for group, bound in bounds:
            if bound > tol.polarization_fraction and not all(i in excluded for i in group):
                names = ", ".join(sorted({contribs[i].hit.source_id for i in group}))
                if group in spread:
                    out.append(f"polarization: mutually coherent beams of {names} in different polarization states meet "
                               f"{surface_id} with different footprints (displaced from each other, of different sizes or from different "
                               "directions); their combined state "
                               "depends on their relative phase and varies across the surface where they overlap")
                    continue
                out.append(f"polarization: mutually coherent beams of {names} in different polarization states meet "
                           f"{surface_id}; their combined state depends on their relative phase, and up to "
                           f"{PERCENT * bound:.3g} % of their light may be outside the rating's {wanted} polarization "
                           f"(tolerance {PERCENT * tol.polarization_fraction:g} %)")
        return out

    coherent = coherent_texts(frozenset())
    if not shares and condition != "UNPOLARIZED":
        # (no common state to bound them in -- s / p at normal incidence, a circular reference for a linear axis)
        coherent = [f"polarization: mutually coherent beams of "
                    f"{', '.join(sorted({contribs[i].hit.source_id for i in group}))} in different polarization states "
                    f"meet {surface_id}; their combined state depends on their relative phase" for group in mixed]
    # (beams of another state that carry at most the tolerance of the most light in every quantity that gives the
    # comparison its direction: in the exposure, not in the state of the light as a whole -- the rule for light left out as
    # negligible and for minor beams; a beam without a value in one of them is never left out. They are left out only when
    # the light outside the state as a whole -- they entirely outside it, the other beams' own shares and the left-out
    # light, in each of those quantities -- stays within the tolerance too, or would without the left-out light (then a
    # deeper pass may decide it); otherwise the light is judged with them)
    others: frozenset[int] = frozenset()
    apart_share = 0.0
    scale: float | None = None
    faint_text: str | None = None
    if not held and shares and condition != "UNPOLARIZED":
        apart = frozenset(i for i, f in shares.items() if f > tol.polarization_fraction)
        if 0 < len(apart) < len(shares):
            quantities = _line_share_quantities(rating)
            alone = _followed_share(contribs, classes, set(apart), quantities)
            combined = (_largest_share([_faint_share(contribs, classes, shares, q, apart, tol=tol) for q in quantities], tol=tol)
                        if alone is not None and alone <= tol.unlocated_light_share else None)
            if combined is not None and (combined[0] <= tol.unlocated_light_share or not math.isnan(combined[1])):
                others, held, apart_share = apart, True, combined[0]
                worst = max(f for i, f in shares.items() if i not in others)
                if combined[0] > tol.unlocated_light_share:
                    held, scale = False, combined[1]
                    faint_text = (f"polarization: rated for {wanted} light; light the trace leaves out as negligible, in a "
                                  f"state it does not know, may reach {surface_id}: with the beams of another state here the "
                                  f"light outside it may carry up to {PERCENT * combined[0]:.3g} % of the most light there "
                                  f"(tolerance {PERCENT * tol.unlocated_light_share:g} %)")
    if others:
        # (mutually coherent beams of different states, other than those beams: a group beyond the tolerance has every
        # beam among them)
        coherent = coherent_texts(others)
    if held and not others and condition != "UNPOLARIZED":
        # (in peak irradiance and in the rating's own quantity: the larger share -- mutually coherent beams of different
        # states with the most their sum may have outside the state; with beams of another state the share was checked
        # above, in every quantity that gives the comparison its direction)
        faint = _largest_share([_faint_share(contribs, classes, shares, q, tol=tol) for q in (_peak, *_share_quantities(rating))], tol=tol)
        if faint is not None and faint[0] > tol.unlocated_light_share:
            held = False
            scale = None if math.isnan(faint[1]) else faint[1]
            faint_text = (f"polarization: rated for {wanted} light; light the trace leaves out as negligible, in a state it "
                          f"does not know, may reach {surface_id} with up to {PERCENT * faint[0]:.3g} % of the most light "
                          f"there (tolerance {PERCENT * tol.unlocated_light_share:g} %)")
    if held and not unverified and unknown_light is None and _violated(condition, contribs, classes, states,
                                                                       undetermined, tol=tol) is not None:
        held = False  # (light that violates the condition for certain somewhere: the mismatch below)
    if held and not unverified:
        common = " along one common axis" if condition == "LINEAR" else ""
        if others:
            names = sorted({_beam(contribs[i]) for i in others})
            return _Checked(notes=(f"polarization: {wanted}{common} at every beam here but the beam of "
                                   f"{', '.join(names[:3])}{', ...' if len(names) > 3 else ''} (largest share outside it "
                                   f"{PERCENT * worst:.3g} %); the light outside it carries at most {PERCENT * apart_share:.3g} "
                                   f"% of the most light there (tolerance {PERCENT * tol.unlocated_light_share:g} %): it "
                                   "counts in the exposure, not in the rating's polarization",))
        if bounds:
            names = ", ".join(sorted({contribs[i].hit.source_id for group, _ in bounds for i in group}))
            rest = f"; every other beam at most {PERCENT * max(single.values()):.3g} % outside it" if single else ""
            return _Checked(notes=(f"polarization: {wanted}{common} here: the mutually coherent beams of {names}, in "
                                   f"different states, have at most {PERCENT * max(b for _, b in bounds):.3g} % of their light "
                                   f"outside it whatever their relative phase{rest}",))
        return _Checked(notes=(f"polarization: {wanted}{common} at every beam here (largest share outside it "
                               f"{PERCENT * worst:.3g} %)",))
    if unknown_light is None:
        found = _violated(condition, contribs, classes, states, undetermined, tol=tol)
        if found is not None:
            share, centre = found
            if condition == "UNPOLARIZED":
                text = (f"polarization: the rating is for unpolarized light; at the centre of the beam of "
                        f"{_beam(contribs[centre])} at least {PERCENT * share:.3g} % of the light is in one polarization "
                        "state (every traced beam is fully polarized)")
            else:
                text = (f"polarization: at the centre of the beam of {_beam(contribs[centre])} at least {PERCENT * share:.3g} % "
                        f"of the light's power is in the state orthogonal to the rating's {wanted} polarization (tolerance "
                        f"{PERCENT * tol.polarization_fraction:g} %)")
            return _Checked(mismatches=(text,), unverified=tuple(unverified), reasons=tuple(reasons))
    if faint_text is not None:
        return _Checked(unverified=tuple(unverified), faint=(faint_text,), scale=scale, reasons=tuple(reasons))
    if condition == "UNPOLARIZED":
        unverified.append("polarization: rated for unpolarized light; the beams here are polarized, and the degree of "
                          "polarization of their mixture where they overlap is not determined")
        reasons.append(LIGHT_STATE_UNKNOWN)
    elif not held and coherent:
        unverified.extend(coherent)
        reasons.extend(COHERENT_STATE_UNDETERMINED for _ in coherent)
    elif not held and worst_index is not None:
        unverified.append(f"polarization: the beams here are not all within {PERCENT * tol.polarization_fraction:g} % "
                          f"of one {wanted} state (up to {PERCENT * worst:.3g} % outside it: the beam of "
                          f"{_beam(contribs[worst_index])}), and the state of their mixture where they overlap is not "
                          "determined")
        reasons.append(LIGHT_STATE_UNKNOWN)
    return _Checked(unverified=tuple(unverified), reasons=tuple(reasons))


def _beam_sizes(contribs: list[_HitValues], cfg: NumericalConfig) -> tuple[dict[int, BeamSize], list[str]]:
    """The beams of known size on the surface by contribution index (traced, untraced and beside it: both transverse 1/e^2
    diameters and the footprint's largest 1/e^2 extent there) and the beams whose size there is not known (no valid
    Gaussian state, a response model not validated for the beam size). Light bounded by its diffraction limit is in neither
    (``_unsized_share``)."""
    sizes: dict[int, BeamSize] = {}
    unknown: list[str] = []
    for i, v in enumerate(contribs):
        if v.limited:
            continue
        wx, wy = v.hit.beam_radius_x_mm, v.hit.beam_radius_y_mm
        if wx is None or wy is None:
            unknown.append(_beam(v))
            continue
        footprint = _footprint(v.raw, wx, wy, cfg) if v.light == BESIDE else v.hit.footprint_semi_axes_mm  # type: ignore[arg-type]
        sizes[i] = BeamSize((2.0 * wx, 2.0 * wy), None if footprint is None else 2.0 * footprint[0], _beam(v))
    return sizes, unknown


def _share_quantities(rating: PowerRating) -> tuple[_Quantity, ...]:
    """The quantities a share of the light on a surface is measured in for ``rating``: its own exposure quantity
    (``_quantity``: peak irradiance, ``P / d`` or peak fluence); both readings of ``P / d`` for a ``W/cm`` rating that does
    not state its diameter definition."""
    if _diameter_range(rating):
        return _linear("1/E2"), _linear("FWHM")
    return (_quantity(rating),)


def _unsized_share_in(contribs: list[_HitValues], classes: list[_Class], q: Quantity, minor: set[int],
                      tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[float, float | None] | None:
    """``_unsized_share`` in the one quantity ``q`` (contributions without a value in it count in neither part)."""
    outside = most = known = fixed = left = 0.0
    for cls in classes:
        sized = unsized = 0.0
        for i in cls.members:
            value = q(contribs[i])
            if value is None:
                continue
            if contribs[i].limited or i in minor:
                unsized += math.sqrt(value)
            else:
                sized += math.sqrt(value)
        amp = cls.amplitude(q)
        outside += (unsized + amp) * (unsized + amp)
        most += (sized + unsized + amp) * (sized + unsized + amp)
        known += sized * sized
        fixed += unsized * unsized
        left += amp
    if not outside > 0.0:
        return None
    share = outside / most if most > 0.0 else 1.0
    spare = tol.unlocated_light_share * known - fixed
    scale = 0.5 * (math.sqrt(tol.unlocated_light_share * known) - math.sqrt(fixed)) / left if spare > 0.0 and left > 0.0 else None
    return share, scale


def _size_quantities(rating: PowerRating) -> tuple[_Quantity, ...]:
    """The quantities the beams on a surface are weighed in for the beam-size comparison under ``rating``: every quantity
    that gives the comparison its direction -- the rating's own (``_share_quantities``) and, for a CW rating, the power and
    ``P / d`` (a beam's heating scales with its ``P / d``, and a broad beam heats the element as a whole with its power),
    for a pulsed rating the peak fluence and the pulse energy (a broad beam of low fluence meets more of the surface's
    defects). A beam is left out of the comparison only when it is minor in each of them (``_minor_beams``). A beam beside
    the surface counts in each with its tail bound (``_Quantity.weighed`` for ``P / d``)."""
    out = [replace(q, weighed=True) if q.kind == "linear" else q for q in _share_quantities(rating)]
    for q in ((_power, replace(_linear("1/E2"), weighed=True)) if rating.regime == "CW" else (_fluence, _energy)):
        if q not in out:
            out.append(q)
    return tuple(out)


def _unsized_share(contribs: list[_HitValues], classes: list[_Class], qs: tuple[_Quantity, ...],
                   minor: set[int] = frozenset(),  # type: ignore[assignment]
                   tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[float, float | None, _Quantity] | None:
    """The share of the most light the surface can receive that light of unknown size -- untraced light bounded by its
    diffraction limit, the light left out as negligible, and the ``minor`` beams (whose size the comparison leaves out) --
    may carry, ``sum_c (U_c + A_c)^2 / sum_c (S_c + U_c + A_c)^2`` in amplitudes of each quantity of ``qs``
    (``_size_quantities``; the largest share counts, with its quantity), with ``S_c`` the class's other beams of known
    value, ``U_c`` its light of unknown size and ``A_c`` its left-out light; and the factor the left-out amplitudes must
    shrink by for the tolerance in every quantity (with half the margin to spare; ``None`` when the other light of unknown
    size alone exceeds it). ``None`` when there is no such light."""
    found = [(x, q) for x, q in ((_unsized_share_in(contribs, classes, q, minor, tol=tol), q) for q in qs) if x is not None]
    if not found:
        return None
    over = [scale for (share, scale), _ in found if share > tol.unlocated_light_share]
    scale = None if not over or any(s is None for s in over) else min(s for s in over if s is not None)
    (share, _), q = max(found, key=lambda item: item[0][0])
    return share, scale, q


def _line_share_quantities(rating: PowerRating) -> tuple[_Quantity, ...]:
    """The quantities a share of followed light of another line or polarization state is measured in: peak irradiance
    and every quantity that gives a rating's comparison its direction (``_size_quantities``)."""
    out = [_peak]
    for q in _size_quantities(rating):
        if q not in out:
            out.append(q)
    return tuple(out)


def _followed_share(contribs: list[_HitValues], classes: list[_Class], picked: set[int],
                    qs: tuple[_Quantity, ...]) -> float | None:
    """The largest, over the quantities ``qs``, share of the most light the surface can receive that the contributions
    ``picked`` carry: ``sum_c O_c^2 / sum_c (S_c + A_c)^2`` with ``O_c`` their summed amplitudes in class ``c``, ``S_c``
    all its members' and ``A_c`` its left-out light's. ``None`` when one of them has no value in a quantity (it is never
    left out)."""
    best = 0.0
    for q in qs:
        outside = most = 0.0
        for cls in classes:
            summed = mine = 0.0
            for i in cls.members:
                value = q(contribs[i])
                if value is None:
                    if i in picked:
                        return None
                    continue
                root = math.sqrt(value)
                summed += root
                if i in picked:
                    mine += root
            amp = cls.amplitude(q)
            outside += mine * mine
            most += (summed + amp) * (summed + amp)
        if outside > 0.0:
            share = outside / most if most > 0.0 else 1.0
            if share > best:
                best = share
    return best


def _minor_lines(rating: PowerRating, contribs: list[_HitValues], classes: list[_Class], tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[frozenset[int], float]:
    """The followed contributions of lines the rating is not for (``line_differences``: another regime, wavelength,
    pulse duration or repetition rate) when together they carry at most the tolerance of the most light the surface can
    receive in every quantity of ``_line_share_quantities`` (the rule for light left out as negligible and for minor
    beams): they still count in the exposure, not in the rating's conditions. Their share with them (0 and an empty set
    when there is no such light, or it carries more)."""
    off = {i for i, v in enumerate(contribs) if line_differences(rating, v.hit.wavelength_nm, v.pulse, tol)}
    if not off or len(off) == len(contribs):
        return frozenset(), 0.0
    share = _followed_share(contribs, classes, off, _line_share_quantities(rating))
    if share is None or share > tol.unlocated_light_share:
        return frozenset(), 0.0
    return frozenset(off), share


def _minor_beams(contribs: list[_HitValues], classes: list[_Class], sizes: dict[int, BeamSize],
                 qs: tuple[_Quantity, ...], tol: RatingTolerances = DEFAULT_TOLERANCES) -> set[int]:
    """The weakest beams of known size that, with the light of unknown size in ``classes`` (``_beam_condition`` passes the
    classes without the light left out as negligible: the followed untraced light of unknown size), carry at most the
    tolerance of the most light the surface can receive in every quantity of ``qs`` (``_size_quantities``,
    ``_unsized_share``): the beams the size comparison leaves out (a faint ghost of another size does not change the size
    of the light that meets the rating's conditions). A beam is taken in order of its value in the rating's own quantity
    and left out only when the beams left out with it stay within the tolerance in each quantity; a beam without a value in
    one of them is never left out."""
    minor: set[int] = set()
    weighed = [i for i in sizes if all(q(contribs[i]) is not None for q in qs)]
    for i in sorted(weighed, key=lambda k: (qs[0](contribs[k]), k)):
        found = _unsized_share(contribs, classes, qs, minor | {i}, tol=tol)
        if found is None or found[0] > tol.unlocated_light_share:
            continue
        minor.add(i)
    return minor


def _beam_condition(rating: PowerRating, contribs: list[_HitValues], classes: list[_Class], surface_id: str,
                    cfg: NumericalConfig, design_pulses: float | None = None, tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[RatingCondition, float | None, bool]:
    """The ``beam_diameter`` record (``rating_conditions.beam_diameter_condition``) of the light on the surface: the beams of
    known size that carry it (``_minor_beams`` left out) and the light of unknown size beyond the tolerance; the factor the
    left-out light's amplitudes must shrink by when they alone keep it from being decided, and whether a deeper pass can
    decide it at all. ``design_pulses``: the most pulses a site receives (``PulsesPerSite.high``; ``None``: not stated)."""
    if rating.beam_diameter_mm is None:
        return beam_diameter_decision(rating, [], tolerances=tol)[0], None, False
    sizes, unknown = _beam_sizes(contribs, cfg)
    unknown_text = (f"the size of the beam of {', '.join(unknown[:3])}{', ...' if len(unknown) > 3 else ''} on {surface_id} is "
                    "not known") if unknown else None
    qs = _size_quantities(rating)
    # (the weakest beams are minor by their share with the followed light of unknown size; the light left out as negligible
    # is then checked with them: beyond the tolerance it leaves the size not verifiable, and a deeper pass may decide it)
    minor = _minor_beams(contribs, _without_faint(classes), sizes, qs, tol=tol)
    unsized_text: str | None = None
    scale: float | None = None
    found = _unsized_share(contribs, classes, qs, minor, tol=tol)
    if found is not None and found[0] > tol.unlocated_light_share:
        scale = found[1]
        unsized_text = (f"light of unknown size (bounded by its diffraction limit: untraced light of unknown size, light left out "
                        f"as negligible) may reach {surface_id} with up to {PERCENT * found[0]:.3g} % of the most "
                        f"{_SHARE_NAME[found[2].kind]} there (tolerance {PERCENT * tol.unlocated_light_share:g} %)")
    carrying = [size for i, size in sorted(sizes.items()) if i not in minor]
    record, decisive = beam_diameter_decision(rating, carrying, unknown_size=unknown_text, unsized=unsized_text,
                                              design_pulses=design_pulses, tolerances=tol)
    return record, scale if decisive else None, decisive


def _left_out_share(contribs: list[_HitValues], classes: list[_Class], qs: tuple[_Quantity, ...]) -> float:
    """The largest, over ``qs``, share of the most light the surface can receive that the light left out as negligible may
    carry: ``sum_c A_c^2 / sum_c (S_c + A_c)^2`` (``S_c`` a class's summed amplitudes, ``A_c`` its left-out amplitude);
    1 where a member has no value in a quantity."""
    if not any(cls.faint is not None for cls in classes):
        return 0.0
    best = 0.0
    for q in qs:
        outside = most = 0.0
        for cls in classes:
            amp = cls.amplitude(q)
            summed = 0.0
            for i in cls.members:
                value = q(contribs[i])
                if value is None:
                    return 1.0
                summed += math.sqrt(value)
            outside += amp * amp
            most += (summed + amp) * (summed + amp)
        if outside > 0.0:
            best = max(best, outside / most if most > 0.0 else 1.0)
    return best


def _pulses_per_site(rating: PowerRating, contribs: list[_HitValues], classes: list[_Class], minor: frozenset[int],
                     left_out: tuple[Light, ...], open_above: bool, tol: RatingTolerances = DEFAULT_TOLERANCES) -> PulsesPerSite | None:
    """The pulses one site of the surface receives (``PulsesPerSite``; ``None`` for a continuous-wave rating):
    every crossing (``_crossing``: a source's light along one path, all its lines in one pulse) of the followed light of
    the rating's lines (``minor`` left out) brings its source's ``total_pulses``, except the weakest crossings that
    together carry at most the tolerance of the most light the surface can receive in every quantity of
    ``_line_share_quantities`` (a faint ghost's pulses; the crossing of the strongest beam always counts). A site receives
    at least the fewest one counted crossing brings and at most all of them; nothing bounds that where light left out as
    negligible may carry more than the tolerance there, or light may be stored (``open_above``). On a surface only light
    left out as negligible reaches, its sources (``left_out``) bring their totals over crossings it does not count."""
    if rating.regime != "PULSED":
        return None
    candidates = [i for i in range(len(contribs)) if i not in minor]
    if not candidates:
        pulsed_lines = [x for x in left_out if x.pulse is not None]
        if not pulsed_lines:
            return PulsesPerSite(0.0, 0.0, continuous=True, left_out=True)
        missing = tuple(sorted({x.source_id for x in pulsed_lines if total_pulses(x.pulse) is None}))
        stated = [n for n in (total_pulses(x.pulse) for x in pulsed_lines) if n is not None]
        return PulsesPerSite(min(stated) if stated else 0.0, math.inf, missing, crossings=0,
                             unbounded="the light here is left out as negligible, whose crossings are not counted")
    qs = _line_share_quantities(rating)
    own = _quantity(rating)
    strongest = max(candidates, key=lambda i: (own(contribs[i]) is not None, own(contribs[i]) or 0.0, -i))
    crossings: dict[tuple[str, tuple[str, ...], int], list[int]] = {}
    for i in candidates:
        crossings.setdefault(_crossing(contribs[i]), []).append(i)
    first = _crossing(contribs[strongest])
    weak: set[int] = set()
    weighed = [k for k, members in crossings.items() if k != first and all(q(contribs[i]) is not None for i in members for q in qs)]
    for k in sorted(weighed, key=lambda k: (max(own(contribs[i]) or 0.0 for i in crossings[k]), min(crossings[k]))):
        share = _followed_share(contribs, classes, weak | set(crossings[k]), qs)
        if share is not None and share <= tol.unlocated_light_share:
            weak.update(crossings[k])
    # (the crossings counted: one per path of a pulsed source, its lines in one pulse)
    pulsed = [members[0] for members in crossings.values() if members[0] not in weak and contribs[members[0]].pulse is not None]
    if not pulsed:
        return PulsesPerSite(0.0, 0.0, continuous=True)
    missing = tuple(sorted({contribs[i].hit.source_id for i in pulsed if total_pulses(contribs[i].pulse) is None}))
    stated = [n for n in (total_pulses(contribs[i].pulse) for i in pulsed) if n is not None]
    low = min(stated) if stated else 0.0
    high = 0.0
    for n in stated:
        high += n
    unbounded = None
    if open_above:
        high, unbounded = math.inf, "light without an upper bound on this surface may bring more"
    elif _left_out_share(contribs, classes, qs) > tol.unlocated_light_share:
        high = math.inf
        unbounded = (f"light left out as negligible, which may carry more than {PERCENT * tol.unlocated_light_share:g} % "
                     "of the most light here, may bring more")
    return PulsesPerSite(low, high, missing, crossings=len(pulsed), unbounded=unbounded)


def _polarization_condition(rating: PowerRating, checked: _Checked, tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[RatingCondition, bool]:
    """The ``polarization`` record of ``_polarization``'s result, and whether only the light left out as negligible leaves
    it not verifiable (a deeper pass may decide it)."""
    rated = rating.polarization or rating.polarization_condition
    if rated is None:
        return RatingCondition(condition="polarization", state=NOT_VERIFIABLE, reason=RATING_INFORMATION_MISSING,
                               message="polarization: the rating states no test polarization, so its relation to the light "
                                       "here is not established"), False
    if checked.mismatches:
        return RatingCondition(condition="polarization", state=DIFFERS, effect=NO_RELATIONSHIP, rated=rated,
                               tolerance=tol.polarization_fraction, message="; ".join(checked.mismatches)), False
    texts = checked.unverified + checked.faint
    if texts:
        reasons = checked.reasons + tuple(LIGHT_STATE_UNKNOWN for _ in checked.faint)
        return RatingCondition(condition="polarization", state=NOT_VERIFIABLE, reason=reasons[0], rated=rated,  # type: ignore[arg-type]
                               tolerance=tol.polarization_fraction, message="; ".join(texts)), not checked.unverified
    return RatingCondition(condition="polarization", state=VERIFIED, rated=rated, tolerance=tol.polarization_fraction,
                           message="; ".join(checked.notes)), False


#: The note of a ``W/cm`` comparison whose exposure is the range of the two readings of an unstated diameter definition.
LINEAR_RANGE_NOTE = ("the rating does not state its beam diameter definition: P / d is the range from the 1/e^2 diameter 2 w "
                     "(lower end) to the FWHM diameter w sqrt(2 ln 2) (upper end) of the narrower beam axis")


#: The notes of a ``W/cm`` comparison whose rating states its diameter definition.
FWHM_NOTE = "P / d uses the FWHM diameter w sqrt(2 ln 2) of the narrower beam axis"
E2_NOTE = "P / d uses the 1/e^2 diameter 2 w of the narrower beam axis"


def _notes(rating: PowerRating) -> list[str]:
    if rating.unit != "W/cm":
        return []
    if rating.beam_diameter_definition == "FWHM":
        return [FWHM_NOTE]
    if rating.beam_diameter_definition == "1/E2":
        return [E2_NOTE]
    return [LINEAR_RANGE_NOTE]


def _diameter_range(rating: PowerRating) -> bool:
    """A ``W/cm`` rating that does not state its diameter definition: its exposure is a range over 1/e^2 and FWHM."""
    return rating.unit == "W/cm" and rating.beam_diameter_definition not in ("1/E2", "FWHM")


def _exposure_bounds(rating: PowerRating, contribs: list[_HitValues], classes: list[_Class],
                     open_above: bool) -> tuple[float, float | None]:
    """(lower, upper or ``None``) of the surface's exposure in the rating's unit; ``W/cm`` without a stated diameter
    definition: from the 1/e^2 reading's lower bound to the FWHM reading's upper bound."""
    if not _diameter_range(rating):
        return _bounds(contribs, classes, _quantity(rating), open_above)
    lower_e2, upper_e2 = _bounds(contribs, classes, _linear("1/E2"), open_above)
    lower_fwhm, upper_fwhm = _bounds(contribs, classes, _linear("FWHM"), open_above)
    upper = None if upper_e2 is None or upper_fwhm is None else max(upper_e2, upper_fwhm)
    return min(lower_e2, lower_fwhm), upper


def _bound_scale(rating: PowerRating, contribs: list[_HitValues], classes: list[_Class], open_above: bool, lower: float,
                 upper: float | None, applicability: str, lowest: float | None, highest: float | None) -> float | None:
    """The factor the left-out light's amplitudes must shrink by for the exposure's bounds to decide the comparison in a
    direction its applicability allows against the thresholds ``lowest`` / ``highest`` (``_faint_scale``; ``None`` when
    that cannot help)."""
    two_sided = applicability in (AS_RATED, NOT_VERIFIED)
    if two_sided and not _diameter_range(rating):
        return _faint_scale(contribs, classes, _quantity(rating), lowest, lower, upper, open_above, highest)
    sides = (("within", _linear("FWHM")), ("exceeds", _linear("1/E2"))) if _diameter_range(rating) else \
        (("within", _quantity(rating)), ("exceeds", _quantity(rating)))
    wanted = {"within", "exceeds"} if two_sided else {"within"} if applicability == AT_LEAST else {"exceeds"}
    found: list[float] = []
    for side, q in sides:
        if side not in wanted:
            continue
        scale = _faint_scale(contribs, classes, q, lowest, lower, upper, open_above, highest)
        if scale is None:
            continue
        bare_lower, bare_upper = _bounds(contribs, _without_faint(classes), q, False)
        if (side == "within" and lowest is not None and bare_upper is not None and bare_upper <= lowest) or (
                side == "exceeds" and highest is not None and bare_lower > highest):
            found.append(scale)
    return min(found) if found else None


#: The note of a comparison with a typical value (``rated_threshold``).
TYPICAL_NOTE = ("the rating's value is a typical value (status TYPICAL), not a value each part meets: it bounds the damage "
                "threshold of no particular part, so the comparison decides neither within nor exceeds")


def _reading_notes(rating: PowerRating) -> list[str]:
    """The note of a typical value (``rated_threshold``)."""
    return [TYPICAL_NOTE] if rating.status == "TYPICAL" else []


def _minor_note(rating: PowerRating, contribs: list[_HitValues], minor: frozenset[int], share: float,
                surface_id: str, tol: RatingTolerances = DEFAULT_TOLERANCES) -> list[str]:
    """The note naming followed light of lines the rating is not for that the conditions leave out (``_minor_lines``)."""
    if not minor:
        return []
    names = sorted({f"{contribs[i].hit.source_id} at {contribs[i].hit.wavelength_nm:g} nm" for i in minor})
    return [f"light of {', '.join(names[:3])}{', ...' if len(names) > 3 else ''}, which the rating is not for, carries at "
            f"most {PERCENT * share:.3g} % of the most light on {surface_id} (tolerance "
            f"{PERCENT * tol.unlocated_light_share:g} %): it counts in the exposure, not in the rating's conditions"]


# ------------------------------------------------------------------------------------ what a reported number is

#: The part of the light a surface's followed beams leave out that each cause code names (``OmittedPart``; a code absent
#: here: light not followed).
_PART_OF_CODE = {
    ErrorCode.MAX_PATHS.value: OmittedPart.BUDGET,
    ErrorCode.MAX_INTERACTIONS.value: OmittedPart.BUDGET,
    ErrorCode.OUT_OF_MODEL_SCOPE.value: OmittedPart.OUT_OF_MODEL,
    ErrorCode.ZERO_POWER.value: OmittedPart.BELOW_POWER_FLOOR,
}


def _omissions(codes) -> tuple[Omitted, ...]:
    """One omission per cause code, in order (``_PART_OF_CODE``)."""
    return tuple(Omitted(part=_PART_OF_CODE.get(code, OmittedPart.NOT_FOLLOWED), code=code) for code in dict.fromkeys(codes))


def _exposure_basis(causes: tuple[Omitted, ...], value: float, lower: float, upper: float | None) -> ValueBasis | None:
    """The entry of a comparison's ``exposure_value`` that is the followed beams' exposure while other light may reach the
    surface (``causes``: what that light is, ``build_exposure_report``). Light left out as negligible alone is the stated
    truncation of the value: the model's value, with the most that light can move the exposure away from it (the wider side
    of the proven interval ``[lower, upper]``). Any other light leaves the number an estimate: it is coherent with the
    beams and may raise or lower the exposure. ``None``: nothing names why the value is not exact (no number is reported)."""
    if not causes:
        return None
    if all(cause.accounting for cause in causes):
        most = None if upper is None else max(upper - value, value - lower, 0.0)
        return ValueBasis(kind=ValueKind.MODEL, omitted=tuple(cause.model_copy(update={"upper": most}) for cause in causes))
    return ValueBasis(kind=ValueKind.ESTIMATE, omitted=causes)


#: The ``ErrorCode`` of the issue that states a condition's finding (``_condition_facts``).
_NOT_VERIFIED_CODE = ErrorCode.POWER_RATING_CONDITIONS_NOT_VERIFIED.value
_MISMATCH_CODE = ErrorCode.POWER_RATING_CONDITIONS_MISMATCH.value


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _condition_facts(rating: PowerRating,
                     conditions: tuple[RatingCondition, ...], tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[tuple[Assumed, ...], tuple[ValidityCheck, ...]]:
    """What the ratio exposure / rated value takes for granted about a rating's test (``RatingComparison.utilization``):

    * the test conditions the rating does not state (``NOT_VERIFIABLE`` / ``RATING_INFORMATION_MISSING``) are taken as the
      light's own -- one ``UNSTATED_CONDITION_AS_DESIGN`` assumption naming them; a value of unstated definition is read
      as a peak value (``value`` ``PEAK``: the Engine's exposure is a peak value; read as ``TOTAL_PER_1E2_AREA`` the ratio
      to the threshold is half), and a typical value as the part's own (``value`` ``TYPICAL``);
    * every other condition that is not verified is a validity check of the rated value for this light: ``within`` false
      where the light is known to differ (``observed`` the light's value furthest from the rated one, the range the
      rating's with its tolerance), ``within`` null where it cannot be checked; ``setting`` names the run's setting that
      holds the condition's window;
    * a condition verified within a window the caller set (not the Engine default) is a check with ``within`` true and
      that setting: the match is stated with the window it was made in."""
    unstated = tuple(c.condition for c in conditions if c.state == NOT_VERIFIABLE and c.reason == RATING_INFORMATION_MISSING
                     and c.condition != "value_definition")
    assumed: list[Assumed] = []
    if unstated:
        assumed.append(Assumed(input=AssumedInput.UNSTATED_CONDITION_AS_DESIGN, at=unstated, code=_NOT_VERIFIED_CODE))
    if any(c.condition == "value_definition" and c.state == NOT_VERIFIABLE for c in conditions):
        assumed.append(Assumed(input=AssumedInput.UNSTATED_CONDITION_AS_DESIGN, at=("value_definition",), value="PEAK",
                               code=_NOT_VERIFIED_CODE))
    if rating.status == "TYPICAL":
        assumed.append(Assumed(input=AssumedInput.UNSTATED_CONDITION_AS_DESIGN, at=("rating_status",), value="TYPICAL"))
    checks: list[ValidityCheck] = []
    for c in conditions:
        differs = c.state == DIFFERS
        window = tol.setting_of(c.condition) if c.tolerance is not None else None
        matched = c.state == VERIFIED and window is not None and window[1] != window[2]
        if not differs and not matched and not (c.state == NOT_VERIFIABLE and c.reason != RATING_INFORMATION_MISSING):
            continue
        setting = None if window is None else window[0]
        if _number(c.rated) is None and c.rated_low is None and c.tolerance is not None:
            # (a condition without a number of its own -- a polarization state: the share outside it against its largest)
            checks.append(ValidityCheck(parameter=c.condition, maximum=c.tolerance, within=True if matched else False if differs else None,
                                        setting=setting, code=None if matched else (_MISMATCH_CODE if c.effect == NO_RELATIONSHIP else None)
                                        if differs else _NOT_VERIFIED_CODE))
            continue
        rated, low, high = _number(c.rated), c.rated_low, c.rated_high
        if rated is not None and c.tolerance is not None and low is None and high is None:
            low, high = rated * (1.0 - c.tolerance), rated * (1.0 + c.tolerance)
        observed = _number(c.design)
        if observed is None:
            ends = [v for v in (c.design_low, c.design_high) if v is not None]
            centre = rated if rated is not None else (None if low is None or high is None else 0.5 * (low + high))
            observed = None if not ends else ends[-1] if centre is None else max(ends, key=lambda v: abs(v - centre))
        code = None if matched else (_MISMATCH_CODE if c.effect == NO_RELATIONSHIP else None) if differs else _NOT_VERIFIED_CODE
        checks.append(ValidityCheck(parameter=c.condition, unit=c.unit or None, minimum=low, maximum=high, observed=observed,
                                    within=True if matched else False if differs else None, setting=setting, code=code))
    return tuple(assumed), tuple(checks)


def _ratio_basis(exposure: ValueBasis | None, rating: PowerRating, conditions: tuple[RatingCondition, ...],
                 scale: float, tol: RatingTolerances = DEFAULT_TOLERANCES) -> ValueBasis | None:
    """The entry of exposure / rated value (``RatingComparison.utilization``) from the exposure's own entry (``None``: the
    exposure is exact) and the rating's conditions (``_condition_facts``); each omission's ``upper`` in the ratio's unit
    (``scale``: 1 / the rated value). The model's value only for a rating whose every stated condition is verified and an
    exposure that is; ``None``: the ratio is the model's value and nothing qualifies it."""
    assumed, checks = _condition_facts(rating, conditions, tol=tol)
    omitted = () if exposure is None else tuple(
        o if o.upper is None else o.model_copy(update={"upper": o.upper * scale}) for o in exposure.omitted)
    if not (omitted or assumed or checks):
        return None
    plain = not assumed and all(v.within is True for v in checks) and (exposure is None or exposure.kind is ValueKind.MODEL)
    return ValueBasis(kind=ValueKind.MODEL if plain else ValueKind.ESTIMATE, omitted=omitted, assumed=assumed, validity=checks)


def _has_quantity(rating: PowerRating, contribs: list[_HitValues]) -> bool:
    """Whether the rating's quantity has a value for every contribution on the surface: the exposure in the rating's unit
    can then be stated also for light the rating is not for (another wavelength, pulse duration or repetition rate),
    beside the conditions that differ. It decides nothing there: the comparison stays ``CONDITIONS_MISMATCH``."""
    quantities = (_linear("1/E2"), _linear("FWHM")) if _diameter_range(rating) else (_quantity(rating),)
    return bool(contribs) and all(q(v) is not None for q in quantities for v in contribs)


def _compare(index: int, rating: PowerRating, contribs: list[_HitValues], classes: list[_Class], open_above: bool,
             surface_id: str, unknown_light: str | None, cfg: NumericalConfig,
             left_out: tuple[Light, ...] = (), stored: tuple[Light, ...] = (), traced: float | None = None,
             causes: tuple[Omitted, ...] = (), tol: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[RatingComparison, float | None]:
    """One rating against a surface's light ``contribs`` (traced, untraced, beside) in its coherence ``classes`` (with the
    left-out light); ``open_above``: power of unknown fate may reach the surface (no upper bound); ``unknown_light``: what
    that light is (its polarization is not known); ``left_out``: on a surface no followed light reaches, the lines of the
    light left out as negligible (their regime, wavelength and pulse train are the light's, ``scalar_conditions``);
    ``stored``: the lines whose light may be stored between surfaces facing each other (a run open above); ``traced``: the
    exposure of the followed beams alone in the rating's unit (``None``: no single value) and ``causes``: the light that
    may reach the surface besides them -- where the exposure is a range, ``exposure_value`` is that number with its entry
    (``_exposure_basis``), and ``utilization`` its ratio to the rated value with what the ratio takes for granted
    (``_ratio_basis``); neither enters a status. Every condition
    is recorded (``RatingComparison.conditions``) and the rating's applicability decides what the exposure's bounds may
    settle (``rating_conditions``). Also the factor the left-out light's amplitudes must shrink by when they alone keep the
    comparison from being decided (``None`` otherwise)."""
    polarization = _polarization(rating, contribs, classes, surface_id, unknown_light, tol=tol)
    # (followed light of lines the rating is not for that carries at most the tolerance of the most light: in the exposure,
    # not in the conditions -- where nothing bounds the light that may be stored, its share is not bounded either)
    minor, minor_share = _minor_lines(rating, contribs, classes, tol=tol) if not open_above else (frozenset(), 0.0)
    if not open_above:
        faint_unverified, faint_scale = _faint_mismatch(rating, contribs, classes, surface_id, minor, tol=tol)
    elif not contribs:
        # (only the left-out light reaches the surface and nothing bounds its share there: a line the rating is not for
        # leaves each condition it differs in not verifiable)
        faint_unverified, faint_scale = _unbounded_mismatch(rating, (*left_out, *stored), surface_id, tol=tol), None
    else:
        faint_unverified, faint_scale = {}, None
    lights = ([Light(v.hit.source_id, v.hit.wavelength_nm, v.pulse) for i, v in enumerate(contribs) if i not in minor]
              or list(left_out))
    scalar = scalar_conditions(rating, lights, faint_unverified, tolerances=tol)
    pulses = _pulses_per_site(rating, contribs, classes, minor, left_out, open_above, tol=tol)
    design_pulses = None if pulses is None or pulses.missing else pulses.high  # (continuous light: 0 pulses per site)
    beam, beam_scale, beam_deeper = _beam_condition(rating, contribs, classes, surface_id, cfg, design_pulses, tol=tol)
    count = pulse_count_condition(rating, pulses)
    polar, polar_deeper = _polarization_condition(rating, polarization, tol=tol)
    reading = value_definition_condition(rating)
    conditions = (*scalar, beam, count, polar, reading)
    applicability = rating_applicability(conditions)
    base = dict(rating_index=index, regime=rating.regime, unit=rating.unit, rating_value=rating.value,
                rating_wavelength_nm=rating.wavelength_nm, rating_status=rating.status, basis=rating.basis, origin=rating.origin,
                notes=tuple(_notes(rating) + list(polarization.notes)
                            + _minor_note(rating, contribs, minor, minor_share, surface_id, tol=tol)),
                mismatches=tuple(c.message for c in conditions if c.state == DIFFERS),
                unverified_conditions=tuple(c.message for c in conditions if c.state == NOT_VERIFIABLE),
                applicability=applicability, conditions=conditions, conditions_match=applicability == AS_RATED)
    lowest, highest = rated_threshold(rating.value, rating.unit, rating.status, rating.value_definition)
    base.update(rated_threshold_low=lowest, rated_threshold_high=highest)
    if any(c.state == DIFFERS for c in scalar) and not _has_quantity(rating, contribs):
        # (the rating is for other light, and its quantity does not exist for the light here -- a fluence of continuous
        # light, light without a known size: no exposure in its unit)
        return RatingComparison(**base, status="CONDITIONS_MISMATCH"), None  # type: ignore[arg-type]
    lower, upper = _exposure_bounds(rating, contribs, classes, open_above)
    exact = upper is not None and lower == upper
    # (a range: the followed beams' exposure is reported beside its bounds, with what it does not count -- where it lies
    # within them: light the trace does not follow that outweighs the beams leaves no such number)
    if traced is not None and (traced < lower or (upper is not None and traced > upper)):
        traced = None
    value_basis = None if exact or traced is None else _exposure_basis(causes, traced, lower, upper)
    bases: dict[str, ValueBasis] = {} if value_basis is None else {"exposure_value": value_basis}
    reported = upper if exact else traced if value_basis is not None else None
    exposure = dict(exposure_value=reported, exposure_lower_bound=None if exact or lower <= 0.0 else lower,
                    exposure_upper_bound=None if exact else upper)
    # (exposure / rated value: decided, only for a comparison that applies and a value that establishes a threshold)
    utilization = None if lowest is None or applicability not in APPLYING else (
        upper / rating.value if exact else lower / rating.value if lower > rating.value else None)  # type: ignore[operator]
    if utilization is not None and exact:
        # (decided on an exact exposure: the model's value; a match made in a window the caller set is stated with it)
        windows = tuple(v for v in _condition_facts(rating, conditions, tol=tol)[1] if v.within is True)
        if windows:
            bases["utilization"] = ValueBasis(kind=ValueKind.MODEL, validity=windows)
    elif utilization is not None:
        # (a decided exceedance: the lower bound's ratio; what the lower bound does not count can only add)
        if causes:
            bases["utilization"] = ValueBasis(kind=ValueKind.LOWER_BOUND, omitted=causes)
    elif utilization is None and reported is not None and rating.value > 0.0:
        # (not decided: the ratio of the reported exposure to the rated value as stated, with what it takes for granted)
        ratio_basis = _ratio_basis(value_basis, rating, conditions, 1.0 / rating.value, tol=tol)
        if ratio_basis is not None:
            utilization = reported / rating.value
            bases["utilization"] = ratio_basis
    status = comparison_status(applicability, lower, upper, lowest, highest)
    base.update(notes=(*base["notes"], *_reading_notes(rating)))  # type: ignore[misc]
    scale: float | None = None
    # (how deep the run follows the left-out light does not depend on the rating's pulse count, its protocol or the totals
    # the sources state: the decision a deeper pass could reach is judged as if the rating stated no count; a value of
    # unstated definition is judged against the value (within) and twice it (exceeds), the depths of the two definitions'
    # thresholds -- trace depths only: the comparison itself establishes no threshold)
    probe_lowest, probe_highest = (lowest, highest) if reading.state != NOT_VERIFIABLE else (
        rating.value, PEAK_PER_TOTAL_OVER_E2_AREA * rating.value)
    probe_beam, probe_scale, probe_deeper = beam, beam_scale, beam_deeper
    if rating.regime == "PULSED" and rating.pulse_count == 1 and rating.beam_diameter_mm is not None and design_pulses != 1:
        # (the metallic exception keeps a direction for one pulse per site: judged as if a single shot met it)
        probe_beam, probe_scale, probe_deeper = _beam_condition(rating, contribs, classes, surface_id, cfg, 1.0, tol=tol)
    # (and as if the rating stated the test conditions it does not -- its beam diameter, rate, polarization --, as at
    # e344aab7: these are trace depths, not conditions)
    if rating.beam_diameter_mm is None:
        probe_beam, probe_deeper = _NOT_STATED_BEAM, False
    probe_scalar = tuple(_NOT_STATED_RATE if c.condition == "repetition_rate" and rating.repetition_rate_hz is None else c
                         for c in scalar)
    probe_polar = _NOT_STATED_POLARIZATION if rating.polarization is None and rating.polarization_condition is None else polar
    probe_conditions = (*probe_scalar, probe_beam, _NO_COUNT, probe_polar)
    # (the conditions a deeper pass may check: the ones only light left out as negligible leaves not verifiable)
    deeper = [c for c in probe_scalar if c.state == NOT_VERIFIABLE] + ([probe_beam] if probe_deeper else []) + (
        [polar] if polar_deeper else [])
    probe_applicability = rating_applicability(probe_conditions)
    probe_status = comparison_status(probe_applicability, lower, upper, probe_lowest, probe_highest)
    fixable = all(c.state != NOT_VERIFIABLE or any(c is d for d in deeper) for c in probe_conditions)
    if probe_applicability != NOT_APPLICABLE and fixable and probe_status in ("UNKNOWN", "CONDITIONS_NOT_VERIFIED"):
        # where the left-out light alone keeps it from being decided (its bounds, or a condition it alone leaves not
        # verifiable): how much smaller its amplitudes must be
        scales = [s for s in (_bound_scale(rating, contribs, classes, open_above, lower, upper, probe_applicability,
                                           probe_lowest, probe_highest),
                              polarization.scale if polar_deeper else None, faint_scale, probe_scale) if s is not None]
        scale = min(scales) if scales else None
    return RatingComparison(**base, **exposure, utilization=utilization, status=status, value_basis=bases), scale  # type: ignore[arg-type]


#: Conditions that neither verify nor block (``_compare``'s deeper-pass judgement only; never in a result).
_NO_COUNT = RatingCondition(condition="pulse_count", state=NOT_STATED, message="pulse count: not stated by the rating")
_NOT_STATED_BEAM = RatingCondition(condition="beam_diameter", state=NOT_STATED, message="beam diameter: not stated by the rating")
_NOT_STATED_RATE = RatingCondition(condition="repetition_rate", state=NOT_STATED, message="repetition rate: not stated by the rating")
_NOT_STATED_POLARIZATION = RatingCondition(condition="polarization", state=NOT_STATED, message="polarization: not stated by the rating")


def exact_exposure(c: RatingComparison) -> float | None:
    """A comparison's exposure where it is known exactly: ``exposure_value`` without an entry. With an entry the number is
    the followed beams' exposure reported beside the bounds (``_exposure_basis``); every status, bound and decision reads
    this, never that number."""
    return None if c.value_basis and "exposure_value" in c.value_basis else c.exposure_value


def _upper_exposure(c: RatingComparison) -> float | None:
    exact = exact_exposure(c)
    return exact if exact is not None else c.exposure_upper_bound


def _met(c: RatingComparison) -> bool:
    """The exposure's upper bound is within the rating (up to the lowest threshold its value establishes,
    ``rated_threshold``)."""
    upper = _upper_exposure(c)
    lowest = c.rated_threshold_low
    return upper is not None and lowest is not None and upper <= lowest


def _surface_status(comparisons: list[RatingComparison]) -> str:
    """A surface's status from its comparisons (``SurfaceExposure.rating_status``): exceeded when a rating that applies as
    stated or with a threshold at most the rated value is exceeded; within when a rating applies as stated or with a
    threshold at least the rated value and the exposure's upper bound is within it and within every rating not verified
    (it may apply as stated and be the stricter one), and the surface has an upper bound at all (``_surface_bounds``: a
    rating with a threshold at most its value, or one not verified, leaves it none unless a rating of the same unit that
    applies as stated or with a threshold at least its value has a lowest threshold no larger than its highest under every
    reading, ``_at_most_open``). A typical value bounds no threshold (``rated_threshold``): it neither decides nor holds
    back the other ratings, and a surface it alone applies to is ``UNKNOWN`` (``CONDITIONS_NOT_VERIFIED`` when none of its
    conditions can be checked)."""
    if not comparisons:
        return "RATING_MISSING"
    typical = [c for c in comparisons if _typical(c)]
    comparisons = [c for c in comparisons if not _typical(c)]
    if not comparisons:
        if any(c.applicability in APPLYING for c in typical):
            return "UNKNOWN"
        return "CONDITIONS_NOT_VERIFIED" if any(c.applicability == NOT_VERIFIED for c in typical) else "CONDITIONS_MISMATCH"
    below = [c for c in comparisons if c.applicability in (AS_RATED, AT_MOST)]
    above = [c for c in comparisons if c.applicability in (AS_RATED, AT_LEAST)]
    unverified = [c for c in comparisons if c.applicability == NOT_VERIFIED]
    if any(c.status == "EXCEEDS_RATING" for c in below):
        return "EXCEEDS_RATING"
    if above and all(_met(c) for c in above + unverified) and _surface_bounds(tuple(comparisons))[1] is not None:
        return "WITHIN_RATING"
    if unverified:
        return "CONDITIONS_NOT_VERIFIED"
    if below or above or any(c.applicability != NOT_APPLICABLE for c in typical):
        return "UNKNOWN"
    return "CONDITIONS_MISMATCH"


def _typical(c: RatingComparison) -> bool:
    """A comparison with a typical value: it bounds no threshold (``rated_threshold``)."""
    return c.rating_status == "TYPICAL"


def rated_surfaces(project: ProjectSnapshot) -> frozenset[str]:
    """The surfaces (``<component>/<surface>``) at least one power rating applies to."""
    return frozenset(sid for sid, ratings in _ratings_by_surface(project).items() if ratings)


def _ratings_by_surface(project: ProjectSnapshot) -> dict[str, list[tuple[int, PowerRating]]]:
    out: dict[str, list[tuple[int, PowerRating]]] = {}
    for comp in project.components:
        for surface in comp.surfaces:
            applies = [(i, r) for i, r in enumerate(comp.power_ratings) if not r.surface_ids or surface.surface_id in r.surface_ids]
            # a facet surface is traced as its planar patches: a rating of the surface applies to each of them
            ids = facet_patch_ids(comp.instance_id, surface) if isinstance(surface, MeshSurface) else (f"{comp.instance_id}/{surface.surface_id}",)
            for sid in ids:
                out[sid] = list(applies)
    return out


def _codes(*groups) -> tuple[str, ...]:
    return tuple(dict.fromkeys(code for group in groups for code in group))


def _coherent_groups(contribs: list[_HitValues], classes: list[_Class]) -> tuple[CoherentExposureGroup, ...]:
    out: list[CoherentExposureGroup] = []
    for cls in classes:
        if len(cls.members) < 2 or not cls.interferes:
            continue
        first = contribs[cls.members[0]]
        assert first.raw is not None
        sides = {_side(contribs[i].raw) for i in cls.members}  # type: ignore[arg-type]
        incidence = "BOTH_SIDES" if len(sides) > 1 else "ALONG_NORMAL" if 1 in sides else "AGAINST_NORMAL"
        out.append(CoherentExposureGroup(
            coherence_group=first.raw.coherence_group, wavelength_nm=first.hit.wavelength_nm, incidence=incidence,
            members=tuple(contribs[i].ref for i in cls.members), largest_coherence_factor=max(g.gamma for g in cls.gamma.values())))
    return tuple(out)


def _source_groups(project: ProjectSnapshot) -> dict[str, str]:
    return {c.source.source_id: c.source.coherence_group or f"source:{c.source.source_id}" for c in project.components
            if c.source is not None}


def _component_groups(project: ProjectSnapshot) -> dict[str, tuple[str, ...]]:
    """The coherence group of the second polarization component of each partially polarized source (its light left out or
    of unknown fate may belong to either component: ``optics/polarization/partial.py``)."""
    groups = _source_groups(project)
    return {c.source.source_id: (component_group(groups[c.source.source_id], 1),) for c in project.components
            if c.source is not None and partially_polarized(c.source.degree_of_polarization) is not None}


def _unknown_lines(project: ProjectSnapshot, power_budget: PowerBudget | None, unfollowed: dict[tuple[str, int], dict[str, float]],
                   cfg: NumericalConfig) -> list[LineTag]:
    """(coherence group, wavelength) of every source line part of whose light has an unknown fate (the ledger, or untraced
    light its probe does not follow): that light may be coherent with the traced beams of its line anywhere."""
    if power_budget is None:
        return []
    groups = _source_groups(project)
    also = _component_groups(project)
    out: list[LineTag] = []
    for line in power_budget.lines:
        if not line_fate([line], cfg).complete or (line.source_id, line.spectral_index) in unfollowed:
            out.append((groups.get(line.source_id, f"source:{line.source_id}"), line.wavelength_nm))
            out.extend((group, line.wavelength_nm) for group in also.get(line.source_id, ()))
    return out


def _coherence_length(project: ProjectSnapshot, source_id: str, wavelength_nm: float) -> float | None:
    """The coherence length a source declares for one of its lines (``scene/lower.py``: a length, or one from a linewidth)."""
    for c in project.components:
        if c.source is not None and c.source.source_id == source_id:
            if c.source.coherence_length_mm is not None:
                return c.source.coherence_length_mm
            if c.source.linewidth_nm is not None:
                return coherence_length_from_linewidth_mm(wavelength_nm, c.source.linewidth_nm)
    return None


def _unfinished(power_budget: PowerBudget | None, unfollowed: dict[LineKey, dict[str, float]],
                unfollowed_root: dict[LineKey, dict[str, float]] | None = None) -> tuple[set[LineKey], dict[LineKey, float], dict[LineKey, float]]:
    """The source lines part of whose light met the per-path interaction limit (the ledger's or a probe's: it may be stored
    in a cavity), and per line the power of unknown fate of every other cause (the trace's or the probes' budget, a path
    ended out of model scope, a line not launched, the light a tabulated thin lens does not deliver): known power -- with
    the sum of the square roots of its pieces' powers (each ledger entry, each probe flow)."""
    stored: set[LineKey] = set()
    extra: dict[LineKey, float] = {}
    roots: dict[LineKey, float] = {}
    for line in (power_budget.lines if power_budget is not None else ()):
        key = (line.source_id, line.spectral_index)
        for kind in UNKNOWN_FATE_SINKS:
            power = line.sinks.get(kind, 0.0)
            if power <= 0.0:
                continue
            if kind == SINK_INTERACTION_LIMIT:
                stored.add(key)
            else:
                extra[key] = extra.get(key, 0.0) + power
    for entry in (power_budget.entries if power_budget is not None else ()):
        if entry.kind in UNKNOWN_FATE_SINKS and entry.kind != SINK_INTERACTION_LIMIT and entry.power_W > 0.0:
            key = (entry.source_id, entry.spectral_index)
            roots[key] = roots.get(key, 0.0) + math.sqrt(entry.power_W)
    for key, kinds in unfollowed.items():
        for kind, power in kinds.items():
            if kind in STORED_KINDS:
                stored.add(key)
            elif power > 0.0:
                extra[key] = extra.get(key, 0.0) + power
                root = (unfollowed_root or {}).get(key, {}).get(kind)
                roots[key] = roots.get(key, 0.0) + (math.sqrt(power) if root is None else root)
    return stored, extra, roots


def _faint_lines(project: ProjectSnapshot, power_budget: PowerBudget | None, untraced: SurfaceReach | None,
                 extra: dict[LineKey, float] | None = None, extra_root: dict[LineKey, float] | None = None) -> list[_FaintLine]:
    """Every source line with light left out as negligible (``SurfaceReach.left_out_W``) or of unknown fate of known power
    (``extra``, ``_unfinished``): their power together, and the sum of the square roots of their pieces' powers."""
    if power_budget is None or untraced is None:
        return []
    groups = _source_groups(project)
    pulses = {c.source.source_id: c.source.pulse for c in project.components if c.source is not None}
    wavelengths = {(line.source_id, line.spectral_index): line.wavelength_nm for line in power_budget.lines}
    powers = dict(untraced.left_out_W)
    roots = {key: untraced.left_out_root.get(key, math.sqrt(power)) for key, power in untraced.left_out_W.items()}
    for key, power in (extra or {}).items():
        powers[key] = powers.get(key, 0.0) + power
        roots[key] = roots.get(key, 0.0) + (extra_root or {}).get(key, math.sqrt(power))
    also = _component_groups(project)
    return [_FaintLine((source, index), groups.get(source, f"source:{source}"), wavelengths[(source, index)], power,
                       pulses.get(source), spectral_factor(wavelengths[(source, index)],
                                                           _coherence_length(project, source, wavelengths[(source, index)]),
                                                           pulses.get(source)), roots[(source, index)], also.get(source, ()))
            for (source, index), power in sorted(powers.items()) if (source, index) in wavelengths and power > 0.0]


@dataclass(frozen=True)
class ExposureOutcome:
    """``build_exposure_report``'s result: the report, its ``POWER_RATING_EXCEEDED`` notices, per element the
    response-model surfaces (tabulated thin lenses) not validated for the exposure of its rated surfaces behind them, and
    per source line the level a deeper probe pass should follow its light down to, where the light left out as negligible
    alone keeps a comparison from being decided (``deeper_levels``; ``validation/pipeline.py`` runs that pass)."""

    report: ExposureReport
    issues: list[ValidationIssue]
    element_scope: dict[str, tuple[str, ...]]
    deeper_levels: dict[tuple[str, int], float] = field(default_factory=dict)
    # per element the surfaces behind which the power of light reaching its rated surfaces is out of model scope
    # (``SurfaceHit.power_scope``): its ``power_rating_utilization`` is out of model scope
    power_scope: dict[str, tuple[str, ...]] = field(default_factory=dict)


def build_exposure_report(hits: list[SurfaceHit], project: ProjectSnapshot, cfg: NumericalConfig,
                          power_budget: PowerBudget | None, untraced: SurfaceReach | None = None,
                          metric_scope: dict[str, frozenset[str]] | None = None,
                          media_index: Callable[[str, float], float] | None = None,
                          gains: Mapping[float, ReturnGains] | None = None,
                          room: Callable[[str, object], tuple[float | None, float] | None] | None = None,
                          converters: frozenset[str] = frozenset()) -> ExposureOutcome:
    """The run's ``ExposureReport`` (module docstring). ``untraced``: the probe traces' hits of the light the trace does not
    follow and the light left out as negligible (a run with power ratings; ``None``: not followed); ``metric_scope``:
    surface id -> the target metrics a response model there was validated for; ``media_index``: (surface id, vacuum
    wavelength nm) -> the larger refractive index of the surface's two media (1 when not given); ``gains``: per vacuum
    wavelength the factor of the returns between surfaces facing each other, which may bring left-out light to any
    surface (``validation/resonance.py``; ``None``: none); ``room``: (surface id, point) -> the surface's radius (``None``
    on a plane) and the distance from the point to its clear aperture's edge, where the window bound applies
    (``validation/window_bound.py``: ``surface_room``; ``None``: it applies nowhere); ``converters``: the crystal surfaces
    whose second-harmonic generation is out of model scope (the reason code of the light behind them)."""

    scope = metric_scope or {}
    # (the run's windows for matching a rating's test conditions, and the share unlocated light may carry: its settings)
    tol = RatingTolerances.of(cfg)
    index_of = media_index or (lambda _surface, _wavelength: 1.0)
    pulses = {c.source.source_id: c.source.pulse for c in project.components if c.source is not None}
    # (sources of continuous spectrum: their lines count as fully coherent with their partners, ``_wave``)
    continuous = frozenset(c.source.source_id for c in project.components if c.source is not None and c.source.spectral_density is not None)
    fate = line_fate(power_budget.lines, cfg) if power_budget is not None else None
    trace_complete = fate is None or fate.complete
    unknown_W = 0.0 if fate is None else fate.untraced_W
    incomplete_codes = (ErrorCode.INCOMPLETE_TRACE.value, *(fate.cause_codes() if fate is not None else ()))
    unfollowed = untraced.unfollowed if untraced is not None else {}
    # codes of untraced light of unknown fate (any line): it may reach every surface
    unfollowed_codes = tuple(dict.fromkeys(UNKNOWN_FATE_CAUSES[kind].value
                                           for kind in dict.fromkeys(kind for per in unfollowed.values() for kind in per)))
    if untraced is not None:
        # a run that follows the untraced light: light of unknown fate of known power is bounded like the light left out as
        # negligible (its diffraction limit, its returns); light at the interaction limit may be stored in a cavity, which
        # nothing bounds
        stored, extra, extra_root = _unfinished(power_budget, unfollowed, untraced.unfollowed_root)
        groups = _source_groups(project)
        also = _component_groups(project)
        unknown_lines = [(group, line.wavelength_nm)
                         for line in (power_budget.lines if power_budget is not None else ())
                         if (line.source_id, line.spectral_index) in stored
                         for group in (groups.get(line.source_id, f"source:{line.source_id}"), *also.get(line.source_id, ()))]
        # (the lines whose light may be stored: on a surface only left-out light reaches, a line the rating is not for)
        stored_lines = tuple(Light(line.source_id, line.wavelength_nm, pulses.get(line.source_id), left_out=True)
                             for line in (power_budget.lines if power_budget is not None else ())
                             if (line.source_id, line.spectral_index) in stored)
        open_run = bool(stored)
    else:
        stored_lines = ()
        extra, extra_root = {}, {}
        unknown_lines = _unknown_lines(project, power_budget, unfollowed, cfg)
        open_run = not trace_complete or bool(unfollowed_codes)
    faint_lines = _faint_lines(project, power_budget, untraced, extra, extra_root)
    scales: dict[LineKey, float] = {}

    by_surface: dict[str, list[SurfaceHit]] = {}
    owners: dict[str, str] = {}
    # (beams whose chief ray misses a rated surface while their footprint reaches it: ``beside``, traced or untraced)
    beside_by_surface: dict[str, list[tuple[str | None, SurfaceHit]]] = {}
    for h in hits:
        if h.miss_distance_mm is not None:
            beside_by_surface.setdefault(h.surface_id, []).append((None, h))
        else:
            by_surface.setdefault(h.surface_id, []).append(h)
        owners.setdefault(h.surface_id, h.owner_id)
    probe_hits: dict[str, list[tuple[str, SurfaceHit]]] = {}
    for kind, h in (untraced.hits if untraced is not None else ()):
        if h.miss_distance_mm is not None:
            beside_by_surface.setdefault(h.surface_id, []).append((kind, h))
        else:
            probe_hits.setdefault(h.surface_id, []).append((kind, h))
        owners.setdefault(h.surface_id, h.owner_id)
    ratings = _ratings_by_surface(project)
    # (the causes of light of unknown fate, any line: what a reported exposure does not count of it)
    open_codes = ((fate.cause_codes() if fate is not None else ()) if not trace_complete else ()) + unfollowed_codes
    # a rated surface no light the trace or the probes follow reaches: the left-out light alone bounds it
    reached_by_faint = {sid for sid, rs in ratings.items() if rs} if faint_lines else set()
    for sid in reached_by_faint:
        owners.setdefault(sid, sid.split("/", 1)[0])
    surfaces: list[SurfaceExposure] = []
    issues: list[ValidationIssue] = []
    # (the surfaces a closed return without a series leaves without an upper bound, and its wavelengths: one notice)
    closed_surfaces: list[tuple[str, str]] = []
    closed_at: dict[float, tuple[tuple[str, ...], ...]] = {}
    element_scope: dict[str, list[str]] = {}
    power_scope: dict[str, list[str]] = {}
    for surface_id in sorted(set(by_surface) | set(probe_hits) | set(beside_by_surface) | reached_by_faint):
        indices: dict[float, float] = {}

        def n_of(wavelength_nm: float, surface_id: str = surface_id, indices: dict[float, float] = indices) -> float:
            if wavelength_nm not in indices:
                indices[wavelength_nm] = index_of(surface_id, wavelength_nm)
            return indices[wavelength_nm]

        passes: dict[str, int] = {}
        values: list[_HitValues] = []
        for h in by_surface.get(surface_id, []):
            index = passes.get(h.path_id, 0)
            passes[h.path_id] = index + 1
            value = _hit_values(h, index, pulses.get(h.source_id), cfg, scope, converters=converters)
            if room is not None and value.located:
                value = replace(value, room=room(surface_id, h.position), n_max=n_of(h.wavelength_nm))
            values.append(value)
        values.sort(key=lambda v: (v.hit.path_id, v.hit.pass_index))
        # every untraced light the probes find on the surface bounds it, whether or not traced beams of the same line meet
        # it too (a weak traced beam does not stand for a strong ghost)
        reaching = probe_hits.get(surface_id, [])
        open_above = open_run
        probe_passes: dict[str, int] = {}
        light: list[_HitValues] = []
        for kind, h in reaching:
            index = probe_passes.get(h.path_id, 0)
            probe_passes[h.path_id] = index + 1
            light.append(_hit_values(h, index, pulses.get(h.source_id), cfg, scope, kind, UNTRACED, n_of(h.wavelength_nm),
                                     converters))
        light.sort(key=lambda v: (v.hit.path_id, v.hit.pass_index, v.hit.wavelength_nm, v.hit.incident_power_W))
        beside_passes: dict[tuple[str, bool], int] = {}
        beside: list[_HitValues] = []
        for kind, h in beside_by_surface.get(surface_id, []):
            key = (h.path_id, kind is None)
            index = beside_passes.get(key, 0)
            beside_passes[key] = index + 1
            beside.append(_hit_values(h, index, pulses.get(h.source_id), cfg, scope, kind,
                                      n_max=None if kind is None else n_of(h.wavelength_nm), converters=converters))
        beside.sort(key=lambda v: (v.hit.path_id, v.hit.pass_index, v.hit.wavelength_nm, v.hit.incident_power_W))
        contribs = values + light + beside

        def factor_of(wavelength_nm: float, surface_id: str = surface_id) -> float:
            found = gains.get(wavelength_nm) if gains else None
            return 1.0 if found is None else found.exposure_factor(surface_id)

        def crossings_of(wavelength_nm: float, surface_id: str = surface_id) -> float:
            found = gains.get(wavelength_nm) if gains else None
            return math.inf if found is None else found.crossing_field(surface_id)

        classes = _classes(contribs, cfg, unknown_lines, faint_lines, n_of, factor_of, crossings_of, continuous)
        closed = {x.wavelength_nm: gains[x.wavelength_nm].closed_cycles for x in faint_lines
                  if gains and x.wavelength_nm in gains and gains[x.wavelength_nm].closed_cycles
                  and gains[x.wavelength_nm].closed_series is None}
        # left-out light that a pair whose gain bound is 1 may return to this surface (two lossless reflectors, a coating
        # met outside its data, an uncoated face met from inside up to its critical angle): nothing bounds it -- no upper
        # bound, and its class may cancel to 0
        returns = [cls.faint.factor for cls in classes if cls.faint is not None and cls.faint.factor > 1.0]
        # (likewise light whose open paths' crossings are not known)
        stored = any(not math.isfinite(f) for f in returns) or any(
            cls.faint is not None and not math.isfinite(cls.faint.crossings) for cls in classes)
        with_returns = classes
        if stored:
            classes = [replace(cls, faint=None, open_below=True)
                       if cls.faint is not None and not (math.isfinite(cls.faint.factor) and math.isfinite(cls.faint.crossings))
                       else cls for cls in classes]
            open_above = True
        return_factor_here = max((f for f in returns if math.isfinite(f)), default=None)
        faint_peak = 0.0
        for cls in classes:
            amp = cls.amplitude(_peak)
            faint_peak += amp * amp
        withheld = _withheld(surface_id, faint_lines, gains)
        # (pairs at a gain bound of 1 alone leave the left-out light unbounded here: the same light with those pairs at
        # their centre-line incidence, an estimate reported with its entry -- in no bound, comparison or status)
        nominal = _nominal_faint(surface_id, with_returns, gains) if stored and faint_peak == 0.0 and not open_run else None
        # (the return-light analysis reached its work limit at a wavelength of that light: it bounded none of its returns)
        by_work = open_above and not open_run and bool(gains) and any(x.wavelength_nm in gains and not gains[x.wavelength_nm].screened for x in faint_lines)
        unknown_light = ("light of unknown fate" if open_run else
                         "light left out as negligible, whose returns to this surface the return-light analysis did not bound "
                         "within its work limit (RETURN_ANALYSIS_BUDGET)" if by_work else
                         "light left out as negligible, which two surfaces facing each other may hold without bound and "
                         "send on to this one") \
            if open_above else None
        if open_above and not open_run and (withheld.uses or withheld.partners) and not withheld.pairs:
            unknown_light = "light left out as negligible, met on this surface or the surface facing it at angles the coating record does not state"
        if closed:
            unknown_light = ("light left out of the trace can circulate through a multi-surface return path; "
                             "its exposure at this surface has no supported upper bound")
        traced_classes = _classes(values, cfg, unknown_lines, continuous=continuous)
        traced_range = any(cls.interferes_with(set(range(len(values)))) for cls in classes)  # (values: the first contributions)
        peak = _upper(values, traced_classes, _peak, known_only=True)
        peak_surface = _upper(values, traced_classes, _peak_surface, known_only=True)
        linear = _upper(values, traced_classes, _linear("1/E2"), known_only=True)
        pulsed = [v for v in values if v.pulsed]
        fluence = _upper(pulsed, _classes(pulsed, cfg, unknown_lines, continuous=continuous), _fluence, known_only=True) if pulsed else None
        peak_ok = all(v.hit.peak_irradiance_W_per_cm2 is not None for v in values)
        codes = _codes([v.hit.reason for v in values if v.hit.reason is not None],
                       incomplete_codes if not trace_complete else (),
                       [bounded_cause(v.kind) for v in light if v.kind is not None],
                       [ErrorCode.INCOMPLETE_ENVELOPE.value] if beside else (),
                       [v.hit.reason for v in beside if v.hit.reason is not None], unfollowed_codes,
                       [ErrorCode.COHERENT_BEAMS_BOUNDED.value] if traced_range else (),
                       [ErrorCode.FAINT_LIGHT_BOUNDED.value] if faint_peak > 0.0 else (),
                       [ErrorCode.FAINT_LIGHT_RECIRCULATES.value] if returns else (),
                       [ErrorCode.EXPOSURE_BOUND_WITHHELD.value] if closed else ())
        surface_ratings = ratings.get(surface_id, [])
        # (a surface only the light left out as negligible reaches: that light's lines, each with its pulse train)
        left_out = () if contribs else tuple(Light(x.key[0], x.wavelength_nm, x.pulse, left_out=True) for x in faint_lines)
        # (the light that may reach the surface besides its followed beams: what a reported exposure does not count)
        not_counted: list[Omitted] = []
        if faint_peak > 0.0:
            not_counted.append(Omitted(part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, code=ErrorCode.FAINT_LIGHT_BOUNDED.value))
        if stored:
            not_counted.extend(_return_omissions(withheld, closed, by_work))
        not_counted.extend(_omissions([bounded_cause(v.kind) for v in light if v.kind is not None]))
        if beside:
            not_counted.extend(_omissions([ErrorCode.INCOMPLETE_ENVELOPE.value]))
        not_counted.extend(_omissions(open_codes))
        if open_run and not open_codes:
            not_counted.extend(_omissions([ErrorCode.INCOMPLETE_TRACE.value]))
        causes = tuple(dict.fromkeys(not_counted))
        # (the followed beams' own exposure is one number where no two of them interfere)
        single = bool(values) and not any(cls.interferes for cls in traced_classes)
        comparisons: list[RatingComparison] = []
        for i, r in surface_ratings:
            traced = _upper(values, traced_classes, _quantity(r)) if single and not _diameter_range(r) else None
            comparison, scale = _compare(i, r, contribs, classes, open_above, surface_id, unknown_light, cfg, left_out,
                                         stored_lines, traced, causes, tol=tol)
            comparisons.append(comparison)
            if scale is not None and comparison.status != "CONDITIONS_MISMATCH":
                for cls in classes:
                    if cls.faint is not None:
                        for key, _, _, _ in cls.faint.lines:
                            scales[key] = min(scales.get(key, math.inf), scale)
        status = _surface_status(comparisons)
        element_id = owners[surface_id]
        if closed:
            closed_at.update(closed)
            closed_surfaces.append((surface_id, element_id))
        issues.extend(_withheld_notices(surface_id, element_id, withheld))
        if surface_ratings:
            crossed = [s for h in by_surface.get(surface_id, []) for s in _unvalidated(h, scope)]
            crossed += [s for _, h in reaching for s in _unvalidated(h, scope)]
            if crossed:
                element_scope.setdefault(element_id, []).extend(crossed)
            behind = [s for v in contribs if v.raw is not None for s in v.raw.power_scope]
            if behind:
                power_scope.setdefault(element_id, []).extend(behind)
        surfaces.append(SurfaceExposure(
            surface_id=surface_id, element_id=element_id, hits=tuple(v.hit for v in values),
            incident_power_W=_incident_power(values, traced_classes), peak_irradiance_W_per_cm2=peak,
            peak_irradiance_surface_W_per_cm2=peak_surface, linear_power_density_W_per_cm=linear, peak_fluence_J_per_cm2=fluence,
            complete=peak_ok and not open_above and not light and not beside and not traced_range and faint_peak == 0.0
            and not returns,
            reason_codes=codes, rating_status=status, ratings=tuple(comparisons),  # type: ignore[arg-type]
            untraced_hits=tuple(v.hit for v in light), beside_hits=tuple(v.hit for v in beside),
            coherent_groups=_coherent_groups(contribs, classes),
            faint_light_peak_W_per_cm2=faint_peak if faint_peak > 0.0 else None if nominal is None else nominal[0],
            faint_light_return_factor=return_factor_here if nominal is None or nominal[1] is None else nominal[1],
            value_basis=None if nominal is None else {
                "faint_light_peak_W_per_cm2": nominal[2],
                **({"faint_light_return_factor": nominal[2]} if nominal[1] is not None else {})}))
        exceeded = [c for c in comparisons if c.status == "EXCEEDS_RATING"]
        if exceeded:
            # (the most exceeded: the exposure over the damage threshold it was decided against, not over the rated value --
            # a TOTAL_PER_1E2_AREA value sets a threshold twice itself)
            worst = max(exceeded, key=_threshold_ratio)
            exact_value = exact_exposure(worst)
            value = exact_value if exact_value is not None else worst.exposure_lower_bound
            bound = "" if exact_value is not None else "at least "
            threshold, ratio = _damage_threshold(worst), _threshold_ratio(worst)
            times = (f"{bound}{ratio:.3g} x the rating" if threshold == worst.rating_value else
                     f"{bound}{ratio:.3g} x the damage threshold of {threshold:.4g} {worst.unit} that the rating sets for the "
                     "peak exposure")
            assessed = any(t.metric == POWER_RATING_UTILIZATION and t.device_id == element_id for t in project.targets)
            consequence = ("the power_rating_utilization target on the element assesses it as a requirement" if assessed else
                           "the run's status is unchanged, and a power_rating_utilization target on the element makes the "
                           "rating a requirement")
            issues.append(make_issue(
                ErrorCode.POWER_RATING_EXCEEDED, Status.NOT_APPLICABLE,
                f"the beams on {surface_id} reach {bound}{value:.4g} {worst.unit}, above the element's {worst.regime} rating of "
                f"{worst.rating_value:.4g} {worst.unit} at {worst.rating_wavelength_nm:g} nm ({times}); {consequence}",
                location=f"surface:{surface_id}", element_id=element_id, surface_id=surface_id, rating_index=worst.rating_index,
                regime=worst.regime, unit=worst.unit, rating_value=worst.rating_value, rating_wavelength_nm=worst.rating_wavelength_nm,
                exposure_value=value, exposure_is_lower_bound=exact_value is None, utilization=worst.utilization,
                damage_threshold=threshold, threshold_ratio=ratio, utilization_target=assessed,
                rating_status=worst.rating_status, rating_basis=worst.basis, rating_origin=worst.origin,
                exceeded_ratings=len(exceeded),
                suggested_action="reduce the power or enlarge the beam at this element, or choose an element rated for this exposure"))
    if closed_surfaces:
        issues.append(_closed_return_notice(gains, closed_at, closed_surfaces))
    elements = _elements(project, surfaces, ratings, open_run, (incomplete_codes if not trace_complete else ()) + unfollowed_codes)
    levels = untraced.levels_W if untraced is not None else {}
    faint_report = tuple(FaintLight(source_id=x.key[0], spectral_index=x.key[1], wavelength_nm=x.wavelength_nm, power_W=x.power_W,
                                    in_phase_power_W=x.root * x.root, followed_to_W=levels.get(x.key, 0.0)) for x in faint_lines)
    report = ExposureReport(surfaces=tuple(surfaces), elements=tuple(elements), trace_complete=trace_complete,
                            unknown_fate_power_W=unknown_W, untraced_light_included=untraced is not None, tolerances=tol.as_dict(),
                            model=EXPOSURE_MODEL, faint_light=faint_report)
    return ExposureOutcome(report, issues, {element: tuple(dict.fromkeys(found)) for element, found in element_scope.items()},
                           _deeper(scales, faint_lines, levels, cfg),
                           {element: tuple(dict.fromkeys(found)) for element, found in power_scope.items()})


def _nominal_faint(surface_id: str, classes: list[_Class],
                   gains: Mapping[float, ReturnGains] | None) -> tuple[float, float | None, ValueBasis] | None:
    """On a surface whose left-out light only pairs at a gain bound of 1 leave unbounded: (the most transverse peak
    irradiance that light reaches on its own, the largest factor of its returns, their entry) with each such pair
    counted at its centre-line incidence (``ReturnGains.nominal_factor``). One ray stands for the pair's light: light
    that meets the pair off that line (up to the critical angle inside a lens) is not in the number, so it is an
    estimate and bounds nothing. ``None`` where anything else leaves the light unbounded (a closed return, an analysis
    that reached its work limit, a refused coating use) or a pair has no centre-line value."""
    if not gains:
        return None
    pairs: dict[tuple[str, str], ReturnPair] = {}
    cycles: dict[tuple[str, ...], None] = {}
    peak, largest = 0.0, 1.0
    for cls in classes:
        if cls.faint is None:
            continue
        crossings = cls.faint.crossings
        if not math.isfinite(crossings):
            # (closed returns without a series: the open paths' count times the series of the one trail found)
            crossings = 0.0
            for _, wavelength, _, _ in cls.faint.lines:
                found = gains.get(wavelength)
                own = None if found is None else found.crossing_field(surface_id)
                if own is None:
                    return None
                if not math.isfinite(own):
                    own = found.nominal_crossing_field(surface_id)  # type: ignore[union-attr]
                    if own is None:
                        return None
                    cycles.update(dict.fromkeys((*found.closed_cycles, *found.search.candidates)))  # type: ignore[union-attr]
                crossings = max(crossings, own)
        factor = cls.faint.factor
        if not math.isfinite(factor):
            factor = 1.0
            for _, wavelength, _, _ in cls.faint.lines:
                found = gains.get(wavelength)
                own = None if found is None else found.exposure_factor(surface_id)
                if own is not None and not math.isfinite(own):
                    own = found.nominal_factor(surface_id)  # type: ignore[union-attr]
                    if own is None:
                        return None
                    pairs.update({pair.surfaces: pair for pair in found.saturated_pairs()})  # type: ignore[union-attr]
                factor = max(factor, 1.0 if own is None else own)
        amp = replace(cls.faint, factor=factor, crossings=crossings).amplitude(_peak)
        peak += amp * amp
        largest = max(largest, factor)
    if not (pairs or cycles) or not peak > 0.0:
        return None
    code = ErrorCode.EXPOSURE_BOUND_WITHHELD.value
    held = [*sorted(pairs)[:NAMED_RETURNS], *list(cycles)[:NAMED_RETURNS]]
    omitted = (Omitted(part=OmittedPart.POSSIBLE_RETURN, at=tuple(dict.fromkeys(sid for group in held for sid in group)), code=code),)
    assumed: tuple[Assumed, ...] = ()
    if pairs:
        at = tuple(dict.fromkeys(sid for pair in sorted(pairs)[:NAMED_RETURNS] for sid in pair))
        gain = max(pairs[pair].nominal_gain for pair in pairs)  # type: ignore[type-var]
        assumed = (Assumed(input=AssumedInput.SINGLE_RAY_FOR_BEAM, at=at, value=gain, code=code),)
    return peak, largest if pairs else None, ValueBasis(kind=ValueKind.ESTIMATE, omitted=omitted, assumed=assumed)


def _closed_return_notice(gains, closed: dict[float, tuple[tuple[str, ...], ...]],
                          affected: list[tuple[str, str]]) -> ValidationIssue:
    """The run's notice of the exposure upper bounds a closed return without a series withholds
    (``EXPOSURE_BOUND_WITHHELD``, status ``NOT_APPLICABLE``, cause ``CLOSED_RETURN_NOT_BOUNDED``): one for the run, since the
    return holds the left-out light of its wavelengths for every surface that light may reach. It names the returns and
    the surfaces and elements without a bound (at most ``NAMED_RETURNS`` each, with their counts); it changes no bound,
    comparison, target or status."""
    wavelengths = sorted(closed)
    kinds = closed_return_kinds(gains, wavelengths)
    cycles, named, counted = named_returns(sorted({cycle for groups in closed.values() for cycle in groups}),
                                           closed_return_text(kinds), "closed returns")
    where, facts = closed_return_facts(gains, wavelengths)
    surface_ids = [surface_id for surface_id, _ in affected]
    element_ids = list(dict.fromkeys(element_id for _, element_id in affected))
    shown = ", ".join(surface_ids[:NAMED_RETURNS]) + (f" and {len(surface_ids) - NAMED_RETURNS} more surfaces"
                                                      if len(surface_ids) > NAMED_RETURNS else "")
    return make_issue(
        ErrorCode.EXPOSURE_BOUND_WITHHELD, Status.NOT_APPLICABLE,
        f"The exposure upper bound on {shown} is withheld at {', '.join(f'{w:g}' for w in wavelengths)} nm because no series "
        f"bounds the left-out light's closed return through {named}: light leaving it may reach these surfaces; {where}.",
        location="exposures", quantity="exposure_upper_bound", wavelength_nm=wavelengths, cause="CLOSED_RETURN_NOT_BOUNDED",
        closed_return_surfaces=[", ".join(cycle) for cycle in cycles], surface_ids=surface_ids[:NAMED_RETURNS],
        surface_count=len(surface_ids), element_ids=element_ids, return_scope=SCENE_WIDE,
        **counted, **kinds_context(cycles, kinds), **facts)


@dataclass(frozen=True)
class _Withheld:
    """Why the left-out light has no upper bound on one surface: per wavelength the pairs of surfaces whose returns nothing
    bounds by the records' stated values (``pairs``: (wavelength, the pairs' surface ids); ``records``: those pairs), the
    coating uses refused on the surface (``uses``), with the pairs a refusal leaves at a gain bound of 1 that hold the
    surface (``partners``: (wavelength, the pairs' surface ids))."""

    pairs: tuple[tuple[float, tuple[tuple[str, str], ...]], ...] = ()
    uses: tuple[CoatingUse, ...] = ()
    partners: tuple[tuple[float, tuple[tuple[str, str], ...]], ...] = ()
    records: tuple[ReturnPair, ...] = ()


def _withheld(surface_id: str, faint_lines: list[_FaintLine], gains: Mapping[float, ReturnGains] | None) -> _Withheld:
    """Why ``surface_id`` has no upper bound by a pair or a refused coating use (the closed returns name themselves)."""
    pairs: list[tuple[float, tuple[tuple[str, str], ...]]] = []
    uses: list[CoatingUse] = []
    partners: list[tuple[float, tuple[tuple[str, str], ...]]] = []
    records: list[ReturnPair] = []
    for wavelength in sorted({x.wavelength_nm for x in faint_lines}):
        found = gains.get(wavelength) if gains else None
        if found is None or math.isfinite(found.exposure_factor(surface_id)):
            continue
        uses += [use for use in found.refused_coatings if use.surface_id == surface_id]
        unbounded = unbounded_pairs(found)
        if unbounded and not math.isfinite(found.factor(surface_id)):
            pairs.append((wavelength, unbounded))
            records += found.saturated_pairs()
        held = tuple(p.surfaces for p in found.refused_pairs if surface_id in p.surfaces)
        if held:
            partners.append((wavelength, held))
    return _Withheld(tuple(pairs), tuple(uses), tuple(partners), tuple(records))


def _return_omissions(withheld: _Withheld, closed: Mapping[float, tuple[tuple[str, ...], ...]], by_work: bool) -> list[Omitted]:
    """The left-out light nothing bounds on a surface, as the parts a reported exposure does not count: light that may
    come back between the surfaces of a pair or around a closed return (``at``: their surfaces, at most ``NAMED_RETURNS``
    returns), with the code of the notice that names them, and the returns an analysis that reached its work limit did
    not bound."""
    out: list[Omitted] = []
    if by_work:
        out.append(Omitted(part=OmittedPart.BUDGET, setting="max_return_analysis_work", code=ErrorCode.RETURN_ANALYSIS_BUDGET.value))
    returns: dict[tuple[str, ...], None] = {}
    for _, found in (*withheld.pairs, *withheld.partners):
        returns.update(dict.fromkeys(found))
    for use in withheld.uses:
        returns[(use.surface_id,)] = None
    for cycles in closed.values():
        returns.update(dict.fromkeys(cycles))
    if returns or not out:
        at = tuple(dict.fromkeys(sid for cycle in list(returns)[:NAMED_RETURNS] for sid in cycle))
        code = ErrorCode.EXPOSURE_BOUND_WITHHELD.value if withheld.pairs or withheld.partners or withheld.uses or closed else (
            ErrorCode.FAINT_LIGHT_RECIRCULATES.value)
        out.append(Omitted(part=OmittedPart.POSSIBLE_RETURN, at=at, code=code))
    return out


def unbounded_pairs(found: ReturnGains) -> tuple[tuple[str, str], ...]:
    """The surfaces of the pairs whose returns nothing bounds by the records' stated values."""
    return tuple(p.surfaces for p in found.pairs if not math.isfinite(p.returns))


def _refusal_text(use: CoatingUse) -> tuple[str, str]:
    """(how the left-out light meets a refused coating, what obtains a bound)."""
    low, high = use.valid_aoi_deg
    states = f"coating record {use.record_id} states {low:g}-{high:g} deg air-equivalent"
    if use.refused == NOT_BOUNDED:
        met = "and the left-out light meets it at angles its beam state does not bound"
        action = "keep the beam at this element a Gaussian beam inside the clear aperture, or use a coating whose record states this side's angles"
    elif use.refused == OUTSIDE_WAVELENGTH:
        bands = ", ".join(f"{a:g}-{b:g}" for a, b in use.valid_wavelength_nm)
        states = f"coating record {use.record_id} states {bands} nm"
        met = f"and the left-out light meets it from {use.medium} at {use.wavelength_nm:g} nm"
        action = "use a coating whose record states this wavelength"
    elif use.refused == NO_AIR_EQUIVALENT:
        met = f"and the left-out light meets it from {use.medium} up to {use.high_deg:.3g} deg, above the critical angle, which has no air-equivalent angle"
        action = "use a coating with data for this side above the critical angle, or reduce the beam at this surface"
    else:
        met = f"and the left-out light meets it from {use.medium} at up to {use.air_deg[1]:.3g} deg air-equivalent"  # type: ignore[index]
        action = "use a coating whose record states these angles, or reduce the beam at this surface"
    return f"{states} {met}", action


def _pair_text(pair: tuple[str, str]) -> str:
    return f"{pair[0]} and {pair[1]}"


#: The scope of a pair's return factor: every surface and plane of the scene (the light a pair sends out may reach any of
#: them; the surfaces it can reach are not narrowed).
SCENE_WIDE = "SCENE_WIDE_NOT_LOCALIZED"

#: What obtains a bound from a pair at a gain bound of 1, by why a surface's reflectance bound is 1 (``ReturnPair.saturation``;
#: a cause absent here: a coating whose reflectance bound is below 1).
_PAIR_ACTION = "state a coating on these surfaces whose reflectance bound is below 1"
_SATURATION_ACTION = {
    COATING_DATA_RANGE: "state the coating's reflectance over the angles the pair's light meets it at (return_pair_angles_deg)",
    COATING_EVALUATION_ISSUE: "correct the coating record the pair's surfaces name",
    INDEX_UNKNOWN: "state the refractive index of the media at these surfaces for this wavelength",
    REFUSED_COATING_USE: "use coatings whose records state the angles of the refused uses",
}


def pair_facts(shown: list[tuple[str, str]], records) -> tuple[str, dict]:
    """What a withheld-bound notice states about the pairs ``shown`` (their ``ReturnPair`` among ``records``): what obtains
    a bound, by the causes of their gain bound of 1, and the context -- per pair, in ``shown``'s order, why each surface's
    reflectance bound is 1 (``return_pair_saturation``) and the angles of incidence each is met at (deg, low-high); the
    pairs whose records state a round-trip gain at their centre-line incidence, with that gain
    (``return_pair_nominal_surfaces``, ``return_pair_nominal_gain``) -- and the scope of the pair's factor (``SCENE_WIDE``)."""
    by_surfaces = {record.surfaces: record for record in records}
    found = [by_surfaces.get(pair) for pair in shown]
    causes = list(dict.fromkeys(code for record in found if record is not None and record.saturation for code in record.saturation))
    actions = list(dict.fromkeys(_SATURATION_ACTION.get(code, _PAIR_ACTION) for code in causes)) or [_PAIR_ACTION]
    context: dict = {
        "return_scope": SCENE_WIDE,
        "return_pair_saturation": ["" if r is None or r.saturation is None else ", ".join(r.saturation) for r in found],
        "return_pair_angles_deg": ["" if r is None else "; ".join(f"{low:.4g}-{high:.4g}" for low, high in r.angles_deg) for r in found],
    }
    nominal = [r for r in found if r is not None and r.nominal_gain is not None]
    if nominal:
        context["return_pair_nominal_surfaces"] = [", ".join(r.surfaces) for r in nominal]
        context["return_pair_nominal_gain"] = [r.nominal_gain for r in nominal]
    return "; ".join(actions), context


def _held_pairs(pairs) -> dict:
    """The context naming the pairs a refused coating use leaves at gain bound 1 (at most ``NAMED_RETURNS``, then counted)."""
    shown, _text, counted = named_returns(pairs, _pair_text, "pairs of surfaces")
    return {"return_pair_surfaces": [", ".join(pair) for pair in shown], **counted}


def _withheld_notices(surface_id: str, element_id: str, withheld: _Withheld) -> list[ValidationIssue]:
    """The run-level notices of a withheld upper bound (``EXPOSURE_BOUND_WITHHELD``, status ``NOT_APPLICABLE``): they change
    no bound, comparison, target or status; they name the surface and what obtains a bound."""
    out: list[ValidationIssue] = []
    if withheld.pairs:
        wavelengths = [wavelength for wavelength, _ in withheld.pairs]
        pairs, named, counted = named_returns(sorted({pair for _, found in withheld.pairs for pair in found}), _pair_text,
                                              "pairs of surfaces")
        action, facts = pair_facts(pairs, withheld.records)
        out.append(make_issue(
            ErrorCode.EXPOSURE_BOUND_WITHHELD, Status.NOT_APPLICABLE,
            f"The exposure upper bound on {surface_id} is withheld at {', '.join(f'{w:g}' for w in wavelengths)} nm because {named} return left-out "
            f"light with a round-trip gain bound of 1.",
            location=f"surface:{surface_id}", element_id=element_id, surface_id=surface_id, quantity="exposure_upper_bound",
            wavelength_nm=wavelengths, return_pair_surfaces=[", ".join(pair) for pair in pairs], cause="RETURN_PAIR_GAIN_BOUND_1",
            suggested_action=action, **counted, **facts))
    held = {wavelength: found for wavelength, found in withheld.partners}
    for use in withheld.uses:
        text, action = _refusal_text(use)
        out.append(make_issue(
            ErrorCode.EXPOSURE_BOUND_WITHHELD, Status.NOT_APPLICABLE,
            f"The exposure upper bound on {surface_id} is withheld at {use.wavelength_nm:g} nm because {text}.",
            location=f"surface:{surface_id}", element_id=element_id, surface_id=surface_id, quantity="exposure_upper_bound",
            wavelength_nm=[use.wavelength_nm], cause="COATING_USE_REFUSED", suggested_action=action,
            **_held_pairs(held.get(use.wavelength_nm, ())), **use.context()))
    refused_here = {use.wavelength_nm for use in withheld.uses}
    for wavelength, found in withheld.partners:
        if wavelength in refused_here:
            continue
        shown, named, counted = named_returns(found, _pair_text, "pairs of surfaces")
        action = "use coatings whose records state the angles of the refused uses named on the facing surface"
        out.append(make_issue(
            ErrorCode.EXPOSURE_BOUND_WITHHELD, Status.NOT_APPLICABLE,
            f"The exposure upper bound on {surface_id} is withheld at {wavelength:g} nm because a refused coating use leaves {named} with a "
            f"round-trip gain bound of 1.",
            location=f"surface:{surface_id}", element_id=element_id, surface_id=surface_id, quantity="exposure_upper_bound",
            wavelength_nm=[wavelength], cause="COATING_USE_REFUSED_ON_FACING_SURFACE",
            return_pair_surfaces=[", ".join(pair) for pair in shown], suggested_action=action, **counted))
    return out


def _damage_threshold(comparison: RatingComparison) -> float:
    """The damage threshold an exceeded comparison was decided against: ``rated_threshold_high`` (the rated value as a
    threshold: the value, or twice it for a TOTAL_PER_1E2_AREA value), the rated value when the comparison states none."""
    return comparison.rated_threshold_high if comparison.rated_threshold_high is not None else comparison.rating_value


def _threshold_ratio(comparison: RatingComparison) -> float:
    """An exceeded comparison's exposure (its lower bound when not exact) over its damage threshold."""
    exact = exact_exposure(comparison)
    value = exact if exact is not None else comparison.exposure_lower_bound
    threshold = _damage_threshold(comparison)
    return (value or 0.0) / threshold if threshold > 0.0 else math.inf


def _deeper(scales: dict[LineKey, float], faint_lines: list[_FaintLine], levels: dict[LineKey, float],
            cfg: NumericalConfig) -> dict[LineKey, float]:
    """Per line whose left-out light must shrink by the amplitude factor ``scales[line]``: the level a deeper probe pass
    follows it down to -- ``scale^2`` x the smaller of the line's present level and its left-out power, ``DEEPER_HEADROOM``
    below that, not below the power floor -- when that is below the present level."""
    powers = {x.key: x.power_W for x in faint_lines}
    out: dict[LineKey, float] = {}
    for line, scale in sorted(scales.items()):
        level = levels.get(line)
        if level is None or line not in powers or not scale > 0.0 or level <= cfg.power_epsilon_W:
            continue
        wanted = scale * scale * min(level, powers[line]) / DEEPER_HEADROOM
        wanted = max(wanted, cfg.power_epsilon_W)
        if wanted < level:
            out[line] = wanted
    return out


def _utilization_bounds(c: RatingComparison) -> tuple[float, float | None]:
    """(lower, upper or ``None``) of a comparison's exposure / damage threshold, with the threshold the rated value
    establishes (``rated_threshold``, ``RatingComparison.rated_threshold_low`` / ``_high``; none: lower 0, no upper)."""
    lowest, highest = c.rated_threshold_low, c.rated_threshold_high
    exact = exact_exposure(c)
    low_exposure = exact if exact is not None else (c.exposure_lower_bound or 0.0)
    high_exposure = exact if exact is not None else c.exposure_upper_bound
    low = 0.0 if highest is None else low_exposure / highest
    return low, None if high_exposure is None or lowest is None else high_exposure / lowest




def _linear_scale(c: RatingComparison) -> tuple[float, float]:
    """(least, most) factor ``k`` by which a comparison's rated value stands above the same threshold expressed in ``P / d``
    of the 1/e^2 diameter: 1 for a ``1/E2`` diameter and for units other than ``W/cm``, ``1 / 0.5887`` for ``FWHM`` (the
    FWHM diameter is 0.5887 of the 1/e^2 one), either for a definition not stated (``_notes``)."""
    if c.unit != "W/cm" or E2_NOTE in c.notes:
        return 1.0, 1.0
    if FWHM_NOTE in c.notes:
        return 1.0 / FWHM_PER_E2_DIAMETER, 1.0 / FWHM_PER_E2_DIAMETER
    return 1.0, 1.0 / FWHM_PER_E2_DIAMETER


def _at_most_open(comparisons: tuple[RatingComparison, ...]
                  ) -> tuple[tuple[tuple[RatingComparison, bool], ...], bool]:
    """Whether a comparison with a threshold at most its rated value -- or one not verified, which may have such a
    threshold -- leaves the surface without an upper bound, and whether one with a threshold at most its value does so
    against a rating of its own quantity that applies as stated or with a threshold at least its value (the ratings
    disagree). Such a comparison bounds the threshold from above only: the exposure may exceed it below its rated value. A
    comparison of the same unit that applies as stated or with a threshold at least its value, whose lowest threshold is
    no larger than this one's highest under every reading (``rated_threshold``; ``W/cm`` values compared in ``P / d`` of
    the 1/e^2 diameter, ``_linear_scale``), puts the threshold between the two and bounds it; one whose lowest may be
    larger contradicts it; ratings of other units, and typical values, are criteria of their own. A
    comparison not verified is never weaker here than one with a threshold at most its value: removing a condition of a
    rating does not make the surface's bound tighter. The comparisons that leave it open, each with whether it disagrees
    with a rating of its unit (empty: bounded), and whether any does."""
    openings: list[tuple[RatingComparison, bool]] = []
    disagree = False
    for c in comparisons:
        if c.applicability not in (AT_MOST, NOT_VERIFIED):
            continue
        same = [a for a in comparisons if a.applicability in (AS_RATED, AT_LEAST) and a.unit == c.unit
                and a.rated_threshold_low is not None]
        highest = c.rated_threshold_high
        if highest is not None and any(a.rated_threshold_low / _linear_scale(a)[0] <= highest / _linear_scale(c)[1]  # type: ignore[operator]
                                       for a in same):
            continue
        disagrees = bool(same) and c.applicability == AT_MOST
        openings.append((c, disagrees))
        disagree = disagree or disagrees
    return tuple(openings), disagree


def _surface_bounds(comparisons: tuple[RatingComparison, ...]) -> tuple[float | None, float | None, bool, bool]:
    """(lower, upper, exact, disagree) of a surface's utilization (``SurfaceExposure.rating_status``): ``lower`` the
    largest lower bound of the comparisons that apply as stated or with a threshold at most the rated value (``None``:
    none); ``upper`` only with a comparison that applies as stated or with a threshold at least the rated value, the
    largest upper bound of those and of the comparisons not verified (``None``: none, or one is unbounded), and none
    while a comparison with a threshold at most its rated value, or one not verified, is not bounded by a rating of its
    unit (``_at_most_open``); ``exact`` when only exactly known comparisons applying as stated decide it. ``disagree``:
    the ratings disagree -- a rating with a threshold at most its value may lie, under some reading, below every rating
    of its unit that applies as stated or with a threshold at least its value, or the lower bound lies above the upper
    bound --; the lower bound then stands (a rating that bounds the utilization from below is not overruled) and there
    is no upper bound. A typical value bounds nothing and is left out."""
    comparisons = tuple(c for c in comparisons if not _typical(c))
    lows = [_utilization_bounds(c)[0] for c in comparisons if c.applicability in (AS_RATED, AT_MOST)]
    above = [c for c in comparisons if c.applicability in (AS_RATED, AT_LEAST)]
    unverified = [c for c in comparisons if c.applicability == NOT_VERIFIED]
    upper: float | None = None
    if above:
        highs = [_utilization_bounds(c)[1] for c in above + unverified]
        upper = None if any(h is None for h in highs) else max(h for h in highs if h is not None)
    lower = max(lows) if lows else None
    openings, disagree = _at_most_open(comparisons)
    open_ = bool(openings)
    if open_:
        upper = None
    if lower is not None and upper is not None and lower > upper:
        disagree, upper = True, None
    applying = [c for c in comparisons if c.applicability in APPLYING]
    exact = (bool(applying) and not unverified and not disagree and not open_
             and all(c.applicability == AS_RATED and exact_exposure(c) is not None for c in applying))
    return lower, upper, exact, disagree


def _openings(surface_id: str, comparisons: tuple[RatingComparison, ...]) -> list[UtilizationOpening]:
    """What leaves a rated surface's utilization without an upper bound (``_surface_bounds``), one entry per reason
    (``UtilizationOpening``); empty where it has one."""
    lower, upper, _, _ = _surface_bounds(comparisons)
    if upper is not None:
        return []
    counted = tuple(c for c in comparisons if not _typical(c))
    above = [c for c in counted if c.applicability in (AS_RATED, AT_LEAST)]
    out: list[UtilizationOpening] = []
    if not above:
        out.append(UtilizationOpening(surface_id=surface_id, cause="NO_UPPER_RATING"))
    else:
        for c in above + [c for c in counted if c.applicability == NOT_VERIFIED]:
            if _utilization_bounds(c)[1] is None:
                out.append(UtilizationOpening(
                    surface_id=surface_id, rating_index=c.rating_index, applicability=c.applicability,
                    cause="NO_THRESHOLD" if c.rated_threshold_low is None else "EXPOSURE_UNBOUNDED"))
    for c, disagrees in _at_most_open(counted)[0]:
        out.append(UtilizationOpening(surface_id=surface_id, rating_index=c.rating_index, applicability=c.applicability,
                                      cause="RATINGS_DISAGREE" if disagrees else "THRESHOLD_OPEN_BELOW"))
    if not out:  # (the lower bound lies above the upper bound)
        out.append(UtilizationOpening(surface_id=surface_id, cause="RATINGS_DISAGREE"))
    return out


def _opening_text(o: UtilizationOpening) -> str:
    what = {"NO_UPPER_RATING": "no rating applies as stated or with a threshold at least its value",
            "EXPOSURE_UNBOUNDED": "the exposure has no upper bound", "NO_THRESHOLD": "the rated value establishes no threshold",
            "THRESHOLD_OPEN_BELOW": "the damage threshold may lie anywhere below the rated value",
            "RATINGS_DISAGREE": "the ratings disagree"}[o.cause]
    rating = "" if o.rating_index is None else f" rating {o.rating_index} ({o.applicability})"
    return f"{o.surface_id}{rating}: {what}"


def _reported_ratio(c: RatingComparison) -> tuple[float, ValueBasis | None] | None:
    """A comparison's reported exposure over the damage threshold its rated value establishes (the rated value itself
    where it establishes none: read as a peak value), with the entry of that ratio (``RatingComparison.utilization``'s,
    its omissions in this ratio's unit); ``None``: the comparison reports no exposure."""
    if c.exposure_value is None or c.rating_value <= 0.0:
        return None
    threshold = _threshold(c)
    bases = c.value_basis or {}
    entry, scale = bases.get("utilization"), c.rating_value / threshold
    if entry is None or entry.kind is ValueKind.LOWER_BOUND:  # (a decided exceedance states its lower bound's ratio)
        entry, scale = bases.get("exposure_value"), 1.0 / threshold
    if entry is not None:
        # (the entry's omissions are stated in exposure / rated value, or in the exposure's unit)
        entry = entry.model_copy(update={"omitted": tuple(
            o if o.upper is None else o.model_copy(update={"upper": o.upper * scale}) for o in entry.omitted)})
    return c.exposure_value / threshold, entry


def _threshold(c: RatingComparison) -> float:
    """The damage threshold a comparison's rated value establishes (the rated value where it establishes none)."""
    return c.rated_threshold_low if c.rated_threshold_low is not None else c.rating_value


def _unreported(surface: SurfaceExposure, comparisons: list[RatingComparison], value: float) -> Omitted:
    """A rated surface whose counted comparisons report no exposure, as light the element's number does not count: light
    left out as negligible alone where that is all that reaches it (with the most its proven bound can put the largest
    ratio above ``value``), light not followed otherwise."""
    faint_only = (surface.faint_light_peak_W_per_cm2 is not None and not surface.hits and not surface.untraced_hits
                  and not surface.beside_hits)
    if not faint_only:
        return Omitted(part=OmittedPart.NOT_FOLLOWED, at=(surface.surface_id,))
    bounds = [None if c.exposure_upper_bound is None else c.exposure_upper_bound / _threshold(c) for c in comparisons]
    most = None if not bounds or any(b is None for b in bounds) else max(max(b for b in bounds if b is not None) - value, 0.0)
    return Omitted(part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, at=(surface.surface_id,), upper=most,
                   code=ErrorCode.FAINT_LIGHT_BOUNDED.value)


def _element_estimate(rated: list[SurfaceExposure | None]) -> tuple[float, ValueBasis] | None:
    """The largest reported exposure / damage threshold over an element's rated surfaces (``rated``; ``None``: a rated
    surface no light meets) where its utilization is not exact, with its entry: the comparisons counted are those of the
    ratings that are not known to be for other light (every comparison where all are), the number is the largest of
    their ratios, and the entry is that comparison's with what the others leave open -- a rated surface or a counted
    comparison without a reported exposure is light not counted there. ``None``: no comparison reports an exposure, or
    nothing qualifies the number (it is then no estimate: the bounds say what is known)."""
    comparisons = [(s, c) for s in rated if s is not None for c in s.ratings]
    counted = [(s, c) for s, c in comparisons if c.applicability != NOT_APPLICABLE] or comparisons
    found = [(ratio, s, c) for s, c in counted for ratio in (_reported_ratio(c),) if ratio is not None]
    if not found:
        return None
    (value, entry), surface, comparison = max(found, key=lambda row: row[0][0])
    omitted = list(entry.omitted) if entry is not None else []
    model = entry is None or entry.kind is ValueKind.MODEL
    for (_, other), s, c in found:
        if other is not None and c is not comparison:
            model = model and other.kind is ValueKind.MODEL
            omitted.extend(o.model_copy(update={"upper": None}) for o in other.omitted)
    silent: dict[str, tuple[SurfaceExposure, list[RatingComparison]]] = {}
    for s, c in counted:
        if _reported_ratio(c) is None:
            silent.setdefault(s.surface_id, (s, []))[1].append(c)
    if silent or any(s is None for s in rated):
        model = False
        omitted.extend(_unreported(s, cs, value) for s, cs in silent.values())
    omitted_ = tuple(dict.fromkeys(omitted))
    assumed = entry.assumed if entry is not None else ()
    validity = entry.validity if entry is not None else ()
    if not (omitted_ or assumed or validity):
        return None
    if model and (assumed or any(v.within is not True for v in validity) or any(not o.accounting for o in omitted_)):
        model = False
    return value, ValueBasis(kind=ValueKind.MODEL if model else ValueKind.ESTIMATE, omitted=omitted_, assumed=assumed,
                             validity=validity)


def _elements(project: ProjectSnapshot, surfaces: list[SurfaceExposure], ratings: dict[str, list[tuple[int, PowerRating]]],
              open_above: bool, open_codes: tuple[str, ...]) -> list[ElementExposure]:
    """Per element: the most severe status of its rated surfaces and the bounds of its largest utilization -- below by the
    largest of the surfaces' lower bounds (a decided exceedance stands), above by the largest of their upper bounds
    (``_surface_bounds``); no upper bound while a rated surface has none (also where its ratings disagree), and no lower
    bound above 0 where only ratings with a threshold at least the rated value bound it. Where the utilization is not
    exact, ``max_utilization`` is the largest reported exposure / damage threshold with its entry (``_element_estimate``):
    a number beside the bounds, which decides nothing."""
    by_id = {s.surface_id: s for s in surfaces}
    out: list[ElementExposure] = []
    for comp in project.components:
        if not comp.surfaces:
            continue
        ids = [sid for surface in comp.surfaces for sid in (facet_patch_ids(comp.instance_id, surface) if isinstance(surface, MeshSurface)
                                                             else (f"{comp.instance_id}/{surface.surface_id}",))]
        rated_count = len(comp.power_ratings)
        if not rated_count:
            out.append(ElementExposure(element_id=comp.instance_id, rating_status="RATING_MISSING"))
            continue
        rated = [sid for sid in ids if ratings.get(sid)]
        unrated = tuple(sid for sid in ids if not ratings.get(sid))
        statuses: list[tuple[str, str]] = []  # (surface, status)
        lows: list[float] = []  # the surfaces' lower bounds
        highs: list[float | None] = []  # the surfaces' upper bounds
        exact = True
        unchecked = mismatched = False
        codes: list[str] = []
        counts: dict[str, int] = {}
        openings: list[UtilizationOpening] = []
        for sid in rated:
            exposure = by_id.get(sid)
            if exposure is None:  # no light meets it: a known zero only when nothing may reach it
                statuses.append((sid, "UNKNOWN" if open_above else "WITHIN_RATING"))
                lows.append(0.0)
                highs.append(None if open_above else 0.0)
                exact = exact and not open_above
                codes.extend(open_codes if open_above else ())
                if open_above:
                    openings.append(UtilizationOpening(surface_id=sid, cause="EXPOSURE_UNBOUNDED"))
                continue
            codes.extend(exposure.reason_codes)
            statuses.append((sid, exposure.rating_status))
            for c in exposure.ratings:
                counts[c.status] = counts.get(c.status, 0) + 1
            openings.extend(_openings(sid, exposure.ratings))
            lower, upper, surface_exact, _ = _surface_bounds(exposure.ratings)
            if lower is not None or upper is not None:
                lows.append(0.0 if lower is None else lower)
            highs.append(upper)
            exact = exact and surface_exact
            unchecked = unchecked or any(c.applicability == NOT_VERIFIED for c in exposure.ratings)
            mismatched = mismatched or not any(c.applicability in APPLYING or c.applicability == NOT_VERIFIED for c in exposure.ratings)
        worst_sid, status = max(statuses, key=lambda row: RATING_SEVERITY[row[1]])
        lower = max(lows) if lows else None
        upper = None if not highs or any(h is None for h in highs) else max(h for h in highs if h is not None)
        # (each surface's bounds are in order -- ratings that disagree leave it no upper bound --, so their maxima are too)
        exact = exact and lower is not None and upper is not None and lower == upper
        if unchecked:
            codes.append(ErrorCode.POWER_RATING_CONDITIONS_NOT_VERIFIED.value)
        if mismatched:
            codes.append(ErrorCode.POWER_RATING_CONDITIONS_MISMATCH.value)
        estimate = None if exact else _element_estimate([by_id.get(sid) for sid in rated])
        out.append(ElementExposure(
            element_id=comp.instance_id, rating_status=status, rating_count=rated_count,  # type: ignore[arg-type]
            max_utilization=lower if exact else None if estimate is None else estimate[0], worst_surface_id=worst_sid,
            utilization_lower_bound=lower if not exact else None,
            utilization_upper_bound=upper if not exact and lower is not None else None, unrated_surface_ids=unrated,
            reason_codes=() if exact else _codes(codes),
            value_basis=None if estimate is None else {"max_utilization": estimate[1]},
            rating_status_counts=dict(sorted(counts.items())) or None,
            utilization_open_by=tuple(openings) if upper is None and not exact else ()))
    return out


# ------------------------------------------------------------------------------------------------- target metric

ELEMENT_METRICS = {POWER_RATING_UTILIZATION: "1"}


@dataclass(frozen=True)
class Utilization:
    """``power_rating_utilization`` of one element: the value, the measurement status, codes, message, and the interval
    ``bounds`` (lower, upper or ``inf``) the target is graded on when the value is not exact. The value is the exact
    utilization, or for an interval the largest exposure / damage threshold of the ratings that apply (``_reported``:
    the exposure exact, or its lower bound -- as a power target reports the traced light's value), within the interval.
    Where nothing is known of the utilization (no rating applies, none can be checked, nothing bounds it) the value is
    the element's reported ratio with its entry (``ElementExposure.max_utilization``, ``basis``) under status
    ``UNKNOWN``: reported, never graded. ``None`` when no comparison reports an exposure (only light the trace leaves
    out as negligible or does not follow, or beams whose fields may cancel, reach the rated surfaces -- as a power
    target on a plane no traced beam reaches has no value while light may arrive)."""

    value: float | None
    status: Status
    codes: tuple[str, ...] = ()
    message: str = ""
    bounds: tuple[float, float] | None = None
    #: what ``value`` is where it is not the exact utilization nor a value inside proven bounds: the element's reported
    #: ratio with its entry (``ElementExposure.max_utilization``), which decides nothing
    basis: ValueBasis | None = None


def _reported(exposures: ExposureReport, element_id: str) -> float | None:
    """The largest exposure / damage threshold over the element's comparisons that apply (as stated, or with a threshold at
    least or at most the rated value) whose exposure is exact or has a lower bound above 0: the value a target reports for
    an interval (``None``: no such comparison). The threshold is the one the rated value establishes (``rated_threshold``);
    a rating with a threshold at least the rated value bounds the utilization from above only, yet its ratio is what the
    known exposure gives against it; a rating not verified, or one that establishes no threshold, is left out."""
    best: float | None = None
    for surface in exposures.surfaces:
        if surface.element_id != element_id:
            continue
        for c in surface.ratings:
            exact = exact_exposure(c)
            if c.applicability in APPLYING and c.rated_threshold_low is not None and (
                    exact is not None or c.exposure_lower_bound is not None):
                exposure = exact if exact is not None else c.exposure_lower_bound
                ratio = exposure / c.rated_threshold_low  # type: ignore[operator]
                best = ratio if best is None else max(best, ratio)
    return best


def _unverified_kinds(exposures: ExposureReport, element_id: str) -> list[str]:
    """The conditions of the element's ratings that cannot be checked against its light, with why (``pulse count: an input
    the Project does not state``, ``beam diameter: light of unknown size``, ...)."""
    kinds: dict[str, None] = {}
    for surface in exposures.surfaces:
        if surface.element_id == element_id:
            for c in surface.ratings:
                if c.applicability != NOT_VERIFIED:
                    continue
                for condition in c.conditions:
                    if condition.state == NOT_VERIFIABLE and condition.reason is not None:
                        kinds[f"{condition.condition.replace('_', ' ')}: {REASON_TEXT[condition.reason]}"] = None
    return list(kinds)


#: What a condition that differs in one direction says about a rating's test, by condition and direction (messages).
_ONE_SIDED_TEXT = {
    ("beam_diameter", AT_MOST): "measured with a test beam smaller than the light here",
    ("beam_diameter", AT_LEAST): "measured with a test beam at least as large as the light here",
    ("pulse_count", AT_MOST): "measured over no more pulses per site than the light here gives",
    ("pulse_count", AT_LEAST): "measured over at least as many pulses per site as the light here gives",
}


@dataclass(frozen=True)
class _Findings:
    """Why an element's utilization is an interval, from the comparisons on its rated surfaces: the tests of the ratings
    that bound it from below only (``below``) or from above only (``above``; ``_ONE_SIDED_TEXT``), the surfaces whose
    ratings disagree (``_surface_bounds``), a ``W/cm`` exposure taken over two readings of an unstated diameter definition
    (``linear_range``), an exposure with no upper bound (``open_exposure``) and a typical value (``typical``)."""

    below: tuple[str, ...] = ()
    above: tuple[str, ...] = ()
    disagree: tuple[str, ...] = ()
    linear_range: bool = False
    open_exposure: bool = False
    typical: bool = False


def _findings(exposures: ExposureReport, element_id: str) -> _Findings:
    below: dict[str, None] = {}
    above: dict[str, None] = {}
    disagree: list[str] = []
    linear_range = open_exposure = typical = False
    for surface in exposures.surfaces:
        if surface.element_id != element_id or not surface.ratings:
            continue
        if _surface_bounds(surface.ratings)[3]:
            disagree.append(surface.surface_id)
        for c in surface.ratings:
            if c.applicability in (AT_MOST, AT_LEAST):
                for condition in c.conditions:
                    text = _ONE_SIDED_TEXT.get((condition.condition, c.applicability))
                    if condition.state == DIFFERS and condition.effect == c.applicability and text is not None:
                        (below if c.applicability == AT_MOST else above)[text] = None
            if c.applicability == NOT_APPLICABLE:
                continue
            linear_range = linear_range or LINEAR_RANGE_NOTE in c.notes
            open_exposure = open_exposure or (exact_exposure(c) is None and c.exposure_upper_bound is None)
            typical = typical or c.rating_status == "TYPICAL"
    return _Findings(tuple(below), tuple(above), tuple(disagree), linear_range, open_exposure, typical)


def measure_utilization(t: Target, exposures: ExposureReport | None) -> Utilization:
    """``power_rating_utilization`` of the element ``t.device_id`` over its rated surfaces: exact, or an interval whose
    lower bound comes from the ratings that apply as stated or with a threshold at most the rated value (a decided
    exceedance stands) and whose upper bound covers every rating that applies as stated or with a threshold at least the
    rated value and every rating not verified (none while a rated surface has no such rating, or a rating there with a
    threshold at most its value or not verified that no applying rating of its unit bounds, ``_at_most_open``, or its
    ratings disagree). The message names what makes it an interval."""

    if exposures is None:
        return Utilization(None, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,), "the physics stage did not run")
    if t.device_id is None:
        return Utilization(None, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,), "power_rating_utilization requires the element it rates: the target's "
                           "device_instance_id (device_id in an Engine request)")
    element = next((e for e in exposures.elements if e.element_id == t.device_id), None)
    if element is None:
        return Utilization(None, Status.UNKNOWN, (ErrorCode.UNKNOWN_REFERENCE.value,), f"{t.device_id!r} is not an optical element of this run")
    if element.rating_status == "RATING_MISSING":
        return Utilization(None, Status.UNKNOWN, (ErrorCode.POWER_RATING_MISSING.value,), f"{t.device_id} has no power rating")
    exceeds = (ErrorCode.POWER_RATING_EXCEEDED.value,) if element.rating_status == "EXCEEDS_RATING" else ()
    # (the exact utilization carries no entry; with one, the number is the reported ratio beside the bounds)
    entry = (element.value_basis or {}).get("max_utilization")
    if element.max_utilization is not None and entry is None:
        return Utilization(element.max_utilization, Status.PASS, exceeds)
    ratio = element.max_utilization if entry is not None else None
    stated = "" if ratio is None else (f"the exposure on {t.device_id}'s rated surfaces is {ratio:.6g} of the rated damage "
                                       "threshold as stated (exposures.elements[].max_utilization with its value_basis); ")
    lower = element.utilization_lower_bound
    upper = math.inf if element.utilization_upper_bound is None else element.utilization_upper_bound
    unchecked = ErrorCode.POWER_RATING_CONDITIONS_NOT_VERIFIED.value in element.reason_codes
    mismatched = ErrorCode.POWER_RATING_CONDITIONS_MISMATCH.value in element.reason_codes
    kinds = _unverified_kinds(exposures, t.device_id)
    found = _findings(exposures, t.device_id)
    reasons = []
    if unchecked:
        reasons.append(f"a rating whose conditions cannot be checked ({'; '.join(kinds)}) may apply")
    if mismatched:
        reasons.append("a rated surface has no rating whose conditions match its beams")
    if found.disagree:
        reasons.append(f"the ratings of {', '.join(found.disagree)} disagree (one puts the damage threshold at most its "
                       "value, below a rating of the same quantity that applies as stated or puts it at least its value): "
                       "the lower bound stands, and there is no upper bound")
    if found.below:
        reasons.append(f"a rating {' or '.join(found.below)} (its damage threshold is at most the rated value) bounds the "
                       "utilization from below only")
    if found.above:
        reasons.append(f"a rating {' or '.join(found.above)} (its damage threshold is at least the rated value) bounds the "
                       "utilization from above only")
    if found.linear_range:
        reasons.append("a W/cm rating that does not state its beam diameter definition takes P / d over its 1/e^2 and FWHM "
                       "readings")
    if found.typical:
        reasons.append("a typical value (status TYPICAL) bounds the damage threshold of no particular part")
    exposure_codes = [c for c in element.reason_codes if c not in (ErrorCode.POWER_RATING_CONDITIONS_NOT_VERIFIED.value,
                                                                   ErrorCode.POWER_RATING_CONDITIONS_MISMATCH.value)]
    explained_above = unchecked or mismatched or bool(found.disagree) or bool(found.below)
    if found.open_exposure or (exposure_codes and not math.isfinite(upper) and not explained_above):
        reasons.append("light whose fate the trace does not know, or a beam whose size there is not known, may add to the "
                       "exposure without bound")
    elif exposure_codes:
        reasons.append("mutually coherent beams combine with phases the exposure leaves open, or light the trace does not "
                       "follow or leaves out as negligible, or the tail of a beam whose chief ray misses the element, reaches "
                       "it too")
    codes = element.reason_codes
    see = f"see exposures.surfaces[].ratings[].conditions{'; causes ' + ', '.join(codes) if codes else ''}"
    if element.utilization_open_by:
        see += f"; no upper bound from {'; '.join(_opening_text(o) for o in element.utilization_open_by)}"
    applying = any(c.applicability in APPLYING for surface in exposures.surfaces if surface.element_id == t.device_id
                   for c in surface.ratings)
    if lower is None or (lower == 0.0 and not math.isfinite(upper)):  # nothing known: no number, never 0
        if element.rating_status == "CONDITIONS_MISMATCH" and not applying and not unchecked:
            return Utilization(ratio, Status.UNKNOWN, (ErrorCode.POWER_RATING_CONDITIONS_MISMATCH.value,), (
                f"{stated}no power rating of {t.device_id} applies to the beams that reach it ({see})"), basis=entry)
        if element.rating_status == "CONDITIONS_NOT_VERIFIED" and not applying:
            return Utilization(ratio, Status.UNKNOWN, (ErrorCode.POWER_RATING_CONDITIONS_NOT_VERIFIED.value,), (
                f"{stated}the power ratings of {t.device_id} cannot be checked against the light that reaches it "
                f"({'; '.join(kinds)}; {see})"), basis=entry)
        if reasons:
            return Utilization(ratio, Status.UNKNOWN, codes, f"{stated}the utilization of {t.device_id} is not bounded: "
                               f"{'; '.join(reasons)} ({see})", basis=entry)
        return Utilization(ratio, Status.UNKNOWN, codes, f"{stated}the exposure of {t.device_id} is not known ({see})", basis=entry)
    if not reasons:
        reasons.append("the exposure of a rated surface is known as a range only")
    reported = _reported(exposures, t.device_id)
    # (no exposure known from below: no number, never 0 -- graded on the interval)
    value = None if reported is None and lower == 0.0 else min(max(lower, reported or 0.0), upper)
    least = (f"the utilization on {t.device_id}'s rated surfaces is at least {lower:.6g}" if lower > 0.0 else
             f"the utilization on {t.device_id}'s rated surfaces has no lower bound above 0")
    if value is not None and value > lower:  # (the value reported: what the exposure gives against the ratings that apply)
        least = f"the exposure on {t.device_id}'s rated surfaces reaches {value:.6g} of a rating that applies; {least}"
    message = f"{least}: {'; '.join(reasons)} ({see})"
    return Utilization(value, Status.PASS, _codes(exceeds, codes), message, (lower, upper))
