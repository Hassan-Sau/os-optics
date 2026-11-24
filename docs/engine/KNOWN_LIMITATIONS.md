# Model limits of os-optics

What the models of the Engine do not compute, and the validity range of each model. A limit that can occur in a run is
reported there, at the quantity it affects (an issue, a state word, a `value_basis` entry); this file states the model
behind it. Section letters are kept so that references to them stay valid.

## B. Physics model limits

* **Ideal thin lens** is aberration-free with constant `f`; real-lens aberrations only appear through `singlet_lens` (exact spherical refraction) for *rays*;
  the Gaussian layer is paraxial (Coddington) for every element.
* **Surfaces are planes, spheres and sag profiles** (conic + even asphere, rotationally symmetric or cylindrical); apertures are circle / rectangle / ellipse / D shape /
  simple polygon, optionally decentred (+ central circular obstruction on a circle), with an opaque mount ring or an explicit mount outline, and
  plane and sag surfaces may have holes through the element (no surface inside them). A beam whose chief ray passes through a hole goes on with its
  power; the part of its footprint on the surface around the hole is light beside an element (not taken out of the beam). The 3D view draws
  every clear outline (ellipse, D shape, convex or non-convex polygon) and every hole within its drawing tolerance: a surface with holes or a
  non-convex outline is triangulated over its regions (clear aperture minus holes, the rim out to the glass outline) and refined on its
  profile; a surface with its own mechanical edge is drawn to its regions out to that edge and then as its flat flange (holes open). A lens array
  (`CompoundMemberV1.array`) is its cells as members of one unit: a beam is traced through the cell its chief ray meets, the share over other
  cells goes on beside it as for any element; a beam spanning many cells needs sampled light (an extended emitter), and the paraxial analysis
  refuses an array (its cells lie off the unit's axis). A gradient-index element is a radial
  GRIN rod (`n(r)^2 = n0^2 (1 - (g r)^2)`, rays in closed form) or an axial one (`n(z) = n0 + a1 z + a2 z^2 + a3 z^3`, rays by
  Gauss-Legendre quadrature of their path integrals), between two end faces that are planes across the axis, spheres or conics
  centred on it, or planes at any slant on a round body (the Gaussian beam by its first-order transfer about the chief ray, out
  of model scope for a skew chief ray with an astigmatic beam -- a beam off the plane of a slanted face's tilt); a combined
  radial and axial profile, other profiles, aspheric or decentred curved faces, light reaching the rod's side, an element or
  observation plane inside the rod, a ray in an axial medium more than acos 0.05 (87.1°) from the axis (near turning back; close to that limit the leg's fixed
  quadrature loses accuracy: at a cosine of 0.05-0.06 at the least index the exit point may be off by up to 0.3 mm on a 300 mm leg and
  the optical path by 0.9 waves on a 10 mm leg; at 0.1 by up to 1.5e-3 mm and 0.3 waves on a 300 mm leg; from 0.3 on below 1e-9 mm) and a ray whose slope times the
  face's largest slope reaches 1 are outside the model (`GRADIENT_INDEX_UNSUPPORTED`; part of a beam doing so -- a pupil ray,
  or the beam's envelope at its negligible-share radius -- puts the beam's power behind the rod out of model scope); the ray-path export draws each pupil ray inside
  the rod as the chord between its end-face points (the chief ray's segments follow its curved path, and the local runtime's
  edge check reads them as one run from end face to end face); the envelope delay's
  spread across a beam is not bounded behind a rod; the return-light bounds read the index on the axis (the largest). A
  crystal element is a `SURFACE_SEQUENCE` or `SURFACE_SET` model of plane faces whose crystal medium states its optic axis (below). Catalog solids (cubes, prisms, wedges: public catalog `SURFACE_SET` models) are their traced faces and internal coated planes inside a convex
  body; light reaching a body face that is not a traced surface leaves the model and the local runtime reports it (`RAY_LEFT_ELEMENT_THROUGH_EDGE`,
  `RAY_CROSSES_ELEMENT_BODY`).
* **Per-surface edges: blocked light ends on the continued profile.** A `SURFACE_SEQUENCE` surface with its own edge
  (`edge_radius_mm` / `edge_half_widths_mm` on the surface) is a flat flange at its rim height from that edge out to the outline
  of its glass in the optical-solid check, the section and the 3D view. The trace blocks light over the same zone (from the clear
  aperture to the glass outline) on the surface's profile continued past its edge, so a blocked ray ends off the flange by the
  profile's sag beyond the edge: for an R 20 mm sphere edged at 9 mm, 0.26 mm at 9.5 mm, 0.54 mm at 10 mm and 0.84 mm at 10.5 mm
  from the axis. Light parallel to the axis is blocked at the same radii either way; an oblique ray within that sag of the glass
  outline can meet one face inside the outline and the other outside it.
* **Cavities: the analysis computes the resonant build-up; the run traces single passes.** `cavity_analysis` gives a linear cavity's
  stability, fundamental Gaussian eigenmode, Gouy phase, free spectral range, transverse-mode spacing, finesse and linewidth, and with
  `injection` the steady state of light fed through its first mirror (circulating, transmitted, reflected and lost powers, per
  Hermite-Gauss mode of a coaxial input beam; PHYSICS_SCOPE P10f). A run does not: light through a mirror is not traced (a beam meeting
  the back of a transmitting mirror ends with `OUT_OF_MODEL_SCOPE` (cause `MIRROR_TRANSMISSION_NOT_TRACED`)), a path between two mirrors stops at
  `max_interactions_per_path` (`MAX_INTERACTIONS`, a partial outcome), and a target with a cavity metric is `OUT_OF_MODEL_SCOPE` in the
  run (the analysis assesses it). Ring cavities, astigmatic cavities (curved mirrors off normal incidence), a tilted or offset input
  beam (odd transverse orders) and the coatings' own reflection phases are not modelled; a plane-plane Fabry-Perot etalon has no
  Gaussian eigenmode (its build-up is `OUT_OF_MODEL_SCOPE`). The content of a source with `M² > 1` in the cavity's modes is not
  decomposed and a pulsed or broad-linewidth source is not averaged over its spectrum: their powers are bounds
  (`SOURCE_NOT_FUNDAMENTAL_MODE`, `SOURCE_NOT_SINGLE_FREQUENCY`), not values. A source's beam is injected only when nothing but
  observation planes lies between it and the input mirror, whose substrate faces are spherical or plane: with a stop or a lens on
  the way the powers are bounded by the traced arriving power and the mode matching is not formed (carrying the beam through
  coaxial elements before the mirror is open), and where part of its beam meets an element beside its chief ray the arriving
  power is not known. The detuning must be stated (`detuning_Hz`, or `length_offset_nm` from the resonance): a Project's positions do not fix it,
  and without it the powers are bounded over every detuning. Losses of intracavity elements enter only as a stated `round_trip_loss`
  (without it the finesse is an upper bound and the injected powers are bounded for any loss).
* **Gaussian model domain**: paraxial/moderate-NA, untwisted astigmatism only. Twisted astigmatism, anisotropic `M²` with rotated axes, severe truncation, high NA,
  strongly curved or steeply tilted surfaces, and arrivals whose pupil rays spread more than `gaussian_max_ray_spread_ratio` (2) times as wide as any
  aberration-free beam with that Gaussian envelope (`GAUSSIAN_ABERRATION_DOMINATED`, PHYSICS_SCOPE P4.2b) → `OUT_OF_MODEL_SCOPE` (no numbers). Below that
  limit the Gaussian numbers are paraxial and carry the aberration unflagged (the real spot may be up to about `√(1 + r²)` times the Gaussian radius). Diffraction from apertures, pinholes and spatial filters is not propagated.
* **Chief ray + deterministic pupil samples only.** Where the chief ray meets a physical surface or an absorbing plane (on its active area), the pupil rays
  that would first meet another such surface are retired (`diverged`); a transparent observation plane never retires a ray. The packet's power and Gaussian
  beam stay those of the chief's path, so a beam partly falling on a physical surface its chief ray misses is delivered at full power downstream. The ray-path
  export records the surface each retired ray would have reached (`next_surface_id`, `counted_with_chief`); the local catalog runtime turns that into
  `BEAM_PARTIALLY_INTERCEPTED` (out of model scope from there on). The low-level `validate` marks it where it shows: an observation plane the chief ray crosses
  on the leg after a live pupil ray met a physical surface (or after the Gaussian share that meets the surface the chief ray meets next exceeds the line's
  negligible level, as a monitor just before a fold mirror; or after the Gaussian share that a reflection the leg began with sends beyond the plane, as a
  monitor just after a fold mirror) has its power out of model scope there (`power_out_of_scope_at`: that surface), and an absorbing observation plane
  that takes part of a beam its chief ray does not cross makes that beam's power out of model scope from there on in both. Where the surface met first is
  the one the chief ray meets next, an absorbing plane, or the reflecting surface the leg began at (for the part of the beam it sends beyond the plane),
  and its share has a supported bound `S` (the larger of the parallel-ray and the paraxial energy-flow pictures; the energy flow's lines are taken
  where each meets the surface, also beyond the plane's chief-ray crossing and behind the chief ray's start, and a sphere the leg began at also as its
  tangent plane with the cap's departure from it), the plane reads the beam within `[P - S, P]` instead (`power_taken_before_at`,
  `power_taken_before_upper_W`, plane `BOUNDED`, `POWER_TAKEN_BEFORE_PLANE`): a monitor 2.5 mm before a 45 deg fold of a 1 mm beam reads 1 mW
  within 3e-7 of it, a monitor 0.1 mm after a 45 deg fold of a 0.5 mm beam reads 0.99 mW within the 0.34 mW the mirror sends beyond it. Where the
  surface the chief ray meets next stops what it meets from the beam's side (the back of an uncoated mirror, an absorber), the share whose
  energy-flow lines certainly meet it first before the plane, `L`, narrows that to `[P - S, P - L]` (`power_taken_before_lower_W`): a 0.3 mm beam
  1.2 beam radii above a bowl's near rim (0.82 % of it meets the rim's back) reads at most 0.9937 mW (a sag bowl) / 0.9942 mW (a spherical
  cap), so a 0.995 mW minimum is unmet. A surface with holes states no `L` (light that reaches a hole passes it), and no `L` is counted where
  another surface counted before the plane only turns light (a lens or a mirror beside the chief ray: a line it turns may cross the plane after
  all; a thin lens beside a 0.3 mm beam turning 5.9 % of it over a dump gave a false UNMET). An absorbing plane reads its detected power `D` within
  `[D - S, D]` (the light a surface takes before it lowers the share on its active area by at most `S`; the share certainly taken may be light
  that would have missed the area, so no `L`): an absorbing monitor 0.03 mm before a 45 deg dump that takes 42 % of a 0.3 mm beam first had
  read the whole beam. A beam outside the plane's acceptance cone (received 0) adds no share taken. The bound `S` is wide where a
  sphere's whole cap counts (the near rim of a steep cap): up to the whole beam, `[0, P - L]`. A surface beside the chief ray that pupil rays meet
  first is bounded by the cells of the beam's profile whose energy-flow lines may meet its support before the plane (a 3 mm beam whose chief ray
  passes 0.01 mm beside a window's edge: 0.494 of the beam for its front face against 0.474 meeting the window; its back face, which the
  beam's tail may reach on the same leg, adds its own bound, so `S` is 0.989); a support of another shape is taken as the circle about its offset that holds it. A face of the body the
  beam travels in that pupil rays meet first still puts the plane's power out of model scope. Where the beam has no valid Gaussian state, a
  reflecting surface the leg began at whose support may reach beyond the plane, and a surface the chief ray meets next whose support may reach
  before it, put the plane's power out of model scope (its pupil rays do not show such a beam's extent: near a waist they lie well inside it; a
  monitor 0.25 mm after a 77 deg out-of-plane fold of a twisted-astigmatic beam, 17.8 % of which leaves the mirror beyond the monitor, had read
  the whole beam). Such a beam's plane is read only where those supports lie wholly on the far side.
  Light that leaves a transmitting surface beyond the plane is not checked (it crossed the plane on its way there, a partial crossing of the leg before,
  unless it lay beyond the plane all along). A physical surface beside the chief ray's path that no pupil ray meets is weighed too where the beam
  has a valid Gaussian state: every surface whose bounding balls come within `sqrt(2) K` beam radii of the leg (not wholly behind its start or
  beyond its end: a plane surface by its support's own extent, a curved one by its balls) gets the cell bound above; above the line's negligible level it counts in `S` at a plane on the leg and, at the leg's end (the
  surface the chief ray meets, an obstruction included), puts the beam's power behind out of model scope (an `OUT_OF_MODEL_SCOPE` issue with the
  bound, `tail_fraction_upper`). Launched on a converging beam the pupil rays close toward its waist and pass well inside the envelope there: a
  dump disc whose edge passes 0.05 mm beside the chief ray 18 mm past a 0.088 mm waist takes 23.5 % of the beam's energy-flow lines before a
  monitor while no pupil ray meets it; the monitor had read the whole beam and now reads `[P - S, P]` with `S` 0.47 of the beam, and the beam
  behind the next fold is out of model scope. Not weighed this way: the faces of the body the beam travels in, the other surfaces of the elements
  the leg begins and ends on (light reaching them first passes beside the surface met, whose clear aperture books it), a beam without a valid
  Gaussian state (its pupil rays decide) and the annulus behind a central obstruction (its pupil samples decide, below). The threshold is the
  line's negligible level (`negligible_power_fraction` of the line's power, by default 1e-9, a Project setting), not a target's tolerance: an element edge closer than about
  3.2 beam radii to the beam puts the power behind the next surface out of model scope (a side mirror's edge beside a 1 mm beam at 780 nm: at 2.0 / 2.5 /
  3.0 radii, where the beam's tail it clips is 3.2e-5 / 2.9e-7 / 9.9e-10 of the power, the focal-plane power has no value; at 3.25 radii and
  beyond it is stated). Keep elements beside the beam about 3.2 of its radii clear of it. A curved surface beside the leg is tested by
  balls that hold it, so one closer to the leg than its own size may be weighed although the light cannot reach it before the plane: the
  cell bound then states its share (often 0), or a loose bound that puts the power behind out of model scope. A plane through
  a physical surface's chief-ray hit point (within `ray_min_advance_mm`, 1e-7 mm) records the power arriving at that surface before the interaction, and
  each such arrival says so (`surface_coincidence`: `surface_id`, `role` `ARRIVING` or `RETURNING`, `surface_event`; a power target that selects it carries
  `SURFACE_COINCIDENT_ARRIVAL` and reads the arrival; `docs/engine/RESULTS_AND_EXPORT.md`). The same plane 1 nm before the hit point is outside this window and reads the
  beam within the share the mirror meets first (about half the beam at a 45 deg fold). Angle-of-incidence verdicts judge the chief ray only: the local
  runtime's coating / element angle ranges and the Engine's grazing, TIR-margin and thin-element warnings (a pupil ray beyond an Engine coating table or the
  grazing limit is retired silently), so a converging beam's marginal rays may meet a coating outside its range unreported; marginal stray light, ghost images (beyond one optional Fresnel reflection branch per interface), multiple reflections inside elements, scattering and full stray-light Monte Carlo are not modelled in a run. `os-optics results ghosts` lists a run's interface-reflection ghosts up to four bounces (focus, relative power, relative irradiance) with a power bound on the rest; that bound covers every plane, so on an image plane it also counts light travelling back toward the source, and it bounds power, not irradiance. It is proven where every ghost surface met a valid Gaussian beam on a plane or a sphere (its reflectance and transmittance bounded over the beam's angles of incidence); where a ghost surface met a beam without a valid Gaussian state, an annular beam or a sag surface, it rests on the beam's pupil rays and is not proven (`NOMINAL`), as in a double-Gauss lens whose steep inner faces invalidate most ghost beams. Scattered light enters the bound, its own ghosts are not enumerated. A listed ghost's power takes its chief ray's reflectance, and ghosts add to the image in power (no interference with the primary light).
* **Partial clipping** at an aperture reduces the chief-ray power by the analytic Gaussian fraction; the clipped remainder is lost (no branch). A beam whose chief ray is
  on the opaque rim beyond the clear aperture is terminated (`CLIPPED`, structured issue with AOI, projected aperture and centre-to-edge distance) even if its Gaussian tail
  would partly pass; that tail is not traced on. Such light (and untraced reflections / coating transmissions, light passing beside an element's support, a beam's tail
  on an active area it is not counted on) is not part of any plane's power, but a plane no traced beam reaches is not a known 0 W while it may arrive there: each
  such branch is followed through the optics by a probe trace without pupil rays, and its own ghosts recursively, to every order above `negligible_power_fraction` x
  the line power (PHYSICS_SCOPE P10). Where the probes' trace budget cannot follow the cascade (several uncoated lenses) the dark planes stay `UNKNOWN` with a
  `MAX_PATHS` issue; a run with many such optics and a dark plane spends up to the full step budget on it. The tail of a beam whose chief ray misses a surface
  its outer rays meet (`DIVERGED`) is not part of the check. Light inside an element's own medium (a medium of its surfaces other than the surround) that falls beyond
  a surface's support is booked as the element's edge loss, not followed: an element whose surfaces bound a medium that also fills the space outside it (a
  window that is the wall of a liquid cell) is therefore not covered by the check for that light.
* **Vector focal field.** `results focal-field` models an ideal aplanatic objective (no aberration at any field point, sine condition) with an optional stated
  Zernike phase and one plane-parallel window (single pass, its internal reflections not included). A real objective's prescription, a measured pupil or point spread
  function, several refractive-index interfaces and field-dependent aberrations are not modelled. The Debye integral is used within a Fresnel number of at least 100
  and a focal region within 0.1 of the reference sphere's radius; outside them the focal quantities are `OUT_OF_MODEL_SCOPE`.
* **Imaging analysis.** `results imaging` is monochromatic per field, for a uniformly lit pupil (a point object), and traces the run's request again with a dense pupil grid.
  A source beam whose chief ray an aperture cuts ends there, so such a field reports `NO_ARRIVAL`; a beam that fills the stop around its chief ray arrives (on the
  geometric pupil-sample basis when it has no valid Gaussian state) and is analysed. Extended emitters are not fields. Partial coherence and image simulation are not modelled.
* **Central obstruction.** A chief ray on a central obstruction transmits the annulus with its exact Gaussian power; behind it the annulus is traced as geometric
  rays with the chief ray as reference axis, bounded paraxially by the envelope of the unobstructed Gaussian beam and by the image of the opening that cut it.
  Its Gaussian metrics and the phase on the axis are out of model scope, diffraction by the obstruction (annular PSF, Poisson spot) is not modelled, and when a
  later surface, aperture or active area takes only part of the bounds or of the annulus's rays, its power is not followed (`ANNULAR_BEAM_NOT_FOLLOWED`): a
  second obstruction, a detector in the shadow or one smaller than the bounds give `UNKNOWN`, not a value. Without rays (an obstruction wider than the sampled
  rings, control readings, probe traces) the bounds alone decide, and a surface beside the axis that catches the outer annulus between two surfaces is not seen.
  An element inside the beam whose back the chief ray meets (a Cassegrain secondary) is such an obstruction: the beam beside
  its support goes on as an annulus, and a later surface's hole is crossed by the annulus's reference axis on the surface's
  continued profile when the annulus rays' affine map proves the hole inside the element's shadow (else the annulus is not
  followed). A mirror's back, an absorber that does not scatter and the opaque plate of an aperture stop beside its opening
  (when the opening passes a negligible share) start this; a lens inside a beam does not (the light beside its support is an
  untraced branch whose planes read bounds), and a scattering absorber or a stop whose opening passes light keep their own
  rules. Only for a traced beam with pupil rays and a valid Gaussian state. An element with holes of its own proves no later hole dark (light
  through its holes goes on inside its outline): the annulus is not followed past a later holed surface (an absorber with a hole in line with a
  fold mirror's hole had stated the whole annulus, 1.7 % high). A curved support (sag surface, sphere) met by the chief ray more than 1e-7 (sine)
  off its axis puts the annulus's power behind it out of model scope (an `OUT_OF_MODEL_SCOPE` issue naming it): the share used is the support
  mapped onto the plane at the chief ray's hit, which is its shadow along the beam only along that axis (a convex cap of R 12 mm met at 20 / 35
  deg: +1.8 / +4.6 % on the power passing beside it); the ledger books that mapped share. A pupil ray on the element's support meets it wherever it hits it, so a beam decentred on a
  curved secondary goes on as its annulus (exact to 5e-8 for a 15 mm beam up to 5 mm off a 13 mm secondary); where the
  unobstructed beam's envelope reaches past a later clear aperture (the same telescope 7 mm or more off the axis, by a 1e-9
  share) the annulus is still not followed. A pupil ray beside the support that meets another element first leaves the
  annulus, whose power still counts its light: that element joins `power_out_of_scope_at` (an `OUT_OF_MODEL_SCOPE` issue), so
  the annulus's power behind it is out of model scope. A beam whose chief ray passes beside such an element while some of its pupil rays meet
  it (and would reach the next surface inside its clear aperture) is not an annulus: the beam's power follows its chief ray,
  so its power behind that element is out of model scope (`power_out_of_scope_at`, an `OUT_OF_MODEL_SCOPE` issue with the
  share of its pupil samples and the Gaussian share on the element's support), never a known value. Between and beyond the
  pupil samples the beam's tail is weighed by the cell bound (a surface beside the chief ray that no pupil ray meets, above),
  also on the leg to an obstruction. Behind a central obstruction only the annulus's pupil samples find an element beside it:
  one between or beyond them is missed (for a straight edge about `0.5 erfc(√2 ρ cos(π/M))` with `ρ = √(ln(2K) / 2)` the
  outermost ring's radius in beam radii, `K` rings, `M` points per ring: 1.14 % at the default 8 x 12; more rings and points
  narrow it, `compute_budget.pupil_ring_count` / `pupil_points_per_ring`), and an element smaller than the samples' spacing
  may be missed whole.
* **Geometric pupil-sample power.** A beam whose Gaussian state is invalid (severely truncated, a surface too strong for the paraxial model) and whose
  pupil rays a later clear aperture cuts goes on with the share of its equal-power pupil samples that pass (`GEOMETRIC_POWER_BASIS`,
  `power_sampling_resolution_W`): a sampled value, not a proven interval; the default 8 x 12 samples give a resolution of a few to tens of percent behind
  a deep cut (more pupil samples narrow it: `compute_budget.pupil_ring_count` / `pupil_points_per_ring`), the live samples are taken to share the power that reaches the aperture equally, an edge farther than
  one sample spacing beyond the outermost live samples is taken to cut nothing of their cells (one nearer adds their shares to the resolution), and edge
  diffraction is not modelled. An edge that holds the image of an opening the beam passed earlier (the same opening met again, or one that contains it,
  through the samples' affine map between the two surfaces with its residual as margin) cuts nothing and adds nothing. Where the beam's diffraction spread the rays do not follow exceeds a quarter of the radius they picture (a small beam far
  from its waist, a Fresnel number below about 1.3) and an edge comes within the beam's reach, or where no pupil sample falls on a detector's active area
  the chief ray meets, the power is not computed and the path ends (`TRUNCATED_BEAM_NOT_FOLLOWED`; the planes behind it are `UNKNOWN`). A source
  launched beyond the Gaussian scope (far-field half angle above 0.3 rad) has no pupil samples, so an absorbing active area withholds it whatever its
  size: no containment bound is formed for it (its angular spectrum keeps `exp(-2 sin²α / θ²)` of the beam beyond the angle α, about 6e-6
  beyond 79° at θ = 0.4 rad, far above the negligible level). A plane's sampled
  power is `KNOWN` with `power_sampling_resolution_W` beside it and states no upper bound taken from it. An aperture that cuts every pupil sample while the
  chief ray passes ends the path (`TRUNCATED_BEAM_NOT_FOLLOWED`). A beam whose chief ray falls on an aperture's rim while some of its pupil rays pass goes on as
  those rays, the one nearest their centroid its reference ray from there (`reference_ray_promoted_at`): the positions, directions and optical path reported
  behind it are that ray's, its first-order transfer is not reported, and its group path keeps the chief ray's up to that rim. That ray is one sample of the
  light that passed (on a tie, one of a mirror-symmetric pair): a target on the chief-ray hit point, direction cosines, angle of incidence or acceptance margin
  behind it is out of model scope (`CLIPPED`); the spot centroid (of the pupil samples that pass) gives the beam's position there within the samples' resolution. Where no pupil ray passes
  (a small beam beyond the edge) it is `CLIPPED` whole and the light inside the opening goes on as untraced light only. A beam whose chief ray meets the
  back of a mirror whose support lies inside the beam (a Cassegrain secondary in front of the primary) goes on beside it as an annulus (see
  **Central obstruction**). Where that rule does not apply (no pupil rays or no valid Gaussian state, as behind a promoted reference ray, or
  a negligible part of the beam beside the mirror) the path still ends there (`BLOCKED_BY_BACK_SIDE`) and the light passing beside
  that mirror goes on as untraced light only. The pupil sampling is a Project setting (`compute_budget.pupil_ring_count` / `pupil_points_per_ring`, up to 64 x 256).
* **Plate beam splitters, polarizers and waveplates are thin ideal elements** (no thickness, no wedge, no dispersion of the coating between spectral lines beyond a
  `(λ, AOI)` table). Catalog cube splitters (non-polarizing and polarizing) are solids: their faces refract, their coated diagonal splits per polarization
  inside the glass; the cement layer of a cemented cube is not modelled. Waveplate retardance is zero-order `δ₀λ₀/λ` unless flagged achromatic; oblique incidence beyond 20° only warns.
* **Polarization** is Jones calculus. A partially polarized or unpolarized source is traced as two mutually incoherent orthogonal Jones components
  whose Stokes parameters add on each route of a plane (exact for every element the Engine models: none depolarizes the beam it passes on; a
  scattering surface sends its scattered share out as separate unpolarized rays, C2); two-beam quantities
  are formed within one component, and the fringes of the mixture of both are not computed; with another source of its coherence group only its
  polarized part is coherent. No depolarizing element or stress birefringence; the
  linear optical network export takes fully polarized sources only.
* **Birefringent crystals** are uniaxial with plane faces (`UniaxialMaterial` in the Engine request's materials): biaxial crystals, optical activity
  and curved crystal faces are outside the model. A face's split between the ordinary and extraordinary waves is exact at normal incidence and where
  the optic axis is perpendicular to the plane of incidence, approximate elsewhere (warned beyond 2°). Light whose wave normal lies within 1e-6
  (sine) of a crystal's optic axis, where the two waves coincide, crosses as one wave of index `n_o` carrying its whole field (a rolled Rochon
  prism had split it on a basis the crystal does not define: +646 % / -46 % in its two beams at a 30 deg roll; now each beam equals an exact
  plane-wave solution to 1e-12 at every roll tested; a contacted pair at some rotated poses -- 10 to 39 of 72 roll angles for the catalog
  and declared Wollaston and Rochon -- is refused `OPTICAL_BODIES_TOUCH` by the solids check, which does not read the joint), and a nonlinear conversion of such a fundamental is out of model scope. Just outside that band the
  two waves are split on the crystal's own basis and traced apart. Where a later crystal face splits each of two such waves (both above the
  line's negligible level) again into two waves -- each of its waves then fed by both, with a relative phase the trace does not follow, as in a
  Rochon or Wollaston prism tilted off its plane by 2e-6 to 1e-3 rad at a skew azimuth (+646 % / -46 % in its two beams by a power sum) -- the
  power behind that face is out of model scope (`power_out_of_scope_at`, an `OUT_OF_MODEL_SCOPE` issue). A face that passes each wave into one
  wave of the next crystal (a Wollaston or Rochon in its own plane, a single crystal displacing its two waves) is traced as before. Two
  such waves that later meet a surface passing their polarizations by different powers -- a polarizer or analyzer, a coated interface,
  mirror or beam splitter whose s and p values differ, an interface met at an angle, a diffraction grating -- above the line's
  negligible level put the power behind that surface out of model scope too (a crystal half-wave plate before a polarizer had read
  3.62e-4 W where about 8e-7 W arrives, the same at every thickness). Whether the two waves still overlap there is not tested: the
  separated beams of a displacer crossing one polarizer are out of model scope as well. A surface at normal incidence without a
  polarizing coating, a retarder and a plane only add their powers, which is exact for two orthogonally polarized waves. An extraordinary Gaussian beam is carried with
  one diffraction index within a stated error. A Project declares a crystal as a material with `birefringence` and the optic axis on each model
  made of it (`optic_axes`); a cemented crystal pair (Wollaston, Rochon, Glan-Thompson) is a `COMPOUND` unit of two prisms joined face to face
  (`joints`): an optical contact is one interface between the two crystals, a cement layer a uniform isotropic layer between
  parallel faces whose thin-film interference is not modelled (nor is that of an air-spaced pair's gap, as in a Glan-Taylor or Glan-laser
  prism: with coherent light a 20 um gap's passed-wave transmission ranges over 0.816-1.000 at 633 nm against the 0.901 traced); a joined
  face carries no coating. Where a crystal's optic axis lies neither in nor across the plane of incidence the per-wave face rule
  underestimates the rejected polarization's leakage (a Glan-Taylor 8 deg skew: 1.4e-8 W against 1.0e-6 W; a Rochon 3 deg: 1.7e-15 against
  1.7e-9 W), flagged `BIREFRINGENT_INTERFACE_APPROXIMATE`; such skew incidence also puts the beam's Gaussian state out of model scope. Behind a joint the local runtime's edge check
  reads the light as inside the next prism's crystal or the cement layer, bounded by that body's faces (light leaving such a body through its side is
  `RAY_LEFT_ELEMENT_THROUGH_EDGE`). A catalog polarizing prism (catalog schema 3.5, `JOINED_BODIES`) is traced the same way; `catalog_model`
  does not copy it into a Project as one declared model. The linear optical network export and the assembly routing do not follow light through crystals.
  The paraxial analysis refuses an element holding a crystal; the cavity analysis analyses one eigen-polarization (`polarization`) where every
  crystal's optic axis lies in its faces, its extraordinary eigenmode where the axis lies along a section axis and the crystal's element has no
  power. The return-light bounds read a crystal face over the crystal's two indices and widen the directions of light refracted there.
* **Sources** are Gaussian beams (discrete spectral lines, mutually incoherent, or a continuous `spectral_density` traced as quadrature
  lines: a power summed over them is the midpoint quadrature of its band, its convergence stated for the source's coherence, not
  re-traced at twice the lines; light of different lines adds, and two beams of one continuous source interfere with the spectrum's
  coherence at their group-path difference, the beams' balance, overlap and polarization taken at each line) and uniform
  incoherent planar emitters (discrete lines: an emitter's `spectral_density` is refused with the sampling it takes); sampled sources
  (`SAMPLED_INTENSITY`, `SAMPLED_FIELD`, `SPECKLE`, P8b) are traced as rays whose powers add. A sampled field's or a speckle field's
  coherent propagation is `os-optics results field`: scalar; the exact angular spectrum only through free space to a parallel plane,
  otherwise the first-order (Collins) transfer of the source axis, without the elements' apertures, aberrations beyond first order or
  action on polarization (checked conditions; outside them no values: among them, the rays' aberration beyond first order,
  applied to the field as a phase, changes the plane's profile values by at most 0.5 % and its centroid by at most 1 % of its rms width; near an
  image waist this keeps an accepted plane within 1 % of an exact diffraction reference over 405-1064 nm, within 0.97 % at 1310 nm and up to
  1.03 % at 1550 nm: a 0.72 mm beam 2.5 mm past the focus of a 50 mm plano-convex lens). Its rays need four grid points per 2 pi of phase
  (`SOURCE_FIELD_UNDERSAMPLED`); a speckle screen is Gaussian to the central-limit approximation; an intensity map's angular law is
  the same at every point; grids written in the Project hold at most 256 points per axis.
* **Time** enters only as delays on the static trace (`group_delay_ps`, `phase_delay_ps`, `time_delay_ps`: envelope and carrier
  delays, no pulse shape or dispersion broadening). Vibration, timing noise, time series and feedback are refused: a Project is
  evaluated at one instant.
* **Interference** is a *pairwise* analytic two-beam model (plane-wave phase with Gaussian envelopes at the chief-ray axial position). Coherent sums of three or more beams
  are not computed; curvature of the wavefront across the overlap region is ignored (a warning fires when the footprint spans > 10 % of the Rayleigh range);
  coherence is `|γ| = exp(−(OPD/L_c)²)` with a user-supplied `L_c` or Gaussian linewidth — multimode / non-Gaussian spectra are not modelled.
  Irradiance is the transverse-flux intensity `|E|²` normalized to the beam power (relevant to thin-resist absorption), not the Poynting flux through the substrate plane.
* **Materials and media**: every material is evaluated on the vacuum wavelength by one rule (a law on air wavelengths at
  `λ / n_ref(λ)`, an index relative to air times `n_ref(λ)`, `n_ref` the source's reference air). A native Project operates in
  its environment (air at a temperature and pressure, default 20 °C / 101325 Pa; zero pressure is vacuum; a declared model's
  gap may be `VACUUM`); the Engine-level snapshot's built-in `air` stays `n = 1` (the native lowering always states the
  environment's air). The air model is Edlén (1953) with OpticStudio's temperature and pressure scaling, fitted over
  200–2000 nm: beyond it, in particular in the thermal infrared, it is an extrapolation (about 1–2 % of `n − 1`), not a
  measured infrared law of air; humidity and CO₂ are not modelled. A source that states no convention is evaluated as vacuum
  wavelength / absolute index and reported as `NOT_STATED`.
  Temperature: Schott dn/dT and the CHARMS temperature Sellmeier law (Si, Ge: 20–300 K, mid-wave only) are applied at the
  environment temperature; a material without thermo data is evaluated at its source temperature. Thermal expansion of
  elements and mounts (athermalization) is not modelled: a material's thermo-optic response is not a lens's.
  Bulk loss: measured `k` or internal transmittance on its own wavelengths; outside them the loss is not established
  (`BULK_LOSS_NOT_ESTABLISHED`, transmitted power is an upper bound). A material that states no loss at all is traced without
  bulk absorption and is not flagged in a run; `os-optics catalog material ... --thickness-mm` reports its transmission as
  not stated, never 100 %. Group index and group-delay dispersion are analytic for formula laws; a measured table gives no
  group delay (its derivatives are not established). Tabulated data are interpolated (PCHIP/linear) inside their validity
  band only.
* **Observation planes** are virtual and unbounded: each records every live ray of a beam that crosses it before the surface that would stop or turn that ray,
  where that ray crosses it, whatever the other planes and the declaration order (planes through one point, coplanar, opposite or tilted). Adding a
  transparent plane changes no other plane's, target's or exposure's result; it is one more trace step (`max_trace_steps`: a budget-limited run may stop
  elsewhere), and an escaping path's recorded length ends at the chief ray's last event, which the plane may be. An absorbing plane with an active area
  stops a ray only where the ray crosses it on that area. A beam's power follows its chief ray, so:
  * a plane part of a beam reaches on a straight leg its chief ray does not cross the plane on (before a tilted detector or absorber, a fold mirror, a lens)
    gets no value from that beam, only a bound: the Gaussian beam's share that meets the plane after the leg's start surface, before its end surface and on
    its active area, the beam taken as rays parallel to its chief ray (`PARTIAL_CROSSING`, `UNTRACED_LIGHT_BOUNDED`). A plane surface is taken as its plane;
    a curved one as a plane shifted as far as the surface reaches -- beyond for an end surface, back for a start surface -- for the rays within `K` beam
    radii (`K` makes the share beyond them an eighth of the line's negligible level): a sphere as the plane across the ray, by its nearest roots where those
    stay inside the cap's aperture, else over its whole cap; a sag surface (asphere, cylinder, axicon, off-axis paraboloid) as its tangent plane at the
    chief ray's point, shifted by how far its profile departs from it over the part of its support those rays can meet (a bound of the profile's
    curvature there; across the axis' plane, as on an off-axis paraboloid's reflected leg, the part the hit that continues the chief ray's can reach --
    another hit of the same ray, at a fold of the surface across it, is not taken into account), else as the plane across the ray with its sag range
    there or its support's extent. Near a surface that bulges away from the light the bound is wide (up to the whole beam), and the curvature bound is
    coarse on a very steep profile (a monitor 2 mm after a 6.35 mm off-axis paraboloid reads the beam within 2.1e-7 of it for a share of 8e-12); only a sag surface without a
    bounded support leaves the bound to the pupil rays. The surface the chief ray meets next, for the share of the beam that meets it before a plane
    the chief ray crosses (above), is taken the same way, except that a sag surface's tangent plane is used only where the part of its support those
    rays can meet bounds every hit of theirs: across the axis' plane it is the plane across the ray before every point of that part, and the share
    such a sag surface takes is then bounded on the surface itself, cell by cell of the beam's profile (the smaller bound counts). So a beam whose
    edge a steep sag mirror's near rim clips at oblique incidence reads at a plane in front of the mirror's far side within the share the rim may
    take, and an off-axis paraboloid used as a collimator (the beam from its focus side) reads a monitor on its incoming leg: for every off-axis paraboloid of the catalog
    (monitors 0.5-8 mm before the hit) every monitor from 2 mm reads the beam, at 0.5 and 1 mm it reads within a bound of at most 3.2 % /
    0.015 % of the beam that holds the share the mirror meets first (a 0.85 mW minimum on 1 mW met). A monitor on its collimated leg reads the beam
    from 2 mm; at 0.5 / 1 mm after the hit it reads within a bound of at most 36 % / 2.7 % of the beam that holds the share the reflection sends
    beyond it (the start surface's stand-in plane is coarse on these steep profiles: up to 13 / 260 times that share), so a 0.85 mW minimum on 1 mW is
    met at 73 / 79 of the 86 off-axis paraboloids and not decided at the others. A sphere is taken by the root of each ray nearest the chief ray's point, except where
    the chief ray's line crossed the sphere before its hit and the cap may lie within the envelope's reach of that crossing (the near rim of a steep
    cap the beam passes at oblique incidence): there the whole cap counts, so a plane in front of the mirror's far side is refused while the beam passes within
    about 5 beam radii of the cap's near rim, whatever share the rim takes: a 0.3 mm beam over the near rim of an R 50 mm cap is refused at
    1.2 to 3 beam radii above the rim landing on its far side at 57 deg (0.82 % to 1e-9 of it clipped), and at 1.5 to 5 beam radii landing near
    the bowl's centre (1e-3 to 1e-23), where it is read from 6 beam radii. The reach and the nearest roots' spread along the ray are taken together as one ball around the line's earlier crossing of the
    sphere, which makes the band wider than the clipped share needs. A plane along the leg (exactly parallel to the chief ray) with an active area is bounded by the share
    whose energy flow crosses it on the stretch of the leg along that area (a paraxial Gaussian beam's flow lines keep their place in its profile; an
    astigmatic beam crossed off its axes: the share beyond the plane where the beam is widest); without an active area, by the share beyond it where
    the beam is widest on the leg. The bound of that part is the larger of this parallel-ray share and the paraxial energy flow across the plane, each
    crossing counted (PHYSICS_SCOPE P10); on a leg that leaves the scene only the energy flow exists. On a leg that leaves the scene, a plane without an active area counts the light that crosses it up to `max(1000 mm, support)` from the leg's start, `support` being the farthest along the ray of the design's physical surfaces, source points and active areas; no plane's origin enters it. The 1000 mm is a declared numerical choice, recorded with every run as `laboratory_reach_mm` in `manifest.numerical_tolerances` and not configurable. Part of a beam beyond that distance on a leaving leg is not counted on a plane with no active area; a chief ray that crosses the plane is an arrival wherever it crosses. To read it, state an active area or end the leg on a dump. A plane with an active area counts what crosses that area, wherever it lies; a chief ray does not cross such a plane beyond `max(1000 mm, support)` along it (that crossing is off the area), so that beam is not listed in `outside_active_area` and only its light on the area is counted. A plane more than `K` beam radii from the chief ray
    over the stretch reads as dark, `K` about 3.3 at the default `negligible_power_fraction`. Behind a severe hard-edge truncation (`GAUSSIAN_SEVERE_TRUNCATION`) a traced beam has no Gaussian state, and its diffraction falls off too slowly for its rays to locate it. Any plane the line does not light with a point ahead of that leg's start, within that reach on a plane with no active area, reads `BOUNDED` from 0 to the beam's power: `total_power_W` null, `power_lower_bound_W` 0 and `power_upper_bound_W` the beam's power, `DIFFRACTED_BEAM` in the plane's `followed_light` and in the `OBSERVATION_NOT_REACHED` notice's `untraced_light`, status `PASS`, and `GAUSSIAN_SEVERE_TRUNCATION` among its target's codes. A `power_W` maximum at or above that bound is `MET` and a minimum above it `UNMET`; any other power limit, and any beam metric there, is `INDETERMINATE`. A plane the line does light keeps its reading, and the light is not taken out of the beam, on an absorbing plane too. To decide any other target there, open the aperture until the beam keeps its Gaussian state, or place the plane where the line lights it. A milder truncation keeps its Gaussian state, and aperture diffraction is not propagated. A beam without a Gaussian state for any other reason, and untraced light (a ghost a lens's rim truncates), is bounded by the whole beam within three envelopes of its live pupil rays; beyond that reach, or with no live pupil ray, nothing is recorded for that leg. A share at or below
    `negligible_power_fraction` of the line is left out. Such a plane without a traced arrival reads `BOUNDED` from 0 to that part's bound
    (`power_lower_bound_W` 0, `total_power_W` null, `PARTIAL_CROSSING` in `followed_light`, status `PASS`), `UNKNOWN` without a finite bound or with
    light of unknown fate. Its `power_W` and `coherent_power_W` targets are decided only in the direction the bound proves (a maximum at or above it
    `MET`, a minimum above it `UNMET`); any other limit, and any beam or pair metric with a selector that selects no arrival, is `INDETERMINATE`. To
    decide one, place the plane where the chief ray crosses it or outside the beam's envelope. A plane where nothing but light left out as negligible
    may add is `BOUNDED` too, its targets decided when both ends of [0, bound] agree;
  * an absorbing plane part of the beam meets that way (on its active area) also takes that part out of the beam: from there on the beam's power is out of
    model scope (`power_out_of_scope_at`), and so is the power of a transparent plane the beam reaches after part of it met an absorbing plane (also one tied
    with it) or a physical surface (above);
  * where an absorbing plane and a transparent plane meet the chief ray at one point, the transparent plane records only the rays that reach it no later
    than the absorbing plane.
* **Exposure upper bound: what it does not count.** `EXPOSURE_BOUND_WITHHELD` names a rated surface without a bound through a pair whose gain bound is 1 or a refused coating use, and a plane through such a pair or a closed return (`quantity` `power_upper_bound`); it moves no bound, comparison, target or status. A closed return on a rated surface carries `OUT_OF_MODEL_SCOPE`, stored light at the interaction limit `MAX_INTERACTIONS`.
  * **Internal folds inside one plane-faced solid.** A plane-faced solid (beamsplitter cube, polarizing cube, penta prism, right-angle prism) whose left-out light can return to a face through two or more of its other faces has those returns counted. The directions inside are the mirror images of the cones the light enters with (from every surface outside the solid that redirects light and every source's beam, refracted in); a step from one face to the next exists only where a straight segment of that cone joins the two clear apertures, and carries the face's reflectance bound over the step's angles, 1 where they reach the critical angle (total internal reflection); a step that meets an air-equivalent coating record beyond its stated angles reads the record's stated value, and that face has no exposure upper bound (`EXPOSURE_BOUND_WITHHELD` names the record), as for the first run of the left-out light; the closed walks at each face are summed with a certified bound, and the open paths' counts add every walk that closes a loop other than an alternation with a face the pair bounds already count. The closed walks back to an entrance face sum to 0.0076 on the catalog beamsplitter cube, 0.0102 on the polarizing cube and 0.0049 on the penta prism (its fold, 0.005 x 0.99^2); exact ray trees traced inside these solids never exceed them. The faces' exposure upper bounds rise by 0.03-1.2 % on the cubes, whose decisions stay, and by 3.9-10.3 % on the penta prism, whose S4 then reads `CONDITIONS_NOT_VERIFIED`. A solid whose closed walks have no such bound (a lossless return, more than 192 direction cones) is a closed return: no surface the left-out light reaches has an exposure upper bound, and an `OUT_OF_MODEL_SCOPE` issue names its faces. A solid whose light meets each face once between the returns of one pair (a window, a plate, a wedge, a dove prism) keeps the pair bound; there the pair model leaves out total internal reflection at a face met beyond its critical angle (a wedge's walk toward its apex). A solid with a crystal body, a non-convex body or a curved face keeps the pair and open-path bounds without its internal folds. Which solids are analysed: the plane faces of one element with one isotropic body index, convex (every corner of a polygon face, as a mesh solid's planar patches are, and every other face's aperture extent on the body's side of each face's plane) and with no source inside: catalog prisms and cubes, and mesh solids (`FACET_SOLID`) alike, each cell of an arrayed mesh member its own solid. A closed mesh solid of glass (a corner cube, a penta, rhomboid or dove prism, a Fresnel rhomb, a light guide) has no certified sum: the analysis keeps, per direction cone and face, one part of the face's aperture for all the visits there, so light that advances along faces met beyond the critical angle or silvered closes a lossless walk of states although the light itself moves on and leaves (a rhomboid's or a light guide's zigzag between two parallel faces, a dove prism's between its base and top, a penta's two silvered faces and a face met beyond the critical angle), and a corner cube's cones, imaged again by its partly reflecting entrance face, do not close within 192 (nor do those of a mesh whose rounded coordinates leave its faces slightly off right angles); such a solid is a closed return (`POTENTIAL_RETURN`, no exposure upper bound where its left-out light reaches). The analysis does not bound it; it has found no lossless return of the light: ray trees traced inside these meshes over the light the model admits find every circulation passing a partial reflection, and finite per-face sums of a ray's visits in amplitude (a corner cube's back face 1.25 with a bare entrance face, 1.053 with an AR coating of reflectance 0.0025; a penta's 1.25; a rhomboid's 3.75 and a light guide's 8.5 at the faces of their zigzag, a few lossless visits per pass). A mesh cube of index 1.05 lit across a corner (no face near its critical angle) is bounded, and the traced walks stay within its sums. The open paths' counts leave out the walks whose only loops are runs of alternations with one paired face, which the pair factor counts; that factor counts one such run per walk and a run with one partner, so a walk whose loops are alternation runs at two or more faces of one open path, or a run alternating with two partners at one face, is counted by neither. Its share is second order in the runs' sums `A - 1`: at most `prod A - 1 - sum (A - 1)` over a solid's faces, 1.1e-3, 1.5e-3 and 3.7e-3 of the solid's whole closed-walk product on the catalog beamsplitter cube, polarizing cube and penta prism (0 on the right-angle prism, which pairs no face); exact ray trees inside these solids have not exceeded the bounds.
  * **Light held by total internal reflection in a lens body.** The exposure upper bound on a lens face counts the single return between the face and its pair; light that enters the glass body and reaches the total-internal-reflection range is not counted. In the reference designs (focus, folded, beam expander, 4f relay, three-lens AR train) the internal angle reaches at most 0.9 deg against a critical angle of 41.4 deg (N-BK7 at 780 nm), no ray enters the TIR range, and the omitted term is zero (K = 1.000000). That angle is the entering beam's: ghosts that fill the same lenses reach 39.8 deg in glass, without reaching the TIR range, and the coating angle reference below states what they do to the exposure bound. A curved lens body has no bound on consecutive total internal reflections over arbitrary light inside it: in the 0.25 %-AR biconvex singlet (R 101.3 mm, 4 mm thick, 12.7 mm clear radius), rays started inside it in random directions meet its faces beyond the critical angle up to 12 245 times in a row before a partial reflection, so no series over its reflections converges for such light (this does not show that light the lens receives is held: at the reference designs' feeds none reaches the total-internal-reflection range); a bound on that light needs a limit on the power it holds, which the exposure model does not state. For wide entry cones the held power grows: at a 75 deg cone the 1 %-AR catalog lens holds 2.6 % of the power (4.6 % of rays ever reach TIR; amplitude factor 1.27, returned-power ratio 4.3); near grazing it holds 13.0 % (amplitude factor 2.36, returned-power ratio 16.0, up to 5 consecutive TIRs), and a 0.25 %-AR singlet reaches 16.9 % held (amplitude factor 4.53, returned-power ratio 69.6, up to 30 consecutive TIRs). A power-based bound on TIR-held light is needed only for such wide or grazing feeds; the reference designs are unaffected.
  * **Returns among refracting surfaces.** The pair model, the open-path count and the `A < 1` series count a refracting surface at most once between two reflections; light that refraction turns back onto a surface it already passed before being reflected again is not counted. A bounded search (14 executed fixture runs and 540 configurations of a two-singlet strong-curvature train, reflections traced with an explicit budget) produced no traced path that meets a refracting surface twice between two reflections; refracting-surface returns themselves are traced (0.25 %-AR faces return 0.99 % of the incident power to a plane behind the source). The search did not cover plane-faced prisms and retroreflectors, ball and hemispherical refractors, trains of three or more elements, or catalog products. The rule: this return class is outside the exposure bound.
  * **Coating angle reference.** Every coating record states the medium its angle range refers to: 629 of 707 records (catalog release 2026-10-28.1) declare an air-equivalent angle, 78 the incident medium. Met from a medium denser than air, an air-equivalent record is evaluated at the bounded angular extent of the light the trace leaves out, converted by `asin(n sin theta)`; a use outside the record's declared angle or wavelength range, a use above the critical angle (no real air-equivalent angle) or a use whose extent cannot be bounded is refused rather than approximated, and that face has no exposure upper bound (`EXPOSURE_BOUND_WITHHELD` names it, with the record, the medium, the extent and the air-equivalent angle). The refusal concerns the exposure upper bound of the refused face and of a pair it leaves at a gain bound of 1; a plane's power bound and every other surface read the record's stated value. The reference-design lens coating `AR_400-1100R0.01T0.99_AOI0-10` is published for 0-10 deg air-equivalent. In `focus` and `folded` the left-out light meets it from the glass within 9.82 and 9.80 deg air-equivalent: the value holds and the bounds stand. In `beam_expander` and `relay_4f` faint ghosts fill the lenses, their extent is not bounded below the record's range, and the four faces of each design have no exposure upper bound; the planes keep their power upper bound. At the 5.715 mm clear radius the face's normal alone gives 19.6 deg air-equivalent on R 25.8 mm and 9.65 deg on R 51.5 mm; the refusal stands on the ghosts' own rays, which reach 22-75 deg air-equivalent on those faces. To obtain a face's bound, use a coating whose record states those angles, or keep the beam well inside the clear aperture. A glass-side reflectance at high angles needs per-side or angle-resolved data.
* **Power of several beams.** `power_W` and `total_power_W` add the powers of the beams that arrive: the sum stays constant across a fringe. Where two or more beams of
  one wavelength arrive `power_basis` is `INCOHERENT_SUM`, a `power_W` target that sums them carries `POWER_INCOHERENT_SUM`, and `superpositions` and `coherent_power_W`
  state the power with their fields added. Catalog coatings publish a group-delay bound and no phase, so in a two-arm interferometer the coherent power is the value at
  the trace's default interface phases (`value_basis`: `ESTIMATE`, `UNSTATED_PHASE_AS_INTERFACE`) and its range over every phase stands beside it (`total_power_min_W` /
  `total_power_max_W`, `combined_power_collinear_min_W` / `_max_W`): a `coherent_power_W` target reads that range widened by the light the trace follows that may reach
  the plane, a limit inside that interval is `INDETERMINATE`, and stating the coatings' phases on a declared coating gives the value (`docs/engine/RESULTS_AND_EXPORT.md`).
* **Multi-spectral** sources are traced line by line; different wavelengths never interfere (`WAVELENGTH_MISMATCH`, `NOT_APPLICABLE`).
* **Weak-signal measurement chain and nonlinear conversion** (`docs/engine/PHYSICS_SCOPE.md` P9b, `capabilities()["model_scope_boundary"]`).
  Supported: extended emitters, weak bulk absorption, the static camera response, the static acousto-optic deflector,
  fluorescence and spontaneous Stokes Raman emission of a thin sample derived from the excitation the run delivers to it
  (P9c), and a plane's detector with its bandwidth, shot / dark / thermal noise and a chopper-and-lock-in measurement (P9d).
  Their limits: a thin, optically dilute fluorescence sample (absorbance at most 0.1, refused above); linear excitation
  without saturation, bleaching or reabsorption; emission following the excitation instantly; isotropic, unpolarized Raman
  scattering without a depolarization ratio; Stokes lines only; the excitation beam not attenuated by the sample; white
  noise only, referred to the detector's input (no 1/f, gain or excess noise, amplifier voltage noise or ADC quantization);
  a modulation within the detector's bandwidth (above it the lock-in's noise and SNR are `UNAVAILABLE`,
  `MODULATION_ABOVE_DETECTOR_BANDWIDTH`); a photocurrent linear in the power (a stated `saturation_current_A` refuses the
  photocurrent above it, `PHOTOCURRENT_ABOVE_LINEAR_RANGE`; the shape of saturation is not modelled); a sine reference at
  the fundamental with an RC output filter that averages over at least 10 periods per time constant; no heterodyne
  beat-note or balanced detection. Not modelled: feedback and modulator timing and a spectrometer's
  instrument function (second-harmonic generation is modelled: next item). A target on `spectral_resolution_nm` reads
  `OUT_OF_MODEL_SCOPE` with the route; optical model kinds `CHOPPER` / `LOCK_IN` (declared in a plane's `detection`) are
  refused as `SIMULATION_MODEL_UNSUPPORTED` with the route. A two-line source standing for a fundamental and its harmonic
  traces the optics at both wavelengths with the powers stated; it computes no conversion.
* **Second-harmonic generation** (`NONLINEAR_CRYSTAL`, `docs/engine/PHYSICS_SCOPE.md` P10l): undepleted pump only (pump depletion
  beyond `nonlinear_depletion_rel_tol` is out of model scope, not computed); a TEM00, polarized, single-line, continuous-wave
  fundamental (a source of several lines or a partially polarized one is out of model scope: no sum-frequency mixing, no
  incoherent mixture; a pulsed source is out of model scope: no peak-power or pulse-shape model; a line whose stated width
  changes the line-averaged phase matching by more than `nonlinear_linewidth_rel_tol` is out of model scope); cascaded
  crystals and second passes (a harmonic that travels with the light entering a crystal) are out of model scope; uniaxial
  crystals only (biaxial KTP, LBO and cubic crystals are not modelled) and no built-in crystal data (index laws, `d` and
  thermo-optic data are stated; with a quasi-phase-matching grating the stated coefficient is the unpoled one, e.g. `d33`);
  one attenuation coefficient for the fundamental and the harmonic; a slab with two parallel plane faces; dn/dT linear per stated row (no temperature-dependent
  Sellmeier coefficients); the quasi-phase-matching grating's 50 % duty cycle; type II only in the near field; walk-off
  beyond `nonlinear_harmonic_gaussian_rel_tol` leaves the harmonic without a Gaussian state. Light of probe traces and an
  extended emitter's sampled rays pass the crystal unconverted. Thermal effects, photorefraction, higher harmonics,
  parametric and chi(3) processes are not modelled.
* **Return-light analysis work.** The bounds of light left out of the trace read an analysis of every pair of reflecting
  surfaces (a few milliseconds a pair) and of the closed paths through three or more surfaces (a search that can grow
  exponentially with the reflecting surfaces). It runs only for a wavelength whose left-out light a bound reads, and the
  whole run's analysis stops at `max_return_analysis_work` units (default 4,000,000, about 15 to 25 s on a desktop CPU, however
  many source lines leave light out: the wavelengths share it in the order the run reads them, and one that finds too
  little left is not finished): a scene of 128 reflecting surfaces with left-out light reaches it, and its power and
  exposure upper bounds are then withheld (`RETURN_ANALYSIS_BUDGET`, run `PARTIAL`), not computed. Raise the budget to
  obtain them. The closed-return search stops at the first closed return it cannot rule out (shortest first) and names that
  one; its notice also lists the surfaces of every group the search screens (`closed_return_candidates`): closed returns
  between separate elements run only among the surfaces of one such group, so one run says where any other can be. A stack of
  parallel planes is not searched for paths: its surfaces are named as a group, bounded by an all-orders series where one
  applies.
* **Many beams on one plane.** Pairs of beams from independent sources are counted per plane (`incoherent_pairs`), not analysed;
  pairs of one coherence group are analysed up to `max_interference_pairs` per plane. With `--ray-data full` a run writes about
  100 kB of ray data per beam path; `--ray-data summary` keeps the chief rays only.
* **Run resource limits.** `--max-memory-mib` and `--max-artifact-mib` (a `RunMonitor`'s limits) are checked at the run's
  checkpoints: the memory at most every 0.1 s, so an allocation between two checks can pass the memory limit before the run
  stops; the run-file limit is checked before each piece is kept and is never passed. The estimate before the trace is beams
  times per-beam costs measured on 2500 independent beams (two paths a beam, 8 x 12 pupil rays): a scene whose beams split
  more or meet more surfaces needs more (one beam through 64 window-and-tap stages wrote 92 times its estimate). When the
  trace ends, a second estimate from the beam paths and steps it counted is stated with a range: from the run-file bytes
  the run writes for certain (the Project and Engine request files, and each path and step of the result at no fewer bytes
  than its shortest record) to 1.5 times the estimate (measured runs wrote 0.66 to 1.42 times it). A run is refused there
  only when those certain bytes, or the memory read then, exceed a limit; the certain bytes are far below what a run
  writes, so mostly the checks during the run hold the limits. The memory costs after the trace were measured natively, not in a browser. A
  cancelled or stopped run reports nothing of what it computed.

* **Configurations** (`ProjectDocumentV2.configurations`): a configuration changes component poses, observation plane poses,
  source models and budgets. Removing a component in one configuration (a flip mirror taken out of the path) is expressed by
  moving it out of the beam; there is no per-configuration exclusion. In a laboratory assembly a component with two or
  three retained rotation axes may take any orientation in a configuration (the order of a stage's rotations, which decides
  the reachable orientations, is not stated), and a configuration's displacement is not compared with a DOF's stated
  travel. Each configuration is a separate Engine execution: time and memory grow with the number of configurations run.

## C. Layout / control limits

* Layout is **2.5D**: one height slab per component; plan-view footprints, bounding-box-pruned pair distances, moving envelopes by sequential sweeps
  (rotation is sampled, `moving_envelope_rotation_samples=9`), tool/cable regions are *declared* polygons. Mounting holes, fasteners, cable bend radius and tool
  reachability are not computed.
* Control: sensors are `centroid_u`, `centroid_v`, `power` at an observation plane; Jacobian by central finite differences (no analytic sensitivity); LM is local
  (no global optimization); capture scan covers at most two DOFs on a regular serpentine grid; disturbances are modelled as offsets of the same DOFs. There is no
  separate API for parameter sweeps, randomized disturbances or resets (sweeps are repeated `validate` calls).
* **Performance**: every control evaluation re-lowers and re-traces the whole scene (no incremental cache). Standard local validation meets the 2 s target for the standard 100-component benchmark (native 0.57 s,
  Pyodide 1.31 s warm median, 1.43 s max), but the control fixture (sensitivity + capture scan + 2 disturbances + 6 Monte Carlo samples) takes 1.2 s natively and
  2.2 s in Pyodide.

## C2. Scattered light, thin samples and facet surfaces

* Surface scatter is single scatter per traced beam, leaving from the beam's chief-ray hit point (the footprint is not resolved),
  `LAMBERTIAN` or `COSINE_POWER` only, incoherent and unpolarized; light a scattered ray scatters again, light of probe traces and
  the scatter events beyond the 64 strongest of a run are booked `SURFACE_SCATTER` and not followed. Bulk scattering is booked
  `BULK_SCATTER` and not followed (no volume emission, `anisotropy_g` reported only). The planes their source lines light then read
  a power that is not complete.
* A thin complex-transmission sample turns rays by its local phase gradient (geometric); diffraction by its structure is not
  computed in a run, only through the analyses that read the rays' optical paths. A beam whose phase is not a plane across its
  footprint loses its Gaussian quantities. Where the sample turns part of a beam beyond grazing, the share that goes on is a sampled
  value (the share of the beam's equal-power pupil samples, with its sampling resolution) and the evanescent share's fate is not
  known; a turned share at most the line's negligible level is left out as negligible light and the beam keeps its power. The return-light analysis treats the sample as returning no light and not turning
  the light it passes. Polarization-resolved maps and birefringent samples are not modelled.
* Triangle-facet surfaces are traced as their planar faces. The part of a beam beyond a face's edge is not followed by that beam
  (its fate is not known, `FACET_EDGE_DIVIDES_BEAM`); a beam spread over many small facets loses most of its followed power.
  Facets approximating curved surfaces are traced as the flat facets: the curved surface's normals and curvature are not
  recovered, and no Gaussian beam is stated behind them. Facets are coplanar within 4 sqrt(3) times the coordinates' rounding (at
  least 1e-10 mm, at most 1e-4 of the part's size): a face whose corners are given coarser than that (two decimals on a 10 mm part)
  splits into several, whose shared edges divide a beam; a real bend smaller than that tolerance is traced flat (stated by
  `FACET_FACE_FITTED` above 1e-5 of the part's size). A face's outline has at most 1024 corners (more: traced facet by facet). An
  open sheet between two media, a crystal on a side of the facets, more than 1024 planar faces and closed facets whose shells
  cross, overlap or touch (two parts in contact) are refused (`FACET_SURFACE_NOT_TRACED`). The return-light analysis examines every pair of faces of a solid within its work limit: from about
  60 faces it reaches the default limit and the power upper bounds that read it are withheld (`RETURN_ANALYSIS_BUDGET`). Diffraction at facet edges is not modelled; no CAD file
  format (STL, STEP, OBJ) is read: the facets are stated as vertices and triangles.
