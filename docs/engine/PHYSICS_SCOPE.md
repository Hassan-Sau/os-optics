# os-optics — Physics Scope

For every physics module: governing equations, assumptions, applicability
domain, numerical failure modes, and the independent reference cases used to
validate it. Each physics module has a section here with its equations,
assumptions, validity range and independent references; the section is written
or updated in the change that adds or alters the module. Section numbers are
referenced from test docstrings.

Global conventions (details in `NUMERICAL_CONVENTIONS.md`): right-handed 3D,
mm / nm / W / degrees at the API, field `E ∝ exp[i(k·r − ωt)]`, `k = 2π n d̂ / λ₀`
(λ₀ = vacuum wavelength).

A result answers four questions in four places: `outcome.execution` (did the calculation complete),
`outcome.requirements` and each target's `assessment` (are the declared targets met), each quantity's state and
`value_basis` entry (is the number usable), and the aggregate `status`, a diagnostic summary that sorts and flags and
decides nothing.

Status vocabulary (every result): `PASS`, `WARNING`, `FAIL`, `UNKNOWN`,
`NOT_APPLICABLE`, `OUT_OF_MODEL_SCOPE`.

---

## P1. Ray geometry and surface interaction

### Governing equations

Ray `r(t) = o + t d̂`, `|d̂| = 1`, `t > t_min`.

* Plane through `p₀` with unit normal `n̂`: `t = n̂·(p₀ − o) / (n̂·d̂)`, valid if `|n̂·d̂| > ε_par`.
* Sphere centre `c`, radius `R>0`: `|o + t d̂ − c|² = R²`. With `b = d̂·(o−c)`,
  `cc = |o−c|² − R²`, `Δ = b² − cc`; roots `t = −b ± √Δ`. To avoid
  cancellation the smaller-magnitude root is `t₂ = cc / t₁` where
  `t₁ = −(b + sgn(b)√Δ)`. The smallest root `> t_min` is selected. Outward
  normal `n̂ = (r − c)/R`; a ray that misses the cap (`t = ∞`) is given the cap
  axis as a finite unit normal (never used for physics).
* Sag surface (`SagSurface`: conic / even asphere, rotationally symmetric or
  cylindrical, any pose). Local frame: vertex `p₀`, axis `n̂`, transverse axes
  `û`, `v̂ = n̂×û`; a point at transverse coordinates `(a, b)` is
  `p₀ + a û + b v̂ + z(s) n̂` with `s = a² + b²` (rotational) or `s = a²`
  (cylindrical: curved along `û`, straight along `v̂`) and
  `z(s) = c s/(1 + S) + Σⱼ A₂ⱼ₊₄ s^(j+2)`, `S = √(1 − (1+k)c²s)` (`c`: vertex
  curvature, positive when the centre of curvature is on the `+n̂` side; `k`:
  conic constant; `A₄, A₆, …`: even-asphere coefficients). The profile exists
  where `1 − (1+k)c²s ≥ 0` (its domain); a clear aperture reaching beyond it is
  rejected at input (`DEGENERATE_SURFACE`).
  - Rays are expressed in the local frame from their point of closest approach
    to the vertex (no cancellation from distant origins).
  - Conic-only profiles (any `k`, including the paraboloid `k = −1` and the
    cylinder) are intersected in closed form: the quadric
    `c(s + (1+k)z²) − 2z = 0` gives `α τ² + 2β τ + γ = 0`, solved without
    cancellation (`q = −(β + sgn β √Δ)`, `τ₁ = q/α`, `τ₂ = γ/q`); only roots on
    the vertex branch `1 − (1+k)c z ≥ 0` belong to the surface.
  - With polynomial terms, `F(τ) = z(τ) − z_sag(s(τ))` is sampled at
    `sag_bracket_samples` points across the ray's crossing of the support's
    bounding box (transverse box of the support, axial bounds from the
    monotonic conic part plus `Σ|A| ρ_max^(2j+4)`); each sign change, nearest
    first, is refined by Newton steps kept strictly inside the shrinking
    bracket (bisection otherwise) to `sag_intersection_tol_mm`, until a root
    lies inside the support. Two intersections closer than one sample spacing
    along the ray (a ray grazing a local bump) are not separated.
  - The clear aperture and the mount ring are measured on `(a, b)` (circle or
    rectangle, optionally decentred: off-axis sections such as off-axis
    paraboloids); the support rule is the planes' one. Hits outside the domain
    or the support are misses. Gaussian aperture truncation maps the beam onto
    `(a, b)` through the tangent plane at the chief-ray hit (axes `û`, `v̂`
    lifted onto it along `n̂`, area factor `|d̂·N̂| / (n̂·N̂)`).
  - Normal toward `+n̂` (finite at the domain edge `S = 0`):
    `(−a h_S, −b h_S, S)/W_S` with `h_S = c + 2S P₁(s)`,
    `W_S = √(S² + s h_S²)`, `P₁ = dP/ds` of the polynomial part.
  - Principal curvatures (positive when the centre of curvature is on the `+n̂`
    side), along the meridian / curved axis
    `κ₁ = (h_S S² + (1+k)c³ s + 4 s S³ P₂(s))/W_S³` and along the parallel
    circle `κ₂ = h_S/W_S` (rotational) or `0` along the straight axis
    (cylindrical); meridians and parallels are the lines of curvature, so these
    are the principal directions (used by P4).
  - A cylinder may carry a conic and polynomial terms (an acylinder, e.g. a
    parabolic cylinder): `s = a²` in every formula above.
* Axicon term. A sag surface may add `L ρ` to its sag (`linear_sag_coefficient`,
  `ρ = √s`): a cone of slope `dz/dρ = L` with its apex at the vertex, alone or
  added to a conic / asphere; on a cylinder a roof of two planes meeting along
  the straight axis. A face of base angle `α` (between the cone and the plane
  normal to the axis) has `|L| = tan α`.
  - Sag `z(s) + L√s`; the intersection always uses the bracketed root search
    (the cone is not a quadric), with the axial bounds widened by `L ρ_max` and
    Newton steps on `dz/dτ = … + L (r_a d_a + r_b d_b)`, finite at the apex.
  - Normal toward `+n̂`: `(−(a h_S + r_a L S), −(b h_S + r_b L S), S)/|…|` with
    the radial unit components `r_a = a/ρ`, `r_b = b/ρ` (a cylinder:
    `r_a = sign a`, `r_b = 0`).
  - The apex is a single point without a tangent plane. A ray meeting the
    surface within `AXICON_APEX_RADIUS_MM` (1e-9 mm) of it takes `r_a = r_b = 0`
    (the normal of the profile without the cone term: the axis for a pure cone),
    so a ray along the axis passes undeviated, as it does by symmetry; every other
    ray is refracted or reflected by the cone's own normal. A plano-axicon of
    index `n` deflects a ray parallel to its axis by
    `β = asin(n sin α) − α`, and a ring of rays at height `h` crosses the axis
    `h / tan β` behind the cone: the Bessel zone of a beam of radius `R` ends at
    `R / tan β` (`tests/test_axicon_surfaces.py` checks both against Snell's
    law, and the roof's deflection across its straight axis only).
  - Gaussian beam: a cone turns a Gaussian beam into a Bessel-like zone, which
    the Gaussian model does not describe. The beam's Gaussian state becomes
    invalid at an axicon surface (`GAUSSIAN_UNDEFINED_AFTER_AXICON`,
    `OUT_OF_MODEL_SCOPE`): Gaussian metrics behind it are out of model scope,
    and so is the first-order (ABCD) transfer of an arrival (a cone is not a
    linear map of rays about the chief ray). Rays, powers (Fresnel and coatings
    at the cone's local angle of incidence), polarization and pupil-ray metrics
    (`spot_rms_mm`, centroid, wavefront) are traced as on any surface. A clear
    aperture that cuts the pupil rays of such a beam passes the share of its
    pupil samples that pass it (the geometric pupil-sample basis below). An
    element with an axicon surface has no paraxial matrix.
* Two-variable sag shapes (`SagSurface.sag_shape`; catalog and Project surfaces
  `sag`, schema 3.4). `(a, b)` are the surface's local `(x, y)`, `z` is measured
  along its axis.
  - Zernike: `z += Σ value_j Z_j(ρ, θ)`, `ρ = √(a² + b²) / R`, `R` the
    normalization radius (a clear aperture beyond `R` evaluates the stated
    polynomial at `ρ > 1`; `surface_profiles[].aperture_reach_mm` states the
    reach). `NOLL` indexing pairs with `RMS_ORTHONORMAL` (R. J. Noll,
    J. Opt. Soc. Am. 66, 207, 1976: unit RMS over the unit disc, `Z₄ = √3 (2ρ² − 1)`,
    `Z₆ = √6 ρ² cos 2θ`, index 4–231, radial order ≤ 20), `FRINGE` with `PEAK_UNIT`
    (the 37-term Fringe ordering: unit peak, `Z₄ = 2ρ² − 1`, `Z₅ = ρ² cos 2θ`, index
    4–37). The terms are not referenced to the vertex: an `m = 0` term moves the
    point on the axis by its value at `ρ = 0` and a coma term tilts the surface
    there, as in a sequential ray-tracer's Zernike sag surface; the result states
    `vertex_sag_mm`. The clear aperture lies inside `R`
    (`SAG_APERTURE_BEYOND_NORMALIZATION`). Beyond `R` (a rim, a flange) the same
    polynomial continues.
  - XY polynomial: `z += Σ value (a/R)^i (b/R)^j`, `2 ≤ i + j ≤ 20`, each coefficient
    a length.
  - Biconic: `z = (c_x a² + c_y b²) / (1 + √(1 − (1+k_x) c_x² a² − (1+k_y) c_y² b²))`.
  - Toroid: the curve `z_y(b) = c b² / (1 + √(1 − (1+k) c² b²)) + Σ α_i b^(2i)` turned
    about the axis parallel to `b` through `z = R`:
    `z = z_y + p a² / (1 + √(1 − p² a²))`, `p = 1 / (R − z_y)`, the cancellation-free
    form of `R − sign(R) √((R − z_y)² − a²)`. Without `R` the curve is extruded along `a`.
  - Zernike and XY add to the surface's radius, conic and aspheric terms; biconic
    and toroid replace them. Piston and tilt state a pose and are refused
    (`SAG_TERM_NOT_SUPPORTED`), as are a mismatched indexing and normalization
    (`SAG_NORMALIZATION_MISMATCH`), any shape with `cylinder_axis` or an axicon term
    (`SAG_SHAPE_COMBINATION_NOT_SUPPORTED`) and a profile undefined somewhere in the
    clear aperture (`SAG_OUTSIDE_DOMAIN`).
  - Numerics: each Zernike term is expanded once into integer monomials of
    `(a/R, b/R)`, so no trigonometric function is evaluated and one Cartesian evaluator
    serves every polynomial shape. The intersection is the bracketed solver of the
    asphere on the two-variable residual `z(τ) − sag(a(τ), b(τ))`: `sag_bracket_samples`
    samples across the support's bounding box, whose sag bound is
    `Σ |c| (ρ_max / R)^(i+j)` on top of the base terms; a ray that turns back between
    two samples is split at its closest approach. Normals are the analytic gradient,
    principal curvatures the eigenvalues of the shape operator of the first and second
    fundamental forms, with their directions (a Gaussian beam meets them as it does
    any astigmatic surface: `GAUSSIAN_TWISTED_ASTIGMATISM` where they turn from
    the beam's axes).
  - Evidence: `ValidationResult.surface_profiles` states per surface the kind, indexing,
    normalization, radius, units, vertex sag, the rays solved, the most iterations,
    the largest residual `|z − sag|` and the rays that did not converge. A ray that does
    not converge within `sag_max_iterations` continues as if it missed the surface,
    and the run says so: `SAG_INTERSECTION_NOT_CONVERGED`, status `UNKNOWN`.
  - An element with a Zernike or XY polynomial surface has no paraxial matrix
    (`PARAXIAL_SURFACE_NOT_SUPPORTED`: its low-order terms move, tilt and bend the
    surface at the vertex); a biconic or a toroid has one per section along local X
    and Y.
* Reflection: `d̂' = d̂ − 2(d̂·n̂)n̂`. In the paraxial analysis (`os-optics project paraxial`) a mirror of vertex radius
  `R` is `((1, 0), (−2/R, 1))` on the unfolded axis (`R > 0`: centre of curvature on the side the light comes from, concave),
  a plane mirror the identity; only a plane mirror may be tilted, folding the axis. The section axes follow the same reflection.
* Snell (vector form): with `η = n₁/n₂`, `c₁ = −d̂·n̂ ≥ 0` (n̂ oriented against the ray),
  `k = 1 − η²(1 − c₁²)`. If `k < 0`: total internal reflection (TIR). Else
  `d̂' = η d̂ + (η c₁ − √k) n̂`.
* Ideal thin lens (slope form). Lens axis `â` (oriented along the ray:
  `â·d̂>0`), centre `C`, focal length `f` (`f>0` converging). Transverse offset
  `h = (P − C) − ((P−C)·â)â`; slope vector `s = (d̂ − (d̂·â)â)/(d̂·â)`;
  `s' = s − h/f`; `d̂' = normalize(s' + â)`. Property (exact for all angles):
  a bundle of parallel rays with slope `s` converges to the point `C + f â + f s`
  on the back focal plane; the ray through `C` is undeviated.
* Tabulated thin lens (`TabulatedThinLensInteraction`, the Project's
  `TABULATED_THIN_LENS`): an external-response equivalent model -- a thin
  focusing element whose focal length `f(λ)` and transmittance `T(λ)` are
  tables (for example fitted to an RCWA / angular-spectrum simulation or a
  measurement of a metalens). Each traced line takes its own `f(λ)`: optical
  power `1/f` linear in vacuum wavelength between rows
  (`LINEAR_IN_OPTICAL_POWER`), or the row within 0.5 % (`POINTS_ONLY`, and a
  one-row table); `T(λ)` linear between rows or the matching row. Rays follow
  the slope form above with `f(λ)`; the Gaussian beam the thin-lens ABCD step of
  power `cos θ / f(λ)` (P4); the line's power is multiplied by `T(λ)` and
  `1 − T(λ)` goes to the power-ledger sink `UNMODELLED_RESPONSE` (light the
  response model does not deliver into its focused beam: other orders,
  reflection, scatter, absorption). That light is not followed, so its fate is
  unknown: a plane no traced beam reaches is not dark for that line, and a power
  summed on a plane is bounded, not known (graded as in P10, trace budget). The optical path length through
  the element is not modelled: phases, wavefronts and network exports of beams
  through it are refused, as for an ideal thin lens.
  - Validity domain, checked by the Engine at the chief ray's arrival, each
    violation an `OUT_OF_MODEL_SCOPE` issue at the element's surface (the trace
    itself goes on, with the nearest row's values outside the table). The field
    angle, beam size, polarization and metric limits are optional statements of
    the model: one it does not state is not checked, and the run summary shows
    it as `null`:
    `MODEL_WAVELENGTH_OUTSIDE_BAND` (a line outside the rows' span; with
    `POINTS_ONLY` farther than 0.5 % from every row; one row: farther than 0.5 %
    from it), `MODEL_FIELD_ANGLE_EXCEEDED` (chief-ray angle to the element axis
    above `max_field_angle_deg`), `MODEL_BEAM_SIZE_EXCEEDED` (the arriving
    Gaussian's larger 1/e² radius above `max_beam_radius_mm`, or no valid
    Gaussian to check), `MODEL_POLARIZATION_NOT_VALIDATED` (the arriving chief
    ray is not linearly polarized along a validated element axis: more than
    sin²(1°) of its power in the orthogonal linear polarization; the validated
    direction is the element's local X / Y projected onto the ray's transverse
    plane). An angle or a radius is above its maximum only when it exceeds it by
    more than 1e-9 of the maximum plus 1e-12 deg / 1e-12 mm: a beam computed
    at the validated radius (a 30 µm waist at the element computes as
    0.030000000000000002 mm at some wavelengths) stays inside the domain.
  - Metric-level scope: the response is validated for the metrics it lists
    (`supported_metrics`, each with its error bound; `transmittance` stands for
    the power metrics `power_W` and `coherent_power_W`, any other entry for the
    target metric of that name). A target whose selected beams crossed the
    element (for a two-beam or geometric pair metric: the two beams of the pair
    it selects) and whose metric is not listed is `OUT_OF_MODEL_SCOPE`
    (`METRIC_NOT_VALIDATED_FOR_ELEMENT`, assessment `INDETERMINATE`). Its value is
    the number computed with the element's ideal thin-lens step, with
    `value_basis["value"]` an `ESTIMATE` that names `RESPONSE_MODEL_UNVALIDATED`
    at the element's surface; the target is not graded on it, because the
    element's record states no accuracy for that metric. A focal position
    or a delivered power can be validated this way; a PSF, side lobes, high-NA
    or wide-field image quality cannot be derived from the model.
  - Provenance: the run summary's `element_models` row of the element carries
    its `response` -- wavelength and validity domain, supported metrics with
    their errors, the Engine metrics they cover and the evidence (method, tools
    and versions, source-data and configuration SHA-256, fit, validation
    summary, date) -- so an external origin stays identifiable in every result.
* Aperture: point in surface plane, in local 2D coordinates `(u,v)`; circle
  `u²+v² ≤ a²`; rectangle `|u|≤a_x, |v|≤a_y`; optional circular obstruction
  `u²+v² < a_in²` blocks. Ellipse: `(x/a)² + (y/b)² ≤ 1` in its axes (turned by
  `rotation_deg`); D shape: the disc `u²+v² ≤ r²` with `x ≤ d` along the flat's
  outward normal (`|d| < r`); polygon: a simple polygon (convex or not, a
  triangle included; 3-64 corners, counter-clockwise, no two edges crossing or
  touching), membership by the even-odd rule. Boundary is inclusive within `aperture_edge_tol_mm`.
  The element's *support* is the aperture dilated by `mount_margin_mm`: hits inside the aperture
  interact; hits on the mount ring are blocked (`CLIPPED`); rays beyond the support miss the element.
  The ring grows a circle's radius, a rectangle's half sides, an ellipse's semi-axes, a D shape's
  radius and flat, and moves each polygon edge out (corners where the moved edges meet; a
  non-convex polygon whose grown outline would not stay simple needs a `mount_outline`).
  Every aperture question (membership, distance to the edge, extents along a direction, bounds)
  is answered by `optics/apertures.py`; where a shape's exact answer is not closed form a
  sound bound is used (an ellipse's distance `b (|A d| − 1)`, `b` the smaller semi-axis).
* Holes through an element (`holes`, plane and sag surfaces): openings inside the clear
  aperture, each a circle, rectangle, ellipse, D shape or polygon, apart from each other
  (their enclosing circles about their centres do not overlap). There is no surface in a
  hole: a ray through it passes straight on, a ray on the element around it interacts. The
  Gaussian power on the surface is the clear aperture's less every hole's (exact for the
  incident Gaussian); the share through the holes goes on beside the element like the light
  beyond the support (an `UntracedBranch` inside `APERTURE_TRUNCATION`). A beam whose chief
  ray passes through a hole goes on with its power: the surface around the hole is then light
  beside an element, by the same rule as a beam passing the outline of an element.
* Gaussian power on an aperture region (`optics/gaussian/truncation.py`,
  `optics/gaussian/aperture_regions.py`): circle and rectangle as below; an ellipse by the
  affine map `d = c + R diag(a, b) z` onto the unit disc (the circle integral of
  `S' = Mᵀ S M`); a D shape and a polygon by the boundary integral in the beam's whitened
  frame, `I = (1/√det S) ∮ (P × P′) (1 − e^{−2|P|²})/(4|P|²) dt` (Green's theorem on the polar
  area integral about the beam centre; valid wherever the centre lies), Gauss–Legendre panels
  walked along each edge or arc no wider than `max(1, ρ)/(2V)` of its parameter (`ρ` the
  boundary's distance from the beam centre at the panel start, `V` its speed): nothing of the
  Gaussian is cut off, the result is exact to rounding (checked against the rectangle's `erf`
  product, the disc, and adaptive 2D quadrature to 1e-9 relative), libm-free.
* An element inside the beam (obscured pupil, a Cassegrain secondary): when the chief ray meets the back of a mirror, an
  absorber that does not scatter, or the opaque plate of an aperture stop beside its opening (the opening passing a negligible
  share), whose whole support lies inside the beam and a non-negligible part of the beam passes beside it, the element is a
  central obstruction. The Gaussian share on its support is stopped (`BLOCKED_BACK_SIDE`, `ABSORBED`, `CLIPPED`); the rest, exactly
  `1 − ∫_support I dA` (centred disc of radius `a`: `exp(−2a²/w²)`), goes on as an annulus by the rules below: its pupil rays
  beside the support are its rays (carried to the plane across the chief ray at the hit), the chief ray an unpowered
  reference axis through the element. A pupil ray on the support meets the element wherever it hits it, before or beyond
  that plane (a convex secondary met off its axis: rays nearer its axis hit it beyond the plane). A pupil ray beside the
  support that meets another physical surface first leaves the annulus there, but the annulus's power still counts its light:
  that surface joins `power_out_of_scope_at` (an `OUT_OF_MODEL_SCOPE` issue with the share of the pupil samples), and the
  annulus's power behind it is out of model scope, never a known value. The support's outline projected along the chief ray onto that plane is the
  annulus's shadow (plus a curved element's sag times the chief ray's tilt as margin). A later surface with holes that
  the annulus meets around them: the reference axis takes the surface's continued profile across the hole, and the holes
  pass none of the annulus when the annulus rays' least-squares affine map from the shadow's plane to the surface carries
  the shadow's outline around every hole with more room than the map's largest residual plus the margin; otherwise the
  annulus's power is not computed from there on (`ANNULAR_BEAM_NOT_FOLLOWED`). An element with holes of its own proves no later hole dark (light
  through its holes goes on inside its outline): the annulus's power is not computed past a later holed surface. The share on the
  support is the Gaussian integral over the support mapped onto the plane at the chief ray's hit, which is the shadow of a
  curved support (sag surface, sphere) only along its axis: met more than 1e-7 (sine) off it, the annulus's power behind the
  element is out of model scope (`power_out_of_scope_at`, an `OUT_OF_MODEL_SCOPE` issue). Reference (`tests/test_obscured_pupil.py`): a classical Cassegrain
  (paraboloid primary with a central hole, hyperboloid secondary `k = −((m + 1)/(m − 1))²`) delivers `exp(−2a²/w²)` of the
  beam (rel 1e-6) to its image plane, the annulus rays meeting within 1e-6 mm of the axis; decentred 1.5-5 mm (and 5 mm
  diagonally) it delivers `1 − ∫_disc I dA` of the offset disc (Gauss-Legendre with `I0`, rel 1e-6).
* A beam whose chief ray passes beside an element while some of its pupil rays meet that element first on the way to the
  surface the chief ray meets, where they would reach that surface inside its clear aperture (a Cassegrain secondary beside
  a decentred beam's chief ray): the Gaussian integrals at that surface count the light as the beam's, and the beam's
  power follows its chief ray. With a valid Gaussian state the element joins `power_out_of_scope_at` (an `OUT_OF_MODEL_SCOPE`
  issue states the share of the equal-power pupil samples and, for a plane or sag surface, the Gaussian integral over its
  support in the plane of its vertex, exact for a plane and for a sag surface met along its axis), so every power behind it
  is out of model scope; the beam's own light reaching a plane behind lies between the power stated less that share and the
  power stated. Without a valid Gaussian state the samples' share is booked instead (geometric pupil-sample basis).
  Reference (`tests/test_obscured_pupil.py`): the Cassegrain with the beam 14, 16, 20 mm off its axis (the image's power
  over-counted by 21.6, 15.8, 7.5 % against `1 − ∫_disc I dA`): the share on the secondary equals the offset-disc integral
  (rel 1e-7) and the image's true power lies in the stated interval. The beam's tail between and beyond the pupil samples:
  with a valid Gaussian state every physical surface whose bounding balls come within `sqrt(2) K` beam radii of the leg is
  weighed by the cell bound of the lines that may meet it before the leg's end (`lower_share.possible_meeting_share`, below);
  above the line's negligible level it joins `power_out_of_scope_at` the same way (`Tracer._tail_surfaces`; the faces of the
  body the beam travels in and the other surfaces of the elements the leg begins and ends on are left out). Behind a central
  obstruction only the annulus's pupil samples find an element (for a straight edge the share missed is about
  `0.5 erfc(√2 ρ cos(π/M))`, `ρ = √(ln(2K) / 2)` the outermost ring's radius in beam radii, `K` rings, `M` points per ring:
  1.14 % at the default 8 x 12); an element smaller than the samples' spacing may be missed whole.
* Central obstruction hit by the chief ray: the light passes the annulus. Its power is the Gaussian
  integral over the annulus, `T = ∫_annulus I dA` (centred: `exp(−2a_in²/w²) − exp(−2a²/w²)`); the
  obstruction's share is booked `CLIPPED`, the part beyond the opening `APERTURE_TRUNCATION`. The
  packet goes on with power `T·P`; its pupil rays inside the annulus are the annulus's geometric rays;
  the chief ray goes on as an unpowered reference axis that keeps the surface sequence. The annulus is
  not a Gaussian beam: its Gaussian state is out of model scope (`GAUSSIAN_CENTRAL_OBSTRUCTION`) and the
  field phase on the axis is not defined. Behind it the Gaussian state of the *unobstructed* beam is
  carried on only as the envelope that bounds where the annulus can be (geometrically the annulus is part
  of that beam), and the opening that cut the annulus is kept as a second bound: paraxially a ray launched
  along the Gaussian phase front at transverse offset `x` reaches `m x` (`m = A + B/R` per Gaussian axis)
  while the envelope grows by `w'/w = sqrt(m^2 + (B M^2 lambda / (pi n w^2))^2) >= |m|`, so the light
  that passed the opening (a disc `D` about its own centre, which holds the chief ray) lies, per Gaussian
  axis, within `diag(a s_x w'_x/w_x, b s_y w'_y/w_y) D` for some `a, b` in [0, 1] and the signs `s` of
  `m`: inside the convex hull of that scaled disc (the 64-gon around it), its projections onto the two
  axes and the chief ray. The signs come from the annulus rays (their offsets at the opening and at the
  surface correlate at least 0.5); without rays both signs are taken. A later clear aperture passes the
  whole annulus when at most `negligible_power_fraction` (1e-9) of the envelope's power falls outside it,
  or when every such hull, projected along the chief ray onto the surface, lies inside it (a convex
  aperture without its own obstruction) -- so a decentred annulus is judged by where its opening lies.
  An active area (absorbing or transparent) counts the annulus whole when every ray of the annulus
  falls on it and the whole annulus provably does, or when at most `capture_min_power_fraction` of the
  annulus's power can fall beside it (the envelope's power outside, relative to the annulus's share of
  it; counted with an `INCOMPLETE_ENVELOPE` WARNING that states the bound); not at all when at most
  1e-9 of the envelope and no ray do. Its pupil rays, where any are left, must also reach the surface
  the axis reaches and take the same interaction there; with none (an obstruction wider than every
  sampled ring -- the rings reach 1.18 w -- or a trace without pupil rays: control readings, probe
  traces, the network recorder) the bounds alone decide at each surface (a surface beside the axis that
  catches the outer annulus between two surfaces is then not seen). Anything else -- an aperture or an
  active area that takes part of it (a second obstruction, a stop or detector smaller than the bounds, a
  small detector anywhere inside them, the geometric shadow included), a surface or interaction that
  takes some of the rays, no envelope (an invalid Gaussian state) -- leaves the power that continues not
  computed (the issue `ANNULAR_BEAM_NOT_FOLLOWED`): the annulus's power is booked once to
  `TERMINATED_OUT_OF_MODEL`, the rays the surface cuts are retired and the others go on as geometry with
  no power and the bound of what arrived (`power_upper_bound_W`); targets behind it have no value. The
  power is never taken from where the samples land (1/96 of the power each, never summed as power). A
  probe trace ends there instead (path termination `ANNULAR_BEAM_NOT_FOLLOWED`), as does an obstruction
  that no pupil ray passes beside. The bounds are paraxial and geometric: diffraction by the obstruction (annular
  PSF, Poisson spot in the shadow, spreading beyond the bounds) is not modelled.
* Geometric pupil-sample basis. A beam without a valid Gaussian state (`GAUSSIAN_SEVERE_TRUNCATION` upstream,
  `GAUSSIAN_SURFACE_TOO_STRONG`, or another reason) whose chief ray passes a clear aperture that cuts some of
  its pupil rays: the pupil samples carry equal shares of the line's power (equal-power rings and azimuths,
  `pupil_offsets`) and keep them along their rays, so the aperture passes `f = n_pass / n_live` of the power that
  reaches it. The cut rays retire, the power is scaled by `f`, the rest is booked `APERTURE_TRUNCATION`, and outside
  the element's own body the share of the cut samples that land beyond its support (and of those on the support
  whose cells may reach beyond it) goes on as an untraced branch. Samples retired earlier (by an aperture or as
  `DIVERGED`) do not count: the live samples share the power that reaches the aperture equally. The power then
  carries a sampling resolution, as a fraction of it: at each such cut it grows by `n_b / n_pass`, `n_b` the live
  samples whose cells may straddle the edge -- a live neighbour (beside it on its ring, the two nearest on each
  adjacent ring) met the other fate, or a passing sample of the outermost ring, whose cell reaches the Gaussian's
  tail -- and compounds as `r + r' + r r'`. It is convergence evidence, not a proven bound: for a collimated beam
  cut first at 1 w (13.5 %, Gaussian) and then at a2 = 0.2-0.8 w, the sampled power lies within it of
  `1 - exp(-2 a2^2)` at 8 x 12 samples, and the resolution halves with each doubling of rings and points
  (`tests/test_geometric_power_basis.py`). Arrivals behind carry `power_basis` `GEOMETRIC_PUPIL_SAMPLES` and
  `power_sampling_resolution_W`; a `GEOMETRIC_POWER_BASIS` WARNING names the surface, the samples cut and the
  reason; the Gaussian metrics stay unavailable with their reason; edge diffraction is not modelled.
  - Edges between samples. A passing sample at the frontier of the live ones (on the outermost ring, or beside a
    sample retired earlier: its cell reaches the Gaussian's tail or an earlier cut's edge) whose distance to the
    edge is less than its spacing to its nearest live neighbour may lose part of its cell to an aperture that cuts no
    sample: its share is added to the resolution (once per cell along the path; `GEOMETRIC_POWER_BASIS`). An edge
    farther out than that spacing is taken to cut nothing. A collimated beam cut at 1 w and then at 0.95 w (between
    the outermost live ring at 0.915 w and the first edge) reads 0.8647 mW within 0.124 mW of the true 0.8355 mW.
    The light that passed a bounded opening lies inside it, so an edge that holds the image of an opening the beam
    passed earlier cuts nothing and adds nothing: the live samples map that opening's `(u, v)` to this surface's by
    their least-squares affine map; its largest residual over them, grown to the opening's edge as the cube of the
    ratio of the edge's distance to the farthest sample's from their centroid (a departure from affine that grows no
    faster than third-order aberration), is the margin by which the region must hold the opening's image (corners, or
    the ellipse holding a round region; a convex region less its holes, each hole inside the image of a hole of the
    opening), within `aperture_edge_tol_mm`. The same clear aperture on both faces of a plate, and a lens's rear face
    behind a front face that cut the beam, are such cases; an edge that may cut light that passed the earlier opening
    (a hole met again while the beam converges) keeps the addition.
  - Diffraction the rays do not follow. The pupil rays leave the source as the Gaussian beam's energy-flow tangents;
    through a paraxial system of transfer block `B` from the source the beam's radius is the rays' picture and
    `B theta` added in quadrature (`theta = lambda M^2 / (pi n w)` at the source plane, `B` from the differential
    rays). Where `B theta` exceeds a quarter of the radius the live samples picture (`sqrt 2` times their rms radius:
    the beam is then more than 3 % wider than its rays, a Fresnel number `w^2 / (lambda L)` below about 1.3 for a
    collimated beam) and the edge of an aperture -- or of an absorbing plane's active area -- comes within
    `aperture_quad_extent_sigmas` times the larger picture of the chief ray, the samples do not represent the beam
    there: the power behind that edge is not computed (the spread and the rms radius are in the issue), the beam goes
    on as geometry and the planes behind it record it with no power, at most what arrived (`power_upper_bound_W`). A 20 um waist at 780 nm cut at 20 mm and met by a 0.5 mm stop 30 mm later is such a case:
    its rays stay within 25 um while the beam is 0.62 mm wide there. Where the differential rays do not reach the
    surface (an axicon crossed, a differential ray retired) the spread is not known and the samples are taken to
    represent the beam.
  - An absorbing plane's active area on which the chief ray of such a beam falls detects the share of the live
    samples on it, with its sampling resolution (as a cut; `detected_power_W`, `power_sampling_resolution_W`); with
    no sample on it, or where the samples do not represent the beam there, the power it detects is not computed: the
    arrival is recorded with no power and its bound, and the plane ends the path (`DETECTED`). A transparent plane's active area decides only which beams count, as for a
    Gaussian beam.
  - A plane's power on this basis is a `KNOWN` sampled value with `power_sampling_resolution_W` (the sum over its
    arrivals) beside it; no upper bound is taken from it: `power_upper_bound_W` is null, or the launched power on an
    absorbing plane.
  An aperture that cuts every pupil sample while the chief ray passes leaves no sample to share the power: it is not
  computed behind that aperture (`OUT_OF_MODEL_SCOPE` issue with `power_upper_bound_W`; what arrived is booked once to
  `TERMINATED_OUT_OF_MODEL`) and the chief ray goes on alone as geometry. In each of these cases every later element
  passes at most what arrives, so the bound stays the power at the surface where it stopped being computed, on every
  branch; nothing more is booked for that beam, no power floor ends it, and no plane, pair or exposure behind reads a
  number from it. Without pupil rays the beam passes at its power as before.
* Chief ray beside the opening (a promoted reference ray). Every later decision -- the surface met next, the
  interaction coefficients, the plane crossings -- is taken on the reference ray (row 0, the chief ray). Where the chief
  ray falls on the opaque rim beyond a clear aperture while live pupil samples pass the opening, the beam goes on as
  those samples and the passing sample nearest their centroid (on the surface's `(u, v)`) becomes its reference ray from
  there: its position, direction, polarization basis, optical path and length replace the chief ray's, and its Jones
  vector carries the power that passes. That power is the Gaussian integral over the opening where the beam arrives with
  a valid Gaussian state (exact for the incident beam; the rest `APERTURE_TRUNCATION`, the part beyond the element's
  support handed on as light passing beside it), else the geometric pupil-sample share above with its resolution (and
  not computed where the samples do not follow the beam's diffraction). The light that passed is not a Gaussian beam
  about its new reference: the Gaussian state is invalid behind it (`GAUSSIAN_SEVERE_TRUNCATION`) and later edges take
  the geometric basis. The `CLIPPED` issue keeps its status and states how many pupil rays pass, the reference ray and
  the power; the arrivals behind it list the surface in `reference_ray_promoted_at` and report no first-order transfer.
  The reference ray is one sample of the light that passed, and where two passing samples are equally near the centroid
  (a mirror-symmetric pair) the lower row is taken: its hit point and direction are not properties of the beam (behind a
  2 mm stop with 2 of 96 samples passing, 0.30 mm off the design's symmetry plane while the light that passed is
  symmetric about it). A target on `position_u_mm`, `position_v_mm`, `direction_cosine_u`, `direction_cosine_v`, `aoi_deg`
  or `acceptance_margin_deg` behind it is therefore out of model scope (`CLIPPED`, INDETERMINATE); `spot_centroid_u_mm` /
  `spot_centroid_v_mm` give the centroid of the pupil samples that pass: the beam's position within the samples' resolution
  (behind a stop where 2 of 105 samples pass: 1.863 mm against the passed light's 1.779 mm, 0.22 of its rms half-width).
  The differential rays are relaunched about the reference ray, so the diffraction check above keeps a transfer: the
  chief ray's up to the promotion composed with the reference ray's from there -- an approximation within the
  aberration between the two rays, which the check's quarter-radius threshold absorbs. The group path keeps the chief
  ray's up to the promotion (two beams on this basis do not interfere: their Gaussian states are invalid). The beam
  keeps the whole-beam `CLIPPED` outcome (the part inside the opening then goes on as untraced light only) where no
  pupil sample passes, in a probe, on a facet patch, in a crystal leg, and where its power or polarization is not
  known. Edge diffraction is not modelled, as for every cut.
* Projected aperture at incidence angle `θ` on a circular aperture of radius
  `a`: the aperture, as seen by the beam, has semi-axes `a cosθ` (tangential)
  × `a` (sagittal).
* Optical path length of a segment in medium `n`: `n L`; total `Σ nᵢ Lᵢ`.

### Assumptions

Geometrical optics; homogeneous isotropic non-absorbing media (optional
constant bulk absorption coefficient); thin ideal lens = no thickness, no
aberration and a constant focal length (no chromatic behaviour); surfaces are
exact planes, spheres or sag profiles (conic + even asphere, rotational or
cylindrical; no surface form error or roughness).

Surface supports. A surface's support is its clear aperture grown by an opaque
mount ring (`mount_margin_mm`) or, when the outline is not the clear aperture
grown evenly (a rectangular plate, a decentred aperture, the face of a solid),
an explicit `mount_outline` in the same in-surface coordinates: a ray inside the
support but outside the clear aperture is blocked (`CLIPPED`), a ray beyond the
support passes by.

Catalog solids. The local runtime lowers a public catalog `SURFACE_SET` model
(a cube splitter, a prism, a wedged plate, an off-axis mirror section) to posed
planes and sag surfaces in the element frame: each surface carries the media of
its two sides (the body's glass inside, air outside); a coated internal plane
with the same glass on both sides is a beam splitter inside the glass
(`BeamSplitterInteraction`), an uncoated interface between two media that is
traced in both branches reflects its bare Fresnel fraction. Total internal
reflection at an uncoated face is the refraction law's own. Faces of the body
that carry no traced surface are not surfaces of the Engine: light reaching one
from inside stays in the glass medium and light crossing the body from outside
stays in air; the local runtime reports both (`RAY_LEFT_ELEMENT_THROUGH_EDGE`,
`RAY_CROSSES_ELEMENT_BODY`, the latter by exact clipping of the chief-ray
segment against the body's half-spaces).

### Applicability

Element sizes ≫ λ. Ideal lens: no claims about real lens aberration.
Refractive spheres and sag surfaces: exact ray tracing of the chief ray and the
pupil samples (their aberrations are real); the Gaussian beam through them is paraxial (P4),
and each arrival's Gaussian state is checked against its own pupil rays
(P4, assumption 2b).

Each packet follows its chief ray: the live pupil rays reach every physical
surface the chief reaches, and where the chief ray meets a physical surface or an
absorbing plane, a pupil ray that would first meet another such surface (or none)
is retired (`DIVERGED`). Observation planes are virtual: each records every live
ray that crosses it before the surface that would stop or turn that ray, where it
crosses it, and none moves, turns or retires a ray. The packet's power and Gaussian beam
remain those of the chief's path, so light of the beam that falls on a surface
the chief ray misses is not subtracted. The ray-path export
(`validate_with_ray_paths`) records, for every retired ray, the surface it would
have reached (`next_surface_id`) and whether it would otherwise have reached the
chief's surface inside its clear aperture (`counted_with_chief`: the packet still
counts it); the local catalog runtime reports such an interception as out of
model scope (`BEAM_PARTIALLY_INTERCEPTED`). The low-level `validate` marks the
power of an observation plane the chief ray crosses after part of the beam met a
physical surface on the same leg (a live pupil ray first, or the Gaussian share
that meets the surface the chief ray meets next) out of model scope there
(`power_out_of_scope_at`), and so does a reflection the leg began with for the
part of the beam it sends beyond the plane. Where every surface met first is the
one the chief ray meets next (a pupil ray meeting it first included), an
absorbing plane or the reflecting surface the leg began at, and each states a
bound `S_i` of the share it takes, the plane
reads the beam within `[P - sum S_i, P]` instead (`power_taken_before_at`,
`power_taken_before_upper_W`, plane `BOUNDED` with `power_lower_bound_W`).
For the surface the leg began at, `S_i` is the larger of the parallel-ray share
and the energy flow's (`decisions.start_flow_share`): every energy-flow line
within `K` radii meets the plane standing for the surface (its own plane; a
curved one shifted as far beyond the chief ray's start as it reaches for those
lines, a sphere also as its tangent plane with the cap's departure `l^2 / 2R`
from it) on a stretch `[o - h, o + h]` of the chief ray, `h = K |beta| W` with
`W` the widest radius there (behind the chief ray's start too; it exists while
`K |beta| theta < 1`), and it leaves the surface beyond the plane only where
`a + c . rho > 0` at that root; with each axis's radius at most its widest `M_i`
there, the share is at most `P(A > -a) / 2 + P(B > -a) / 2 + P(A + B > -a)`
(`A`, `B` the axes' `M_i c_i xi_i`), `P(A + B > -a)` alone for a profile that
scales as one, plus `exp(-2 K^2)`; it holds where the next surface lies beyond
that stretch (a line that leaves the surface before the plane then crosses it).
Without a valid Gaussian state, a pupil ray that does not cross the plane ahead
on the leg left the surface beyond it, and where the surface's support (its
bounding balls) may reach beyond the plane light may leave it there (the pupil
rays do not bound such a beam's footprint: near a waist they lie well inside it):
the plane's power is out of model scope. Likewise the surface the chief ray meets
next, where its support may reach the near side of the plane.
The same flow bound, for the lines that meet the plane standing for the next
surface before the plane, joins that surface's `S_i` (a tilted plane is
crossed by some lines beyond its chief-ray crossing, where the beam is wider).
Where the next surface stops what it meets from the beam's side (the back of an
uncoated mirror, an absorber), the lines that certainly meet it first before
the plane take at least `L` (`lower_share.certain_meeting_share`: the profile
within `K` radii in cells, the chief ray in stretches, each cell's lines boxed
as below; a cell counts when every stretch before some stretch keeps the
surface function `F` -- a plane's signed distance, `z - sag`, `sign(R) (|p -
C|^2 - R^2)` -- away from 0 or off the support, the stretch just before has
`F` of the stopping side's sign, and every stretch from there to one of the
other sign lies on the support and before the plane; none for a surface with
holes, and none where another surface counted only turns light -- a line it
turns may still cross the plane), and the plane reads
`[P - sum S_i, P - L]` (`power_taken_before_lower_W`). An absorbing plane
reads its detected power `D` within `[D - sum S_i, D]`: removing light from the
beam lowers the share on its active area by at most what is removed, while the
light certainly taken may be light that would have missed the area. A surface
beside the chief ray that a live pupil ray meets first, or whose bounding balls
come within `sqrt(2) K` beam radii of the leg (the tail no pupil ray samples),
states the share of the cells whose boxes may lie before the plane, on its
support, with `F` able to vanish (`lower_share.possible_meeting_share`; cells
left at the last cut count whole, plus `2 exp(-2 K^2)` beyond the `K` square; a
support other than a circle or a rectangle as the circle about its offset that
holds it), except a face of the body the beam travels in.
For the next surface, `S_i` is the larger of the parallel-ray share and the paraxial energy-flow share
(`decisions.leg_share`) of the plane standing for that surface and, for a sag
surface, the smaller of that and the region bound on the surface itself
(`_sag_region_share`: the beam's profile within `K` radii in cells, each cell's
energy-flow lines -- and the parallel rays at the widest radius -- enclosed in
boxes along stretches of the chief ray; a cell counts whole unless on every
stretch its box lies beyond the plane, misses the support or the profile's
domain, or keeps the surface's `z - sag` away from 0 over the sag's range there;
cells refined 4 times from an 8 x 8 grid, stretches halved up to 12 times). A
`power_W` target is judged on that interval (`POWER_TAKEN_BEFORE_PLANE`); the
other power-derived metrics stay out of model scope. An absorbing plane with an active area
stops a ray only where the ray crosses it on that area.
Such a diverged tail is also not part of the known-zero check of a plane no
traced beam reaches (P10): only the untraced branches listed there are.
Angle-of-incidence verdicts (grazing, TIR margin, thin-element warnings, and the
local runtime's coating / element angle ranges) judge the chief ray; a pupil
ray beyond the grazing limit or an Engine coating table is retired, not
reported.

### Numerical failure modes

Parallel/near-parallel ray-plane (`GRAZING_INCIDENCE`), tangent sphere hits
(`Δ≈0`), self-intersection after an interaction (`t_min`), zero-length or
non-finite direction (`ZERO_LENGTH_VECTOR`, rejected at input), degenerate
surface (zero normal, `R≤0`, zero aperture, a sag clear aperture beyond the
profile's domain or a `u_axis` not perpendicular to its normal →
`DEGENERATE_SURFACE`), sag intersections beyond the domain or not converged
within `sag_max_iterations` (reported as misses), TIR
boundary (`k≈0`: `TIR_BOUNDARY` warning), aperture-edge and floating-point
boundary (inclusive tolerance), path loops (`MAX_INTERACTIONS`: physical interactions only; observation
planes do not count).

### Independent references

* 45° mirror: `d̂' = R d̂` computed by the closed form `(1,0,0) → (0,1,0)` for `n̂=(−1,1,0)/√2`.
* Plano-convex thin-lens focal length from the lensmaker's formula
  `1/f = (n−1)/R`, versus exact refractive-sphere ray tracing (paraxial rays).
* Snell angles vs `arcsin(n₁ sinθ₁ / n₂)`; TIR critical angle `arcsin(n₂/n₁)`.
* Tilted collimated beam onto an ideal lens focuses at lateral offset `f tanθ`.
* Sag surfaces (`tests/test_sag_surfaces.py`): `k = 0` without terms traces
  exactly like the `SphereSurface` (refraction, reflection, Fresnel branch,
  Gaussian, pupil rays); a paraboloid focuses an on-axis collimated beam at
  `R/2` and a 90° off-axis paraboloid at its parent focus (reflected focal
  length `2f`) to 1e-9 mm; a plano-cylindrical lens matches the thick-lens
  ABCD matrix per axis (first-order transfer) and an exact 2D trace; an even
  asphere's intersections, normals and traced pupil rays match a 40-digit
  decimal reference (bisection on the sag equation, vector Snell).
* Two-variable sag shapes (`tests/test_freeform_surfaces.py`). Comparison case
  **FS-1** (`tests/freeform_reference.py`, standard-library `decimal` at 60 digits,
  sharing no code with the Engine or the contracts: the published polar formulas,
  the Chebyshev recurrence for `cos mθ`, a central-difference normal at step
  `1e-25` mm, plain bisection, vector Snell). A plano singlet in N-BK7, 5 mm thick,
  at 780 nm, whose rear surface is (a) `R = −80`, `k = −0.5` plus Noll Z4, Z5, Z6, Z7,
  Z8, Z9, Z11, Z22; (b) `R = −120` plus an XY polynomial of order 2–5; (c) a biconic;
  (d) a toroid with a `y²` term. Seven rays parallel to the axis, up to 8.5 mm off it,
  and three tilted rays; the hit point, the exit direction and the crossing of a plane
  75 mm behind. Stated tolerances, fixed before running: **1e-9 mm** on every
  position, **1e-11** on every direction component. Observed: positions within
  9e-13 mm and directions within 2.2e-16. Published-formula cross checks: Noll Z4–Z11
  and all 37 Fringe polynomials against the typed tables; Noll orthonormality over the
  unit disc for index 1–36 in exact rational arithmetic; a biconic of equal sections
  against the conic, an XY `x² + y²` against the paraboloid, a toroid of equal radii
  against the sphere and one without rotation against the cylinder (rays to 1e-12 mm).
  Astigmatism: `Z₆` of value `v` on a rear surface of curvature `c` has the section
  curvatures `c ± 2√6 v / R²` and, from glass of index `n` into air, the line foci
  `1 / ((1 − n)(c ± 2√6 v / R²))` behind the vertex; the traced rays reach them to
  1e-4 mm at a ray height of 0.02 mm and the Gaussian `q` of both sections matches
  the section's ABCD by hand to 1e-6 mm. Normals and principal curvatures against
  finite differences; the sag and Hessian bounds against a dense grid. Not compared:
  a commercial ray tracer (none is available to the test suite), shapes on a posed
  `SURFACE_SET` surface other than through the shared solver.
* Catalog solids and schema-3.0 products through the local runtime
  (`tests/test_catalog3_physics.py`): a cube splitter's 90° reflected port, its
  power split and a closed power budget (AR residuals as untraced reflection),
  the same split for a beam entering its fourth optical face (a combiner) or
  entering OUT_R backwards,
  the bare Fresnel transmission of an uncoated face; a polarizing cube's s / p
  split and Malus' law with a rotated input (extinction `T_p / T_s`); a
  right-angle prism's total internal reflection versus `arcsin(1/n)` (and no
  fold for a 1.38-index glass); a penta prism's 90° deviation under ±3°
  rotations; a dove prism's image mirroring and the 180° image rotation of two
  crossed dove prisms (first-order transfer); a wedge prism's deviation versus
  exact Snell and `(n − 1) α`; the erf product of a rectangular aperture's
  Gaussian truncation; an off-axis paraboloid's focus at its reflected focal
  length; a cylindrical lens's line focus; a typical spectrum's linear
  interpolation; the bare Fresnel reflection of a wedged sampler's front face;
  out-of-scope reports at a coating's band and angle edges (an angle range read in
  the glass) and for light crossing or leaving a body through an untraced face.

---

## P2. Materials, spectrum, coatings

References for the continuous spectrum: M. Born and E. Wolf, Principles of Optics, 7th ed., sec. 7.5.8 and 10.3; L. Mandel and
E. Wolf, Optical Coherence and Quantum Optics (1995), sec. 4.3. Checked in `tests/test_continuous_spectrum.py`: bin powers against
`math.erf` / `math.atan` and a dense trapezoid of the table; `|γ(D)|` against a dense NumPy integral (2e-8) and its 1/e
crossing against `brentq` (1e-6); a narrow Gaussian's `(2√ln2/π) λ²/F` and a narrow top hat's first `sin x / x = 1/e`;
the gamma difference falling by about four per doubling of the lines.

### Governing equations

* Constant index `n(λ) = n₀`, declared valid on `[λ_min, λ_max]`.
* Sellmeier: `n²(λ) = 1 + Σ_{i=1..3} Bᵢ λ² / (λ² − Cᵢ)`, λ in µm, `Cᵢ` in µm².
* Tabulated `n(λ)`: monotone PCHIP interpolation (shape preserving; no
  overshoot) or linear; no extrapolation.
* Bulk absorption: `P = P₀ exp(−α L)` (constant α, 1/mm).
* Coatings: `R_s, R_p, T_s, T_p` on a `(λ, AOI)` regular grid, the AOI being the
  physical angle of incidence in the medium the light arrives from; bilinear
  interpolation written out (`v₀₀(1−u)(1−v) + v₀₁(1−u)v + v₁₀u(1−v) + v₁₁uv`, only
  `+ − × ÷`: the same bits in CPython and Pyodide, and the same values as SciPy's
  linear `RegularGridInterpolator`). Power conservation is checked:
  `R+T ≤ 1 + tol` for each polarization. A table whose values do not depend on the
  AOI (every AOI column of a wavelength row equal) is evaluated as a 1-D linear
  interpolation over wavelength only (`np.interp`), so its R / T cannot change
  with the last bits of an AOI computed by another libm, and a wavelength inside a
  flat band returns the band value exactly.
* Project and catalog coatings lower to these tables: bands and spectra
  (unpolarized, or with their own s / p values) as AOI-independent tables, an
  angle-resolved table (`CoatingAngularSpectrumV1`) on its own grid. Every table is
  held flat beyond its outermost data (to 100 / 20000 nm and to 0 / 90°); the run's
  validity check reports a chief ray outside the coating's stated wavelengths or
  angles (`OUT_OF_MODEL_SCOPE`: `COATING_WAVELENGTH_OUTSIDE_BANDS`,
  `COATING_AOI_OUTSIDE_RANGE`, read in the incident medium for an angle-resolved
  table, never beyond its own angles). Left-out light that meets an air-equivalent
  record from a medium denser than air is read at the air-equivalent angles of its
  bounded extent, and a use outside the record is refused (P10a, the coating angle
  reference).
* Polarizers and retarders may carry wavelength bands of transmittance (and a
  polarizer's `t_⊥ = T / ER` per band): each spectral line uses the band holding its
  wavelength, the smaller transmittance on a shared boundary, the element's scalar
  values outside every band.
* A coating is evaluated on the rays the tracer still follows: a pupil ray
  already retired is evaluated at the chief ray's AOI (its row is stale). A live
  pupil ray outside the coating's angle range while the chief ray is inside it
  goes on with the coefficient at the nearest stated angle (one
  `COATING_OUT_OF_RANGE` WARNING per surface with `pupil_rays_held`); the beam's
  power is its chief ray's, so no reported power uses it. A beam on the
  geometric pupil-sample basis retires such rays. A chief ray
  outside the range goes on with the coating's values at the nearest stated
  wavelength and angle: where the beam goes does not depend on them; its power
  behind the surface is an estimate (`power_out_of_scope_at` names the surface,
  the power's `value_basis` entry states `HELD_AT_NEAREST_STATED_POINT`, the
  value the data was read at and the data's range), a power-derived target on
  it reports the value and is undecided, and a `COATING_OUT_OF_RANGE` notice
  names the path. A sampled ray of an extended emitter, a probe of untraced
  light and the rays of an analysis end there instead. A retarder, a Faraday
  rotator or an attenuator whose response at the line is not stated passes the
  light straight on with a power and polarization that are not computed (its
  issue states `power_upper_bound_W`): what arrived is booked once to
  `TERMINATED_OUT_OF_MODEL`, and each plane behind records the light with no
  power and the bound (a probe of untraced light ends there).
* Discrete spectrum: `{(λⱼ, wⱼ)}` with `Σ wⱼ = 1`; each component is traced
  independently with `n(λⱼ)`; powers `wⱼ P`. Different λ never interfere.
* Continuous spectrum (`SourceSpec.spectral_density`, `optics/interference/spectrum.py`): a density `S(λ)` over the vacuum
  wavelength -- Gaussian `exp(−4 ln2 (λ − c)²/F²)`, Lorentzian `1/(1 + (2(λ − c)/F)²)`, a top hat of width `F`, or a table linear
  between rows -- within a band `[a, b]` (default: a top hat its width, a table its rows, a Gaussian `c ± 1.5 F`, a Lorentzian
  `c ± 10 F`); `P` is the power in the band. Quadrature `MIDPOINT_BINS`: `n` equal bins of width `w = (b − a)/n`, line `k` at
  `a + (k + ½) w` with weight `∫_bin S dλ` (exact: error function, arctangent, the table's linear pieces), each a Gaussian line of
  FWHM `w` (`L_k = (2√ln2/π) λ_k²/w`). A quantity summed over the lines is `Σ P_k T(λ_k)`, the midpoint rule for
  `∫ S T dλ`: error at most `w²/24 · max|(S T)''| (b − a)`.
* Degree of coherence of a continuous source with itself (Wiener–Khinchin): `γ(D) = ∫ S(λ) e^{2πiD/λ} dλ / ∫ S dλ` over the
  band, composite Simpson (at least 2048 intervals and 64 per cycle of the phase across the band; a table's rows are nodes),
  normalized by the same rule (`γ(0) = 1`). `ValidationResult.source_spectra` reports the coherence length (smallest `D` with
  `|γ| = 1/e`, the definition of a line's `L_c`; a scan of 256 steps to 16 times the RMS-width estimate `√2 λ_c²/(2π σ)`, then
  bisection; none when `|γ|` stays above `1/e` over the scan), the same for the traced lines
  `γ_lines(D) = Σ P_k e^{−(D/L_k)²} e^{2πiD/λ_k} / Σ P_k` and for twice as many lines, and the largest `| |γ_lines| − |γ| |`
  over `0 … 3 L_c` for `n` and `2n` lines: the convergence of the quadrature (second order in `w` for a smooth spectrum).
  Lengths are rounded to 1 pm, differences to 1e-12.
* Two beams of a continuous source interfere with the coherence of the spectrum, not of a quadrature line (a line's width is its
  bin, a property of the discretization). Each line is taken as monochromatic within its line, carrying the beams' relative
  spectral phase `φ(k_line) + D (k − k_line) + B (k − k_line)²/2` (`D`, `B` their group-path and group-delay-dispersion differences
  at the line), and a line's pair reports its share of the whole spectrum's interference (`optics/interference/source_coherence.py`):
  the complex degree of coherence `γ = ∫ S e^{i[D (k − k_line) + B (k − k_line)²/2]} dλ / ∫ S dλ` of the declared density (composite
  Simpson, 64 intervals per cycle of that phase across the band, at least 2048; no value beyond 2²¹ intervals), so the coherence
  factor and the fringe visibilities are `|γ|` (with the line's beam balance, overlap and polarization), the coherence length is the
  spectrum's, and the combined power and the plane's coherent sum of a line are `P_a + P_b + 2 √(P_a P_b) Re[γ e^{iφ}]`, which sum
  over the lines to the source's where the beams' balance, overlap and polarization do not change across the band (exactly for a
  dispersion-free path difference). A coating's group-delay bound `B` moves `γ` by at most `B` times the spectrum's mean
  `|k − k_line|`. Two sources of one coherence group of which one states a continuous spectrum and the other a different one
  have no stated mutual coherence (`COHERENCE_UNDEFINED`). The exposure bounds count each line of a continuous source as fully
  coherent with its partners. Reference: `tests/test_source_coherence.py`, an ideal Michelson with Gaussian, Lorentzian and
  top-hat spectra against a dense NumPy integral of `γ` (coherence factor to 2e-7; 0.0446 at 30 µm on a 20 nm Gaussian at
  800 nm, where the line model gave 0.88), unchanged for 15, 31 and 61 lines.

### Applicability / failure modes

Outside `[λ_min, λ_max]` a material returns `MATERIAL_OUT_OF_RANGE` (`OUT_OF_MODEL_SCOPE`: a data-coverage limit
for the light behind it) — never extrapolated. Sellmeier poles (`λ² ≈ Cᵢ`), non-positive `n²`,
non-monotone wavelength tables, coating grid outside range, negative
absorption → structured errors. Coherence length undefined ⇒ coherence-based
metrics are `UNKNOWN`.

### Independent references

Published N-BK7 catalogue values: `n_F(486.13 nm)=1.52238`, `n_e(546.07)=1.51872`,
`n_C(656.27)=1.51432`, `n_d(587.56)=1.51680`; fused silica `n_d(587.56)=1.45846`
(Malitson). PCHIP interpolation of a linear table reproduces it exactly.

---

### Radial gradient-index media (`GradientIndexMaterial`, `optics/rays/gradient.py`)

A GRIN rod's medium has `n(r)² = n0² (1 − (g r)²)`, `r` the distance from its axis, `n0` its index on the axis (a constant,
Sellmeier or tabulated law) and `g` the gradient constant (constant, or linear between stated wavelengths). It fills the
space between exactly two end faces (below: planes across the axis, slanted planes, spheres and conics; `scene/lower.py`
refuses anything else: `GRADIENT_INDEX_UNSUPPORTED`). With `z` along the axis and the ray invariant `β = n dz/ds` (the medium does not vary along
`z`), the ray equation `d/ds (n dr/ds) = ∇n` is `x'' = −W² x`, `y'' = −W² y`, `W = n0 g / β`, so every ray, meridional or
skew, is exactly

* `x(z) = x0 cos(W z) + (p / W) sin(W z)`, `y` alike with its slope `q`; a paraxial ray (`β = n0`) has the period `2π / g`
  along the axis (pitch `P = g L / 2π`);
* the local index from the invariant: `n² = β² (1 + x'² + y'²)` (`x'² + W² x²` is constant on a ray);
* optical path `∫ n ds = (n0² / β)(Z − g² ∫ r² dz)`, `r² = A + B cos 2Wz + C sin 2Wz`, in closed form; geometric length
  `∫ (n / β) dz` and the group path `∫ n_g (n / β) dz` (`n_g = n − λ ∂n/∂λ`, `n = n0 √(1 − g² r²)`, at fixed geometry)
  by 8-point Gauss-Legendre panels of at most a quarter period of `r²`;
* the largest `r²` on the way (`A + √(B² + C²)` where the crest falls on the way): a ray reaching the rod's radius leaves
  through its side, which is not followed (refraction out or reflection back in). The chief ray doing so ends the path out of
  model scope. Pupil rays doing so are retired, and the beam's power behind the rod's exit face is then out of model scope
  (`power_out_of_scope_at`, `GRADIENT_INDEX_UNSUPPORTED` `OUT_OF_MODEL_SCOPE` with the retired samples' share): with a valid
  Gaussian state the beam keeps its power, never a known value behind the rod; on the geometric pupil-sample basis their
  share leaves the beam, booked `TERMINATED_OUT_OF_MODEL`. Where no pupil ray reaches the side but the beam's envelope does
  (its rays at `K` beam radii, beyond which its share past any line is negligible, from its waist's position and angle
  spread, followed in closed form), the same holds. Reference (`tests/test_gradient_index.py`, the ray equation integrated by
  RK4 over 12 000 rays sampled from the beam's Wigner distribution): the 0.02 mm beam tilted 0.5 rad in a 1 mm rod leaves
  44 % through the side (the samples: 40 of 96), 0.49 rad 1.7 % (no sample; the envelope), 0.45 rad none (nothing flagged);
* polarization carried along the ray without turning about it (the Rytov law, parallel transport), by the double-reflection
  rotation-minimizing frame on 256 points per period;
* faces: Snell's law and the Fresnel coefficients with each ray's own index `n(r)` where it meets the face;
* the Gaussian beam: its first-order transfer about the chief ray (central differences of the closed form, the beam's axes
  carried along the chief ray), `q'/n_out = (A q/n_in + B/n_in)/(n_out C q/n_in + n_out D/n_in)` per axis; on the axis this is
  the GRIN matrix `((cos gL, sin gL/(n0 g)), (−n0 g sin gL, cos gL))` (Marchand, *Gradient Index Optics*, 1978, ch. 3). A transfer
  that couples the beam's axes (a skew chief ray) puts an astigmatic beam's Gaussian state out of model scope
  (`GAUSSIAN_TWISTED_ASTIGMATISM`); a round beam is taken on the chief ray's meridional plane;
* the paraxial analysis uses the same GRIN matrix for the rod's gap (`paraxial_matrix`): effective focal length
  `1 / (n0 g sin gL)`, back focal distance `cos gL / (n0 g sin gL)` from the exit face.

The chief ray's leg is recorded as 16 straight pieces per period of its path, each with its own exact length and optical
path (every piece but the last ends inside the medium where the next begins: the local runtime's edge check reads them as one
run from end face to end face, so a beam that leaves through the exit face is within scope); the ray-path export keeps each
pupil ray's end-face points. References (`tests/test_gradient_index.py`): the ray
equation integrated by SciPy's DOP853 (rtol 1e-12) for meridional and skew rays over 1.7 periods (points and directions
to 1e-9, optical path and length to 1e-11 relative, polarization against integrated parallel transport to 1e-9); the paraxial
period; the Gaussian `q` against the GRIN matrix by hand at pitches 0.23, 0.25, 0.29 and 0.5 (rel 1e-7); the paraxial matrix
(1e-12).

### Axial gradient-index media (`AxialGradientIndexMaterial`, `optics/rays/gradient.py` `AxialLegs`)

`n(z) = n0 + a1 z + a2 z² + a3 z³`, `z` along the axis from the axis point (a Project rod: the medium's entrance on local Z),
the same across the axis; `n0` a constant, Sellmeier or tabulated law, the terms the same at every wavelength. The index does
not vary across the axis, so a ray keeps its transverse optical direction `k = n t⊥` and stays in the plane of the axis and
`k`; with `R(z) = √(n² − |k|²)`:

* `x(z) = x0 + k_x ∫ dz / R`, `y` alike; the length `∫ n / R dz`, the optical path `∫ n² / R dz` and the group path
  `∫ (n − λ ∂n0/∂λ) n / R dz`, by Gauss-Legendre quadrature (16 panels of 8 nodes over the leg; smooth integrands);
* the least index over an interval exactly (the ends and the roots of `dn/dz`): a ray whose angle to the axis exceeds
  `acos 0.05` (87.1°) (near its turning point `n = |k|`) is not followed, and a medium whose index falls below 1 within its rod at
  the run's wavelength ends the path (`GRADIENT_INDEX_UNSUPPORTED`);
* the polarization in closed form: the component across the ray's plane is kept, the one in it turns with the ray (parallel
  transport on a plane curve);
* the faces with the local index `n(z)`; the Gaussian beam by the leg's first-order transfer as for a radial rod (the
  meridional plane is the plane of the axis and the chief ray); the paraxial gap `((1, ∫_0^L dz / n), (0, 1))` with the
  index `n(L)` at the back face.

References (`tests/test_gradient_index_axial_faces.py`): the ray equation integrated by DOP853 for meridional and skew rays
through 12 mm of a cubic profile (points 1e-9, directions 1e-10, optical path and length 1e-10 relative -- the integration's
own error -- and exact `∫ n dz` on the axis to 1e-14); the Project rod's chief ray, power (s and p Fresnel with the field
carried along), Gaussian `q` against `i z_R + ∫ dz / n + δ` (1e-9) and the paraxial matrix of a lens with curved faces
(1e-12).

### End faces of a gradient rod that are not planes across its axis (`GradientFaceG`, `gradient.to_face`)

A face is a plane at any slant (a `SURFACE_SET` round body: an angled face) or a sphere / conic of revolution centred on the
axis whose profile is defined out to the rod's radius (a `SURFACE_SEQUENCE`). In the rod's frame the face spans
`[z_min, z_max]` within the rod's radius and has a largest slope `|dz/dr|`; a ray inside the rod is before the face at the
span's near end and beyond it at the far end, and `sign (z − z_face(x, y))` grows by at least `1 − slope s` per mm of axial
travel (`s` a bound on the ray's slope across the axis: radial `|(x', y')|² ≤ p² + q² + W² r0²`, axial `|k| / √(n_min² − |k|²)`),
so for `slope s < 1` the ray crosses the face once and 64 bisections find it; otherwise the ray is not followed (the chief ray:
the path ends `GRADIENT_INDEX_UNSUPPORTED`). The faces must not reach past each other along the axis within the rod's radius.
The Gaussian transfer starts and ends on the faces' tangent planes at the chief ray; a skew chief ray behind a slanted face
couples the beam's axes (`GAUSSIAN_TWISTED_ASTIGMATISM`). References (`tests/test_gradient_index_axial_faces.py`): the
crossing against the ray equation stopped by an event on the face (radial and axial, slants 8° and −25°, a sphere, a conic;
1e-9); rods with spherical faces and a slanted entrance face from the Project: chief ray, optical path, power and exit
direction against the integrated ray equation, vector Snell and s/p Fresnel (1e-9); the Gaussian `q` and the paraxial matrix
of a rod with spherical faces against the hand matrix product.

## P3. Fresnel coefficients, power, OPL, phase

### Governing equations

For a dielectric interface (`θ₂` from Snell):

```
r_s = (n₁cosθ₁ − n₂cosθ₂)/(n₁cosθ₁ + n₂cosθ₂)       t_s = 2n₁cosθ₁/(n₁cosθ₁ + n₂cosθ₂)
r_p = (n₂cosθ₁ − n₁cosθ₂)/(n₂cosθ₁ + n₁cosθ₂)       t_p = 2n₁cosθ₁/(n₂cosθ₁ + n₁cosθ₂)
R_{s,p} = |r|²,   T_{s,p} = (n₂cosθ₂)/(n₁cosθ₁)|t|²,   R+T = 1
```

(basis triad: `ŝ = d̂×n̂/|d̂×n̂|`, `p̂ = d̂×ŝ`, see P5.) Under total internal reflection `cosθ₂` is evaluated
as `+i√(n₁²sin²θ₁/n₂² − 1)` (decaying evanescent wave for `exp[i(k·r−ωt)]`); the same Fresnel
formulas then give `|r_s| = |r_p| = 1`, `T = 0` and the TIR phases.
Phase: `φ = k₀ Σ nᵢ Lᵢ + Σ arg(interface coefficients)`; `phase_rad` of an arrival is
`k₀·OPL + arg(Eₓ)`, or `+ arg(E_y)` where `|E_y|²` exceeds `|Eₓ|²` by more than 1e-9 relative
(`DOMINANT_COMPONENT_REL_TOL`), i.e. relative to the source phase reference; phases
of different (mutually incoherent) sources are never compared.

**Stated coating phases** (`InterfacePhases` on a refracting, reflecting or splitting interaction; a Project or catalog
coating's `CoatingPhaseV1`): `r = √R e^{iφ_r}`, `t = √T e^{iφ_t}` per polarization in the same triad (`ŝ = d̂×n̂`
normalized, `p̂ = d̂×ŝ` before and `d̂_out×ŝ` after), field `exp(−iωt)`; each phase constant or tabulated over
wavelength (linear, held flat beyond the table by the Project lowering, never extrapolated at the Engine level: at a
wavelength outside the table the interface states no phase, its own phases apply as without stated phases, the pass is
recorded as one whose phase is not stated, and a `COATING_OUT_OF_RANGE` issue, `OUT_OF_MODEL_SCOPE`, names the surface
and the path). A mirror uses the stated reflection phases instead of `(π, 0)`, a splitter the stated
reflection / transmission phases instead of `(π/2, r_p = −r_s; 0)`, a coated refracting face the stated transmission
(and, for its reflected light, reflection) phases instead of the bare Fresnel ones; under total internal reflection the
Fresnel amplitudes apply. Without stated phases every default below holds unchanged. Catalog coatings state powers only
(`COATING_PHASE_NOT_MODELLED`).

Ideal mirror/partial reflector coefficients come from a `CoatingModel`
(amplitude `√R`, declared phase). Symmetric lossless beam splitter (`BeamSplitterInteraction`, non-polarizing or
polarizing by its coating): `t_s = √T_s`, `t_p = √T_p`, and in the `(ŝ, d̂_out×ŝ)` triad of the reflected beam
`r_s = i√R_s`, `r_p = −i√R_p` on both sides (`reflection_phase_rad` is the phase of `r_s` relative to `t`; `r_p = −r_s`
for equal `R`). Unitarity `t r* + r′ t* = 0` holds for each polarization. `r_p = −r_s` is a physical reflection, the same
convention as the ideal metal mirror `(π, 0)` and as Fresnel at normal incidence: a non-polarizing splitter (`R_s = R_p`)
reflects every polarization exactly like a mirror at the same pose, times `−i√R` (no half-wave-plate action between s
and p); a polarizing one scales the s and p components of that mirror reflection by `√R_s` and `√R_p`. At normal
incidence, where `ŝ` is the ray's arbitrary `e₁`, a non-polarizing splitter's reflected field `i√R E_in` does not depend on
that basis (`tests/test_splitter_reflection_convention.py`: source-frame rotation, 45° splitter vs the vector mirror law,
both-side unitarity). A **polarizing** coating (`R_s ≠ R_p` or `T_s ≠ T_p`, a splitter's or any coated surface's) met at
normal incidence still depends on it: which field component it treats as s is the ray's internal `e₁` (lab-vertical
light through a PBS at normal incidence transmits 1, 0.75 or 0 for source frames rolled 0°, 30° or 90°), and within a
small angle above `NORMAL_INCIDENCE_SIN` (10⁻⁶) the plane of incidence is set by a misalignment. The run does not flag it; such a layout
is outside the coating model's meaning (real coatings have `R_s = R_p` at normal incidence), and the linear optical
network export refuses a coating model that is polarization-dependent at normal incidence (a constant `R_s ≠ R_p`) met
within 1° of it (`POLARIZING_NORMAL_INCIDENCE`, `docs/engine/LINEAR_OPTICAL_NETWORK.md`).

Engine 0.6.x gave `r_p` the phase of `r_s` (the reflection acted as a mirror followed by a half-wave plate on the s / p
axes). What changes with 0.7.0: the reflected field of p, diagonal or circular light changes the sign of its p component;
the power of that beam alone does not, but every result that depends on the relative phase of its s and p components
does: the power after a polarization-sensitive element downstream of a splitter reflection (diagonal light reflected by a
50:50 splitter at a 45° fold, then a polarizer at −45°: none of the reflected power passed before, all of it now), the
reflected beam's polarization state, and the interference of beams that reach a plane after different numbers of splitter
reflections unless the light is s-polarized at each of them (the p part of the difference gains π). Light that is
s-polarized at every splitter it is reflected by gives the same results as before. In the stored golden snapshots only the
sign of `polarization.ey` and π on `phase_rad` of p light reflected by splitters changed, no power, interference or
optical path value.

### Assumptions

Sharp planar dielectric interfaces or declared coatings; polarization
decomposed into s/p about the actual 3D plane of incidence.

### Failure modes

`n₁cosθ₁ + n₂cosθ₂ → 0` (grazing), TIR boundary, `T>1` from bad tables,
zero power (`ZERO_POWER`, path terminated), negative/NaN power rejected.

### Independent references

Normal incidence `R = ((n−1)/(n+1))²`; Brewster angle `θ_B = arctan(n₂/n₁)` gives
`r_p = 0`; `R+T=1`; critical angle TIR; OPL of a slab `n t` vs the sum over
segments; MZI OPD `= Σ nL` difference from geometry. Fresnel rhomb: two TIRs in N-BK7 at the angle where
`tan(δ/2) = cosθ √(sin²θ − 1/n²) / sin²θ = tan(π/8)` give a quarter wave of s / p retardance and turn 45° linear light
circular (`tests/test_polarization_extensions.py`); stated phases against the field decomposition
`E_out = r_s (E·ŝ)ŝ + r_p (E·p̂_in)p̂_out` and a tilted window's angle-resolved table against independent bilinear
interpolation at each face's own angle (`tests/test_coating_tables_and_phases.py`).

---

## P4. Gaussian beam (ABCD / q-parameter)

### Governing equations

Per transverse axis `a ∈ {x, y}`: `q_a = z_a + i z_{R,a}` with
`z_{R,a} = π w₀ₐ² n /(M²ₐ λ₀)`; `1/q = 1/R − i M²λ₀/(π n w²)`.

```
w(z)  = w₀ √(1 + (z/z_R)²)         θ_∞ = M²λ₀/(π n w₀)      R(z) = z(1 + (z_R/z)²)
propagation over L in medium n:      q' = q + L
flat refraction n₁→n₂ (normal):      q' = (n₂/n₁) q
ideal thin lens (normal):            1/q' = 1/q − 1/f
```

Oblique incidence at angle `θ` (per-axis principal directions: tangential `t`
= in the plane of incidence, sagittal `s`), effective powers (Coddington):

```
refraction, spherical surface R (R>0: centre ahead of the surface):
  1/q'_t = (n₁cos²θ₁ / n₂cos²θ₂)(1/q_t) − (n₂cosθ₂ − n₁cosθ₁)/(n₂cos²θ₂ R)
  1/q'_s = (n₁/n₂)(1/q_s)              − (n₂cosθ₂ − n₁cosθ₁)/(n₂ R)
  flat: R→∞. Anamorphic factor w'_t = w_t cosθ₂/cosθ₁.
spherical mirror (R_c>0 concave):   1/q' = 1/q − P,  P_t = 2/(R_c cosθ),  P_s = 2cosθ/R_c
sag surface: 1/R replaced by the local normal curvatures κ_t, κ_s along the
  tangential / sagittal surface directions at the chief-ray hit (refraction:
  positive when the centre of curvature is ahead along the ray; mirror / TIR:
  positive when concave):
  1/q'_t = (n₁cos²θ₁ / n₂cos²θ₂)(1/q_t) − (n₂cosθ₂ − n₁cosθ₁) κ_t / (n₂cos²θ₂)
  1/q'_s = (n₁/n₂)(1/q_s)              − (n₂cosθ₂ − n₁cosθ₁) κ_s / n₂
  mirror: P_t = 2κ_t / cosθ,  P_s = 2κ_s cosθ
ideal thin lens at incidence θ:     1/q' = 1/q − cosθ/f   (both axes; slope-form lens is stigmatic)
plane mirror:                        q' = q
```

Beam axes are carried as a right-handed transverse frame `(e₁, e₂)` with
`e₁×e₂ = d̂` along the chief ray (parallel transport at non-polarizing deflections;
`(t, s)` mapping at interfaces, P5 for the polarization basis). The axis *labels* follow the
physical axes: `q_x` always belongs to the axis that started as the source's local `x`.
Intensity `I(r) = (2P/(π w_x w_y)) exp(−2x²/w_x² − 2y²/w_y²)`.

Aperture truncation transmission `T = ∫_A I dA / P` evaluated by deterministic
quadrature over the *projected* aperture on the surface plane (elliptical
footprint, offset and turned axes allowed): a circular aperture met by a nearly
round footprint (aspect ratio ≤ 2) in polar form about the beam centre
(closed-form radial integral, trapezoid / Gauss–Legendre in angle); a circular
aperture met by an astigmatic footprint in the beam's whitened principal frame
(closed-form `erf` integral over each chord, Gauss–Legendre panels along the
other axis); a rectangular aperture with the square completed along one of its
axes (closed-form `erf` integral along it, Gauss–Legendre panels along the
other; a beam whose axes are the rectangle's is the product of two `erf`
intervals). The panels follow the scale of the factor that varies fastest on
each piece, so a line focus (100:1 and beyond) is integrated as accurately and
about as fast as a round beam. Circular aperture, centred round beam closed
form: `T = 1 − exp(−2a²/w²)`; rectangle with the beam's axes: `erf` products
(both used as tests).

### Assumptions and applicability

Four of the checks are accuracy limits of the first-order model, each held by a caller setting
(`numerical_settings`): the far-field half angle (`gaussian_scope_divergence_rad`, `GAUSSIAN_HIGH_NA`), the beam
radius against a surface's radius of curvature (`gaussian_max_w_over_radius`, `GAUSSIAN_SURFACE_TOO_STRONG`), the
angle of incidence on a focusing surface (`gaussian_max_aoi_deg`, `GAUSSIAN_LARGE_AOI`) and the share an aperture
truncates (`gaussian_clip_scope_fraction`, `GAUSSIAN_SEVERE_TRUNCATION`). Beyond one of them the state is not
valid (`valid` false, status `OUT_OF_MODEL_SCOPE`, the code as `reason`) and stays so for the rest of the path: no
later element makes it valid again, and no radius, waist, `q`, Rayleigh range, angle or curvature is published for
it. The state states the failed check in `validity` (`parameter`, `observed`, `maximum`, `setting`, `at`: the
surface or source where the limit was first exceeded, `code`). Everything that needs a Gaussian beam (power integrals
over apertures, the irradiance map, the coherent sum, the handoff) reads a valid state only; a target on a Gaussian
metric, and a Gaussian alignment metric, has no value, status `OUT_OF_MODEL_SCOPE`, the code in `issue_codes` and
the same check in its `validity`. With a limit raised by the caller so that the state is valid, the numbers are
published with a `value_basis["*"]` entry of kind `MODEL` that lists the checks passed under that limit. Where no
first-order state exists (twisted astigmatism, an axicon or facets, a central obstruction, an anisotropic medium, a
gradient-index rod met by a state that is not valid, a harmonic beam) the state has the code only. Pupil rays launched
at the source go on whatever the Gaussian state becomes; a source whose own state is beyond the divergence limit
launches none.

1. Paraxial, moderate NA: `θ_∞ ≤ gaussian_warn_divergence_rad` else WARNING; `>
   gaussian_scope_divergence_rad` → `OUT_OF_MODEL_SCOPE` (`GAUSSIAN_HIGH_NA`).
2. Well-defined optical axis and aberration not dominant, checked twice:

   a. at every curved surface: beam radius ≤ `gaussian_max_w_over_radius` ×
      |R| and AOI ≤ `gaussian_max_aoi_deg`, else `OUT_OF_MODEL_SCOPE`
      (`GAUSSIAN_SURFACE_TOO_STRONG`, `GAUSSIAN_LARGE_AOI`). This is necessary
      but not sufficient: a beam well inside it can still be focused into an
      aberration-dominated spot (a fast singlet turned the wrong way round).
   b. at every arrival on an observation plane, against the Engine's own pupil
      rays (`validation/observations.py::gaussian_ray_spread`). In an
      aberration-free first-order (ABCD) system a Gaussian beam is exactly the
      two-ray (complex-ray) decomposition `w² = x_A² + x_B²` per axis (`x_A`:
      the ray launched at the 1/e² radius along the wavefront curvature,
      `x_B`: the diffraction ray from the axis), and the pupil samples are
      launched on the source envelope at `a_i w` (P1 sampling), so every
      geometric sample ray is `a_i x_A` and, on every transverse plane, the RMS
      spread of the rays about their centroid is at most `√m_a · w` per axis
      (`m_a` = the second moment of the live rays' launch offsets along that
      axis). The Engine divides by the rotation-invariant `√m · w`,
      `m = mean(|a_i|²)/2`: the aberration-free ratio is then ≤ 1 for a
      complete or symmetrically truncated sample (`m_a = m`) and ≤ √2 for any
      subset of rays and any rotation of the Gaussian axes (`m_a ≤ 2m`). The
      Engine measures that ratio per Gaussian axis on the observation plane
      (rays carried to the plane through the chief-ray hit point along their
      own directions) and on the plane of that axis' Gaussian waist (rays
      continued in straight lines in the plane's medium, exactly as the
      Gaussian's own waist is). Above `gaussian_max_ray_spread_ratio` (2) the
      arrival's Gaussian state is withheld: `GAUSSIAN_ABERRATION_DOMINATED`,
      `OUT_OF_MODEL_SCOPE`, with the ratio, the plane and both RMS values; its
      Gaussian metrics (radius, waist, divergence, capture margin, active-area
      fraction, interference) become `OUT_OF_MODEL_SCOPE`, its power is still
      reported. Meaning of the limit: it lies above √2, the largest value an
      aberration-free beam can reach whatever pupil rays were lost on the way,
      so a flagged arrival is aberrated; with the full sample its rays spread
      at least twice as wide as any aberration-free beam with that Gaussian
      envelope can, i.e. the aberration's own RMS spread is ≥ √3 times the
      Gaussian's and the real beam is at least ~2× the Gaussian radius there.
      Below the limit the Gaussian numbers carry the aberration unflagged (the
      real second-moment radius is up to about `√(1 + r²)` times the Gaussian
      one, `r` the ratio); they are paraxial values, not diffraction-limited
      guarantees, and `pupil_rms_radius_mm` gives the rays' own spot.
      Waist-derived metrics read at a plane far from the waist are validated
      through the waist-plane check.
3. Truncation: power loss at an aperture `> gaussian_clip_warn_fraction` →
   WARNING; `> gaussian_clip_scope_fraction` → the Gaussian model is no
   longer valid (`GAUSSIAN_SEVERE_TRUNCATION`, `OUT_OF_MODEL_SCOPE`); the
   downstream Gaussian state is marked invalid, never continued as Gaussian.
4. **Untwisted astigmatism.** x/y decoupling requires principal axes to
   coincide with `(t,s)` at every interface. If the beam is round
   (`|q_x−q_y|` and `|M²x−M²y|` within tolerance) the frame is rotated onto
   `(t,s)` exactly. Otherwise a misaligned interface produces twisted
   (general) astigmatism → `OUT_OF_MODEL_SCOPE` (`GAUSSIAN_TWISTED_ASTIGMATISM`).
   A sag surface adds its own axes: its principal directions at the chief-ray
   hit (P1) must lie on the plane-of-incidence directions on the surface
   (sagittal `ŝ`, tangential `n̂×ŝ`) within `gaussian_astig_align_tol`, unless
   the surface is umbilic there (`|κ₁−κ₂| ≤ gaussian_umbilic_rel_tol·max|κ|`:
   spheres, a conic's vertex); at normal incidence the beam axes must lie on the
   principal directions (a round beam is rotated onto them). A skew hit on an
   off-axis conic or a cylinder rotated against an astigmatic beam is
   therefore `GAUSSIAN_TWISTED_ASTIGMATISM`. `GAUSSIAN_SURFACE_TOO_STRONG`
   compares the beam radius with the smallest local radius of curvature
   `1/max|κ|`; `GAUSSIAN_LARGE_AOI` applies as for spheres.
5. Plane-of-observation curvature: `w` is evaluated at the axial position
   where the chief ray meets the plane; if the axial spread across the
   footprint exceeds `gaussian_plane_axial_spread_fraction × z_R` a warning is
   issued.
6. Diffraction at hard apertures and pinholes is **not** propagated
   (see out-of-scope registry).
7. Axicon surfaces: the Gaussian state is invalid behind a cone
   (`GAUSSIAN_UNDEFINED_AFTER_AXICON`, P1).
8. Tabulated thin lens: the ideal thin-lens ABCD step with the line's own
   `f(λ)`; the model's validity domain and validated metrics bound what it
   means (P1).

### Numerical failure modes

`q → ∞` (waist at the plane, `1/q` purely imaginary is fine; collimated `q`
with `z_R=∞` when `M²→0` rejected), `1/q = 0` exactly (flat wavefront:
handled by working with `Γ = 1/q` and `q = Γ⁻¹` only when finite), negative
`w²` from bad user data (validated), astigmatic `x/y` swap ambiguity.

### Independent references

* Thin lens on a collimated waist: `w₀' = w₀ f/√(f²+z_R²)`, waist at
  `d' = f/(1 + (f/z_R)²)` (closed form derived independently of the ABCD code).
* Keplerian/Galilean expander: 2×2 matrix product `L(f₁+f₂)·lens₂·…`
  computed in the test with plain `numpy` matrices; ideal `M = |f₂/f₁|`.
* Tilted spherical mirror: `f_t = (R/2)cosθ`, `f_s = (R/2)/cosθ` (textbook);
  cross-checked by *numerical ray tracing* of a small pupil to find the
  tangential and sagittal foci.
* Tilted refractive spherical surface: Coddington equations vs finite-aperture
  ray-trace foci.
* Sag surfaces: paraboloid mirror waist vs the analytic ABCD `1/q' = 1/q − 2/R`;
  90° off-axis paraboloid: meridional / parallel curvatures
  `c/(1+c²ρ²)^{3/2}`, `c/(1+c²ρ²)^{1/2}` at 45° give `P_t = P_s = 1/(2f)`;
  plano-cylindrical lens `(q_x, q_y)` vs the thick-lens ABCD of each axis;
  principal curvatures of an even asphere vs finite differences of a 40-digit
  reference sag.
* Truncation: `T = 1 − exp(−2a²/w²)`.

---

## P5. Polarization (Jones formalism, 3D transverse basis)

### Governing equations

Each ray carries a Jones vector `(E₁,E₂)` in the ray's transverse basis
`(e₁,e₂)`, `e₁×e₂ = d̂`. The 3D field is `E = E₁e₁ + E₂e₂` (complex vector).

Free propagation / non-polarizing deflection (ideal lens): rotation-minimizing
(parallel) transport. For `d̂ → d̂'` with `d̂ ≠ −d̂'`: `Rot` = rotation about
`d̂×d̂'` by the angle between them; `e_i' = Rot e_i`; Jones components are
unchanged. For `d̂' ≈ −d̂` (retroreflection) the rotation axis is chosen
deterministically as `e₁`.

Interfaces (refraction, reflection, PBS): `ŝ = d̂×n̂/|d̂×n̂|` (perpendicular to
the plane of incidence), `p̂ = d̂×ŝ`... The Jones vector is re-expressed in
`(ŝ, p̂_in)` by the rotation `ψ` between `e₁` and `ŝ`; the interaction acts as
`diag(a_s, a_p)` (Fresnel or coating amplitudes); the output basis is
`(ŝ, p̂_out)` with `p̂_out = d̂'×ŝ` (`ŝ` made exactly perpendicular to `d̂'`); if
`|d̂×n̂| ≤ 10⁻⁶` (`NORMAL_INCIDENCE_SIN`: no resolved plane of incidence) `ŝ:=e₁` (the s and p coefficients differ there
by at most first order in that angle). The cross product of two unit vectors is known to a few 10⁻¹⁶, so below that
bound its direction would be set by rounding: a design turned as a whole about an oblique axis, or written with rounded
decimal positions, meets an exactly normal surface at 10⁻¹⁶–10⁻¹⁰ rad. `e₁` is carried from the source's frame through
every interaction, so it turns with the design and the run is the same for a design and for that design turned rigidly
(`tests/test_rotation_invariance.py`). The Gaussian tangential / sagittal split uses the same bound (normal-incidence
rules below it).

Elements (in the ray's transverse plane; element axes projected onto it):

```
polarizer, transmission axis a:      J = a aᵀ
retarder, retardance δ, fast axis a: J = e^{−iδ/2} a aᵀ + e^{+iδ/2} b bᵀ   (b ⟂ a)
Faraday rotator, rotation θ about z: J = R(sign(d̂·z) θ)                    (section P10c)
Diffraction grating, one order: J = diag(√η_TE, √η_TM) in (groove, across)  (section P10d)
HWP: δ=π; QWP: δ=π/2; retardance scales δ(λ)=δ₀ λ₀/λ (zero-order) or achromatic
PBS (thin ideal): transmit p with amplitude √T_p, reflect s with amplitude i√R_s (+ leakage: reflected p −i√R_p,
  transmitted s √T_s); s = ⟂ plane of incidence about the PBS normal (P3: r_p = −r_s convention).
```

Polarization overlap of two arriving fields (unit vectors `ê₁, ê₂` in 3D):
`η = |ê₁* · ê₂|` (amplitude form); intensity factor in fringe terms.

Source polarization (Project `GaussianBeamSourceV1.polarization`, in the source's local `(X, Y)`, propagation `+Z`,
field `exp[i(k·r − ωt)]`): linear at `α`: `(cos α, sin α)`; elliptical with orientation `θ` (major axis from `+X` toward
`+Y`) and ellipticity angle `χ` (`tan |χ|` = minor / major, sign of `s₃ = sin 2χ`): `R(θ)(cos χ, i sin χ)ᵀ` =
`(cos θ cos χ − i sin θ sin χ, sin θ cos χ + i cos θ sin χ)`. `χ < 0` is right-handed: the field turns clockwise for
an observer looking toward the source (`χ = −45°`: `(1, −i)/√2`, `JonesVector.right_circular`). It is the sign of the
results' `polarization_ellipticity_deg` (`asin(s₃)/2`): a source of ellipticity `χ` arrives with `χ` through free
space. The trigonometric
values are power series in exact decimals (no libm), so the hashed Engine input is identical in every runtime; `χ = 0`
is the linear state bit for bit.

A partially polarized source sharing a coherence group with another source is coherent with it through its polarized part only:
its unpolarized part has no fixed phase relation to another source's field, so the time average of their product is zero. Its
component 0 (power `(1 + P)/2`, along `J`) pairs with the other source's beams with the line's degree of coherence times
`√(2P/(1 + P))` (component 1 has a group of its own); an unpolarized source is coherent with no other source, whichever way it is
turned about its axis. Two beams of one source keep the line's coherence. The two components of one line are counted per plane
on their own (`PARTIAL_POLARIZATION_COMPONENTS`, not `incoherent_pairs`).

Partially polarized sources (`PARTIAL`: polarized part `J` and degree of polarization `P`; `UNPOLARIZED`: `P = 0`, `J`
along local `+X`; `optics/polarization/partial.py`). The coherency matrix of a line of power `L` is
`C = L [P J Jᴴ + (1 − P)/2 I] = L [(1 + P)/2 J Jᴴ + (1 − P)/2 J⊥ J⊥ᴴ]` with `J⊥ = (−conj J_y, conj J_x)`. Every element
here acts by a Jones matrix `M`, so `M C Mᴴ` is the sum of the two components propagated as fully polarized beams: each
line launches one root per component (`<source>#<line>` and `<source>#<line>:p1`), of powers `(1 + P)/2 L` and
`(1 − P)/2 L` (exact shares: `w₀ = (1 + P)/2`, `w₁ = 1 − w₀`), both booked to the line's one ledger entry. The
components' amplitudes are uncorrelated: the second gets a coherence group of its own (the source's group + `:p1`), so
superpositions, irradiance maps and exposures add their powers, and two-beam pairs are formed within one component only
(`PARTIAL_POLARIZATION_COMPONENTS` for a target naming one beam of each). Light left out as negligible, or of unknown
fate, is counted whole with both components' groups in the exposure bounds (an upper bound). On a plane, the arrivals of
one line along the same interactions at the same crossing are one route: `S = Σₖ Pₖ (1, s₁ₖ, s₂ₖ, s₃ₖ)` in the first
arrival's basis, degree of polarization `|S₁..₃| / S₀`, and the polarized part's ellipse from a Jones vector built from
`(S₁, S₂, S₃)/|·|` with square roots only (`ObservationResult.polarization_mixtures`). A route missing one component is
`UNKNOWN` (`INCOMPLETE_TRACE`) only when the path budget left part of that line untraced. A fully polarized source
(`P = 1` included) launches one root, bit for bit as before. The linear optical network export and the assembly routing's
polarized response take fully polarized sources only (the export refuses, the routing uses the unpolarized share).

### Uniaxial birefringent crystals (`UniaxialMaterial`, `optics/polarization/uniaxial.py`)

A crystal medium states its ordinary and extraordinary indices (each a constant, Sellmeier or tabulated law over the
material's band) and its unit optic axis `c`, in the scene frame or in a component's frame (`optic_axis_component`: it
turns with that component's pose). With wave vectors `K` in units of `k0` (`|K|` the phase index along the wave
normal), `A = 1/n_o²`, `B = 1/n_e²`:

* ordinary wave: `|K| = n_o`, ray along `K`, field along `c × K`;
* extraordinary wave: `Kᵀ M K = 1`, `M = B I + (A − B) c cᵀ` (`1/n(θ)² = cos²θ/n_o² + sin²θ/n_e²`), ray (Poynting)
  direction `S = M K / |M K|`, walked off `K` by `ρ`,
  `tan ρ = (n_o² − n_e²) sin θ cos θ / (n_e² cos²θ + n_o² sin²θ)` (away from the axis for `n_e < n_o`), field along
  `S × (c × K)`;
* at a plane face (normal `N`) the tangential `K_t` of the incident wave is kept: an outgoing wave is `K_t + q N`, `q`
  from `|K|² = n²` or from the quadratic `(A − B)(a + q b)² + B(|K_t|² + q²) = 1` (`a = K_t·c`, `b = N·c`; the larger root
  leaves the face, the smaller returns); no real root: that wave is evanescent (total internal reflection of that wave).
  An isotropic face, a crystal face (into or out of a crystal) and a crystal-crystal face (Wollaston, Rochon) are one rule:
  up to two transmitted and two reflected waves, each a beam of its own along its ray;
* a beam in a crystal travels along `S`; its optical path per mm of ray is `K·S = n(θ) cos ρ` (`RaySegment.refractive_index`
  with `crystal_mode`), every pupil ray its own; its group index along the ray is `n_g(θ) cos ρ`, `n_g(θ) = n − λ ∂n/∂λ|θ`
  (the ray path is stationary, so the change of `K` with `λ` does not enter to first order), its dispersion
  `λ³/(2π) ∂²n/∂λ²|θ cos ρ` at the same fixed geometry, as for an isotropic medium.

Power at a face: the incident field, taken transverse to its wave normal, is split on the orthonormal pair `(u_o, u_e)`
(`u_o` the ordinary outgoing wave's field projected transverse to the incident wave normal); each part passes with the
isotropic Fresnel transmittance (or the face coating's) at the outgoing wave's index for its own share of `s` and `p`, the
reflection takes the rest (the face conserves power exactly) and is split again on the reflected waves' pair. This is the
exact plane-wave result at normal incidence for every optic-axis direction (each eigenwave's tangential fields then obey the
isotropic boundary conditions at its own phase index) and at every angle where each crystal's axis is perpendicular to
the plane of incidence (s light meets `n_e` only, p light `n_o` only). Elsewhere it is an approximation: against the exact
4 × 4 boundary solution (calcite, `tests/test_birefringent_crystals.py`) the wave powers differ by at most about 2.6e-4 of
the incident power per degree of incidence up to 10°, 0.023 at 30°; a face met beyond `birefringent_interface_warn_aoi_deg`
(2°) warns `BIREFRINGENT_INTERFACE_APPROXIMATE` once per face.
Along the optic axis (the wave normal within `AXIAL_SIN_TOL` = 1e-6, sine, of it) the two waves are one: both have the index
`n_o` and the same wave vector and ray, and every field transverse to the axis is an eigenwave. The light crosses such a face as
at an isotropic medium of index `n_o` and travels on as one wave carrying its whole field (`crystal_mode` `ORDINARY`), split
only where it leaves into a crystal whose axis is not along it (a Rochon prism's second prism). Within the band the waves'
indices differ by at most `|n_o³ (1/n_e² − 1/n_o²)| / 2 · AXIAL_SIN_TOL²` (2e-13 for calcite) and the extraordinary ray walks
off by at most `|n_o²/n_e² − 1| · AXIAL_SIN_TOL`. Its Gaussian beam has parts of both kinds (`D_o = 1/n_o`, `D_e = n_o/n_e²`),
carried with their mean and half their difference as the stated error. Reference (`tests/test_crystal_axis_along_beam.py`,
an independent 4 × 4 plane-wave boundary solver): the catalog Rochon rolled (eight rolls, and the bench at eight turns the solids check accepts), and the
source frame rolled, give each beam's direction, power and field to 1e-12. Outside the band the waves are split on the crystal's own basis and traced apart.
A face that splits each of two waves of one earlier split (both above the line's negligible level, `_Packet.coherent_twin`) again into two
waves feeds each of its waves from both, coherently (near the axis their phase indices differ by about `(n_o − n_e) θ²`, so they stay one
field over the crystal): the trace does not carry their relative phase, and the power behind that face is out of model scope
(`power_out_of_scope_at`). A face that passes each wave into one wave of the next crystal mixes nothing and is traced as before. A
non-crystal surface that passes the two waves' polarizations by powers differing by `D` (a polarizer's two axes, a coating's or a bare
interface's s and p, any diffraction grating taken as `D` = 1) passes their sum with a cross term up to `D` times the line's power that
depends on their relative phase: above the line's negligible level the power behind it is out of model scope (`Tracer._twins_polarized`).
Overlap is not tested (a displacer's separated beams count too). Isotropic surfaces at normal incidence, retarders and planes add the two
orthogonal waves' powers exactly.

Gaussian beam: an ordinary beam diffracts as in an isotropic medium of index `n_o`. An extraordinary beam's transverse
wave-vector spread `κ ⊥ S` adds `−κᵀ M κ / (2 |M K|)` to its longitudinal wave vector: its two field axes diffract like
isotropic media of indices `1/D₁`, `1/D₂` (`D₁ = B/|MK|`, `D₂ = (B + (A − B)|c⊥|²)/|MK|`). The state is carried with
`2/(D₁ + D₂)`; the footprint maps across each face as for a grating order (`apply_grating`, twisted astigmatism refused);
the accumulated `|D₁ − D₂|/2 × L` (reduced mm) beyond `birefringent_gaussian_rel_tol` (1 %) of the reduced Rayleigh range
puts the Gaussian state out of model scope (`GAUSSIAN_ANISOTROPIC_MEDIUM`; rays and powers stay traced).

Refused before tracing (`BIREFRINGENT_MEDIUM_UNSUPPORTED`): a curved surface bounding a crystal, a crystal face that does
not refract (a mirror, splitter or thin element on it), a source or emitter in a crystal, an observation plane declared in
one; light inside a crystal meeting another element ends there out of model scope. Biaxial crystals and optical activity
are outside the model; the linear optical network export refuses crystal faces, and the assembly routing does not follow a
line through them.

In a Project a crystal is a declared material with `birefringence` (`ProjectModelMaterialV1`: `dispersion` the ordinary index,
`birefringence.extraordinary` the extraordinary one, the same forms and band), and every `SURFACE_SEQUENCE` or `SURFACE_SET` model
made of it states the optic axis in its own frame (`ProjectOpticalModelV1.optic_axes`, one per crystal material; every face of
the crystal is a plane). An observation plane whose `medium` is a declared crystal is refused by `project check` and before a
run (`BIREFRINGENT_MEDIUM_UNSUPPORTED`, location `observation <id>`): light is observed outside crystals. The local runtime
lowers each element's crystal to its own `UniaxialMaterial`
`project:<MATERIAL_ID>@<instance>` (a compound member: `@<instance>-<member>`, its axis turned by the member's pose) whose axis is
given in the component's frame, so it turns with the component. A body holds one medium: a cemented pair of crystals with
different axes (Wollaston, Rochon, Glan-Thompson) is a `COMPOUND` unit of single-crystal members with a joint (below), and
air-spaced prisms are a `COMPOUND` of single-crystal members without one (native `polarizer.glan`).

### Cemented crystal prisms (`CompoundModelV1.joints`)

Two `SURFACE_SET` members of a unit whose plane body faces face each other are joined (`MemberJointV1`). An optical contact
(faces coincident within 1e-6 mm, front normals opposite within 1e-6) becomes one interface: member A's face with member B's
medium on its front side, so light crosses from one crystal into the other by the face rule above (a Wollaston or Rochon
interface between two crystals cut differently). A cement joint keeps both faces, parallel and more than 1e-6 mm and at most
1 mm apart, with the declared isotropic cement between them: light is refracted (or totally reflected: the Glan-Thompson's
ordinary wave) at each face of the layer; the layer's thin-film interference is not modelled. Faces that do not meet as
declared are refused when the unit is compiled (`OPTICAL_MODEL_INVALID`). References (`tests/test_cemented_prisms.py`): the
Wollaston's and Rochon's exit directions against the vector Snell law at the joint and the exit face (1e-12; every wave there
travels perpendicular to or along its crystal's optic axis, so its index does not depend on its direction), the
Glan-Thompson's transmitted power against the exact s-Fresnel product of its four faces (1e-9; its optic axes are
perpendicular to the plane of incidence). The local runtime's edge check (`RAY_LEFT_ELEMENT_THROUGH_EDGE`) takes the body of
a segment from its medium: behind an optical contact the next member, bounded by its faces and the shared face; in a cement
layer the layer, bounded by the two joined faces. A beam that leaves through them is within scope; one that leaves the next
prism or the layer through its side is reported against that body.

### Catalog polarizing prisms (`JOINED_BODIES`, catalog schema 3.5)

Catalog schema 3.5 describes polarizing prisms (forms `GLAN_TAYLOR_PRISM`, `GLAN_LASER_PRISM`,
`GLAN_THOMPSON_PRISM`, `WOLLASTON_PRISM`, `ROCHON_PRISM`) as a `JOINED_BODIES` model: two or more `SURFACE_SET` bodies stated
in the element frame, each with the optic axis of its crystal (`optic_axes`), joined by an optical contact or a cement layer
(`joints`, the same rules as `MemberJointV1`) or apart across air. The crystal is a published material with `birefringence`
(its extraordinary index law; `dispersion` is the ordinary index). The local runtime compiles the model as a `COMPOUND` unit of
its bodies at the identity pose: surfaces `<body>-<surface_id>` (`A-S1`), each body's crystal the `UniaxialMaterial`
`catalog:<MATERIAL_ID>@<instance>-<body>`, the joints applied as above; the 3D view draws the product as an `OPTICAL_UNIT` of its
bodies (basis `PUBLIC_CATALOG_DESIGN_GEOMETRY`). A joined product is not copied into a Project as one declared model
(`catalog_model` refuses it; a Project declares the same prism as a `COMPOUND` unit). References
(`tests/test_catalog_polarizing_prisms.py`, on a synthetic test catalog): a Glan-Thompson's transmitted power against the exact
s-Fresnel product of its four faces (rel 1e-9); a Glan-Taylor's (optic axes in the plane of incidence at its air gap) against
the exact transverse-magnetic transmittance of a uniaxial / isotropic face from the boundary conditions, times its coating's
transmittance (rel 1e-9); Wollaston and Rochon exit directions against the vector Snell law (1e-12); the catalog Wollaston
against the native `splitter.wollaston` of the same geometry (directions 1e-12, powers rel 1e-12).

### Retardance tables, circular polarizers, rotatable axes (`PolarizationElementModelV1`)

A retarder whose `retardance_dispersion` is `TABULATED` states its retardance per wavelength (`retardance_rows`): linear in
wavelength between rows or at the rows' wavelengths only; elsewhere the light passes the plate with a power and
polarization that are not computed (`MODEL_WAVELENGTH_OUTSIDE_BAND` with `power_upper_bound_W`). A `CIRCULAR_POLARIZER` is its linear polarizer
and, `retarder_offset_mm` behind it, a retarder whose fast axis is `retarder_axis_deg` from +X: light entering on the
polarizer side leaves circular, light returning through the retarder (a mirror behind it) is turned by 90° and blocked by
the polarizer (returned power `P T² / ER`). `position_parameter = axis_rotation_deg` turns the element's polarizer and
retarder axes by the component's selected angle (in a `COMPOUND` unit, the member that names it): a tunable splitter's
half-wave plate.

### Assumptions / applicability

Jones calculus: fully polarized beams, and partially polarized sources as two incoherent Jones components (above). No
depolarizing element, stress birefringence, biaxial crystal, optical activity, high-NA vector fields; the fringes of a
partially polarized mixture (both components together) are not computed. Waveplate/
polarizer are treated in the thin limit at the incidence direction; large
AOI on a retarder gets a warning (`RETARDER_OBLIQUE_INCIDENCE`, threshold
config).

### Failure modes

Normal-incidence `ŝ` singularity, retroreflection basis flip, non-unit or
zero polarization input, loss of orthonormality through accumulated
transport (re-orthonormalized with a reported deviation), polarizer extinction
giving zero power.

### Independent references

* Malus's law `I = I₀cos²φ` through two polarizers.
* Uniaxial crystals (`tests/test_birefringent_crystals.py`): a calcite displacer's offset `L tan ρ` against the analytic walk-off
  at 30°, 45°, 48°, 60° (to 1e-12 mm; 6.2° at 45°) and its beams' powers against the normal-incidence Fresnel values (rel 1e-12);
  quartz's small walk-off toward its axis; at oblique entry the ray directions and phase indices of a test-local solver of the
  wave-normal equation `K × (K × E) + ε E = 0` (1e-12); Wollaston and Rochon directions from Snell's law in their principal
  sections (1e-13); an air-spaced Glan prism's exact transmittance; a plate with its axis in the face as a retarder
  (`(n_e − n_o) L` optical path difference); the oblique split against the exact 4 × 4 boundary solution.
* Crystals declared in a Project (`tests/test_project_crystals.py`, `tests/test_native_crystals.py`): a calcite beam displacer
  (Sellmeier indices evaluated in the test) splits two beams displaced by `L tan ρ` (1e-12 mm) with the normal-incidence Fresnel
  powers (rel 1e-12); a partially polarized source's degree of polarization before it (`P`) and of each beam behind it (1) with
  powers `(1 ± P)/2` times the faces' transmittance; two displacers turned 90° apart; the native Glan-Foucault's transmitted
  power against the exact s-polarized Fresnel product (rel 1e-9) and the Glan-Taylor's polarization.
* Retardance tables, circular polarizer, rotation parameter (`tests/test_polarization_families.py`): plain-numpy Jones
  products, the isolated return `P T² / ER`, Malus's law `cos²(2θ)` of a turned half-wave plate.
* Partial polarization (`tests/test_partial_polarization.py`): Mueller matrices `A (J ⊗ J*) A⁻¹` of each element applied
  to `S₀ (1, P s)` (unpolarized light and partial Malus's law through a polarizer of finite extinction, a half-wave and a
  quarter-wave plate), and a Michelson whose partially polarized results equal its two fully polarized runs weighted
  by `(1 ± P)/2`.
* HWP at axis `α` rotates linear polarization by `2α` (analytic Jones product).
* QWP at 45° turns linear 0° into circular.
* HWP + PBS: transmitted power `cos²(2α)` for input polarized at 0° with PBS
  passing 0°, reflected `sin²(2α)`.
* A 3D fold sequence: transport of the reference polarization along a
  non-planar path compared with an independent construction of the composed
  rotation (`s`-polarization stays `s` across a mirror in a plane; after a
  second non-coplanar mirror the s-vector direction equals the closed-form
  `d̂₂×n̂₂` construction); orthonormality preserved to `1e-12`.

---

## P6. Two-beam analytic interference

### Governing equations

Two arrivals `j=1,2` on an observation plane with normal `ν̂` at chief-ray
points `x_j`: directions `d̂_j`, medium index `n`, wavelength `λ₀` (must be
equal), powers `P_j`, Gaussian states, fields `ê_j`.

```
crossing angle           α = arccos(d̂₁·d̂₂),  half angle θ = α/2
wavevector difference    Δk = (2πn/λ₀)(d̂₁ − d̂₂),   |Δk| = (4πn/λ₀) sin(α/2)
3D fringe spacing        Λ₃ = 2π/|Δk| = λ₀ / (2n sin(α/2))
in-plane fringe period   Λ_p = 2π/|Δk_∥|,  Δk_∥ = Δk − (Δk·ν̂)ν̂
   symmetric about the normal: Λ_p = λ₀/(2 n sinθ)   (Λ = λ/(2 sinθ))
OPL_j, OPD = OPL₁ − OPL₂ (mm), coherence:  |γ(OPD)| = exp[−(OPD/L_c)²]
   L_c := OPD at which |γ| = 1/e; from a Gaussian spectrum of FWHM Δλ:
   L_c = (2√(ln2)/π) λ₀²/Δλ.  coherence_margin = 1 − |OPD|/L_c
plane irradiance of one beam I_j(u) = (2P_j/(π w_{x,j} w_{y,j})) exp(−2 u·S_j u)... (transverse-flux
   intensity of the plane-wave component; see below)
```

`I_tot = I₁ + I₂ + 2√(I₁I₂)·η_pol·Re[γ exp(i(Δk_∥·u + Δφ))]`, with
`Δφ = k₀(OPL_a − OPL_b) − arg⟨E_a, E_b⟩` (`⟨E_a,E_b⟩ = conj(E_a)·E_b`, 3D fields) and `γ` the complex degree of
coherence relative to `Δφ`: where the two paths' group-delay dispersions differ by `B` (glass on one path only), the
Gaussian line's `γ = (1 − i b)^(−1/2) exp(−(D/L_c)²/(1 − i b))`, `b = 2B/L_c²`, `D` the group path difference, whose phase
shifts the fringes; without a dispersion difference the real `|γ|`. The coherence factor, the visibilities and the peak
irradiance read `|γ|`.
For two (nearly) collinear beams (fringe period `> interference_collinear_period_ratio ×` beam
diameter) the detector power is `P_a + P_b + 2√(P_aP_b) η_pol η_sp Re[γ exp(iΔφ)]`
(`combined_power_collinear_W`); the fringe period is then undefined (`NO_FRINGES`, `NOT_APPLICABLE`). Where `Δφ` is open
(below) they have the range over every phase, `P_a + P_b ∓ 2√(P_aP_b) η_pol g η_sp`, at least 0, `g` the largest degree of coherence
the group-delay bounds allow (`combined_power_collinear_min_W` / `_max_W`, `phase_statement`). The ordinary `power_W` and
`total_power_W` add `P_a + P_b` whatever the phase; where a coherent quantity exists on the plane `power_basis` says so
(`INCOHERENT_SUM`).
The values at the pair's peak position -- `peak_irradiance_W_per_mm2`, `mean_irradiance_W_per_mm2`, the exposure time
from the mean and the local contrast `fringe_visibility_peak` -- are the light on the plane there, to which light of the
pair's wavelength the trace does not follow adds: a run with a target on one of them follows that light, and the target
is graded on the range it leaves -- with `F` the largest field amplitude that light may have anywhere on the plane (each
crossing of followed light at its Gaussian peak, or the diffraction limit of its power where its Gaussian state is not
valid; the light left out as negligible at the diffraction limit of its in-phase power times its open paths' crossing
field, at least 2, and the plane's return factor), an irradiance `v` lies in `[max(0, √v − F)², (√v + F)²]` and the contrast between its values at the
ends of the peak's and minimum's ranges (`UNTRACED_LIGHT_BOUNDED`, `FAINT_LIGHT_BOUNDED`; light of unknown fate: no upper
bound, the contrast between 0 and 1). The power-weighted `fringe_visibility` is graded the same way over the whole
plane: with `P = P_a + P_b` the two beams' power there and `R` the largest field amplitude that light may add across the
plane as the square root of a power (the same as `coherent_power_W`'s), the contrast as one beam's phase varies lies
between `max(0, V − 2ρ)/(1 + ρ)²` and `min(1, (V + 2ρ)/(1 − ρ)²)`, `ρ = R/√P` (1 when `ρ ≥ 1`; Cauchy–Schwarz and
Minkowski over the plane). The other two-beam metrics describe how the two beams interfere, which no other light
changes.
Two beams that cross the plane in opposite directions (`d̂₁·ν̂` and `d̂₂·ν̂` of opposite sign: a beam and its own
retro-reflection) have no cross term in the power through the plane — the time-averaged Poynting flux of anti-parallel
fields has none; their standing wave modulates the energy density along the beam, not the plane — so
`combined_power_collinear_W` is `P_a + P_b` for them, and the coherent superposition and the irradiance grid
(`observations[].superpositions`, `results irradiance`) form cross terms only between beams crossing the plane in the
same direction.
On an absorbing plane with an active area, `P_a` and `P_b` in `combined_power_collinear_W`, and every `P_j` of the
coherent superposition (`coherent_power_W`), are the powers the plane detects (each beam's power on its active area,
`arrivals[].detected_power_W`), each field scaled by `√(detected/incident)`: a single beam's coherent power is its
`power_W`.

Spatial envelopes: on the observation plane each beam has a Gaussian
intensity `I_j = A_j exp[−2 (u−c_j)ᵀ S_j (u−c_j)]` where `S_j` is the
inverse-covariance-like matrix obtained by projecting the beam's transverse
ellipse onto the plane (`u = c_j + P_⊥⁻¹ r_⊥`); the plane cuts the beam
obliquely, so lengths stretch by `1/cos α_j` along the tilt. Envelope overlap
integral (closed form, Gaussian product):

```
Q_j(u) = (u−c_j)ᵀ S_j (u−c_j)
∫ exp[−(Q₁+Q₂)] du = π/√det(S₁+S₂) · exp[−(c₁ᵀS₁c₁ + c₂ᵀS₂c₂ − bᵀ(S₁+S₂)⁻¹b)],  b = S₁c₁+S₂c₂
spatial_overlap_ratio = ∫√(I₁I₂)du / √(P₁P₂)       (∈[0,1]; 1 for identical envelopes)
local (peak) visibility  V_loc = 2√(I₁I₂)/(I₁+I₂) · η_pol · |γ|   at the envelope-product maximum u* = (S₁+S₂)⁻¹b
integrated visibility    V = 2 ∫√(I₁I₂)du /(P₁+P₂) · η_pol · |γ|
intensity_balance        min(I₁,I₂)/max(I₁,I₂)  at u*
mean irradiance          Ī = I₁+I₂ (fringe average) at u*;  peak = Ī + 2√(I₁I₂)η_pol|γ|
exposure time            t = dose / Ī  (dose in mJ/cm², Ī converted from W/mm²)
```

Irradiance here is the plane-wave transverse intensity `|E|²` normalized so a
single beam integrates to its power over its transverse plane. This is the
quantity relevant to absorption in a thin resist (∝|E|²); it is *not* the
Poynting flux through the substrate plane (which would carry an extra `cosα`).

### Coating group-delay bounds

A coating states no group delay unless the catalog (schema 3.2) or the design publishes a bound for it:
`group_delay_bound_fs`, the largest `|dφ/dω|` of its reflection and transmission, s and p, over the wavelengths and
angles its data cover (a reflection counted from the coated face, a transmission from crossing it). Each pass through such
a coating carries its bound (`arrivals[].group_delay_unknown_at`: `...<=<bound>fs`). Two beams' passes that agree cancel;
each pass one beam makes and the other does not adds its bound, so the relative group delay is
`τ ∈ [τ_traced − B, τ_traced + B]`, `B` the sum (as a group path, `c × delay`; `group_delay_bound_mm`).

The degree of coherence is then bounded for any coating dispersion within the bounds: the relative spectral phase
differs from the traced one by `ψ(k)` with `|ψ'(k)| ≤ B`, so `|e^{iψ(k)} − e^{iψ(k₀)}| ≤ B|k − k₀|` and

```
|γ − γ_traced| ≤ B · ⟨|k − k₀|⟩ = 2B / (√π L_c) = δ        (Gaussian spectrum of L_c: ⟨|k − k₀|⟩ = 2/(√π L_c))
```

(`γ_traced` the complex degree of coherence at the traced delay, the coatings' phases at the line taken as the model's).
The coherence metrics are stated at every `δ` (the interval is the statement): `coherence_factor` at the traced delay
with `coherence_factor_min/max = |γ_traced| ∓ δ` (within [0, 1]); the visibilities and the peak irradiance's excess over
the mean follow it linearly, `coherence_margin` the delay interval. A target on one of them is judged on that interval
(met at both ends: decided; failed at both: FAIL; otherwise UNKNOWN).

The bound limits `ψ'(k)`, not `ψ(k₀)`: the reflection / transmission phase at the line of a coating whose phases are
not stated is open (the trace's default convention is not a statement about it). `|γ|`, the visibilities (fringe
contrast, wherever the fringes lie) and the peak irradiance (the fringe maximum) do not depend on `ψ(k₀)`; the power of
overlapping beams through a plane does, through `2 Re(γ e^{iψ₀} …)`. So the coherent superposition states
`total_power_W` with `total_power_min_W/max_W` -- each pair's cross term within `2δ_jk |∫ E_j* E_k|·flux` of its traced
value and within `±2|∫ E_j* E_k|·flux` -- only where every bounded pass that differs is a coating stating its phase at
the line (a tabulated phase: its value is stated, its derivative is not formed; the pass record carries `#phase`).
Otherwise the pair's cross term may take any phase: `total_power_min_W/max_W` add `∓2 min(1, |γ| + δ) |∫ E_j* E_k|·flux`
(the minimum at least 0) and a `coherent_power_W` target is judged on that range, while `total_power_W`,
`interference_power_W` and `combined_power_collinear_W` are stated at the trace's default interface phases with
`value_basis` (`ESTIMATE`, `UNSTATED_PHASE_AS_INTERFACE` naming the passes). The default phases conserve energy between a
splitter's two output ports (the 2 × 2 splitter is unitary for each polarization): the two ports of a Mach–Zehnder add up
to its input at every arm length (reference: `tests/test_undetermined_group_delays.py`), and a Mach–Zehnder whose
coatings state exactly the default phases gives the same power (`tests/test_bounded_coating_coherence.py`). The two-beam
`phase_difference_rad` follows the same rule without a bound: where a pass one beam makes and the other does not has no
stated phase at the line (an element without a thickness, a coating without a stated phase, bounded or not), it is the
value at the default phases with that entry, and a target on it is reported and not decided; where such a pass also
states no optical path (`path_open`: an ideal or tabulated thin lens, a Faraday rotator, a grating) it is None
(`PHASE_UNDEFINED`, informational, `phase_statement` naming the passes). Passes both make alike cancel, and a coating pass
that states its phase (record `…#phase`) keeps the value without an entry (reference: `tests/test_phase_difference_open.py`).
The fringe period and spatial overlap are geometry and do not depend on it. An oblique coating of unstated phase also
changes the polarization of a beam that meets it in a state mixing s and p (its s and p phases differ by an unstated
amount; the pass record carries the state): there the polarization overlap and the visibility are stated on the default
phases with an entry naming those passes, and a coherent power's range takes every overlap up to 1. A differing pass that
bounds no delay: the metrics at no added delay there (`UNSTATED_GROUP_DELAY_ZERO`), the ranges and bounds taking any
delay. A pulse train keeps the metrics without a value (`COHERENCE_UNDEFINED`) only where a revival is not negligible
anywhere within `B` of the traced delay (the Gaussian line's coherence at the closest approach; a continuous spectrum's
coherence need not fall with the delay, so with a bound it is not stated). The irradiance map and the exposure bounds keep treating differing
passes as undetermined (fully coherent in the bounds). The coatings are read at the line's wavelength, as for their
power coefficients; the bound holds where their data hold (a use outside their bands or angles is reported out of model
scope).

Catalog class bounds (the catalog's coating class group-delay bounds; a supplier states no group
delay): dielectric high reflector 1000 fs, beamsplitter 1000 fs, metallic reflector 100 fs, anti-reflection 50 fs;
interference filters (dichroic, bandpass, edge-pass, notch) and neutral-density coatings publish none (their edges and
cavities reach picoseconds near their bands). Basis: transfer-matrix group delays of representative stacks over their
high-reflectance or specified band at 0 and 45 deg, s and p -- quarter-wave TiO2/Ta2O5/HfO2–SiO2 reflectors ≤ 16 fs
(R ≥ 99 %) and ≤ 26 fs (R ≥ 95 %); stacked or graded broadband reflectors 235–330 fs, 780 fs at the resonance between two
stacked stacks; few-layer and graded broadband splitters ≤ 7 fs and ≤ 45 fs; protected metals ≤ 2 fs, enhanced aluminium
≤ 11 fs; one- and four-layer anti-reflection stacks ≤ 2 fs. The group delay grows towards and beyond a stop band's edge:
the bound holds inside the specified band only.

References: `tests/test_bounded_coating_coherence.py` -- the reference interferometer (`two_beam_interference`, 780 nm,
0.001 nm linewidth: `L_c` = 322.46 mm; plate group-path excess 0.97960 mm from Snell's law and the substrate's Sellmeier
group index; `B` = 4050 fs = 1.21416 mm) gives `|γ|` = 0.99999077 and `V = 2√(P₁P₂)/(P₁+P₂)|γ|` = 0.99997815, independent
of the Engine, with `γ ∈ [0.99574, 1]`; a 50 mm coherence length leaves `δ` = 0.027 (undetermined); a Mach–Zehnder of two
plates leaves its coherent power open over `[0, 4P]` with its coatings' phases unstated, and with stated phases gives
the bright and the dark port (`4P` and `0`, within `2Pδ`) as only BS2's reflection phase changes by `π`, both inside
that range; and `δ` itself is checked by integrating the spectrum against relative phases of bounded slope (step,
oscillating and random slopes).

### Assumptions / applicability

Two mutually coherent quasi-monochromatic Gaussian beams treated as plane
waves for the fringe phase (curvature ignored across the overlap region);
requires `|Δz| ≪ z_R` across the footprint (check), same λ, same observation
plane, paraxial Gaussian validity per P4. Not wave-optics propagation.
Independent lasers (different coherence group) ⇒ `V = 0` reported
`NOT_APPLICABLE` for coherence metrics (they do not interfere), never PASS.

### Failure modes

`α→0` (`Λ→∞`: fringe period undefined → `UNKNOWN`), `α→π`, `d̂₁≈d̂₂`, missing
coherence data (`UNKNOWN`), different wavelengths, non-overlapping beams
(`spatial_overlap_ratio ≈ 0`), singular `S₁+S₂`, polarization orthogonal
(`η_pol=0` → visibility 0, FAIL if targeted).

### Independent references

* 405 nm, θ=arcsin(405/(2·500)) = 23.8863°: symmetric about normal ⇒ `Λ=500.000 nm`
  (closed form; the engine reaches it via *traced* 3D geometry, mirrors placed by the test).
* Two s-polarized beams: `η_pol=1`; two p-polarized symmetric beams: `η_pol=cos(2θ)`.
* Equal intensities, coherent, aligned: `V=1`. `V(OPD)=exp[−(OPD/L_c)²]`.
* Identical Gaussian envelopes at the same centre: overlap = 1; separated
  by `d` with equal `w`: `exp(−d²/(2w²)·…)` (closed form in the test).
* Numerical 2D quadrature of the overlap integral.

---

## P7. Layout (2D/2.5D, Shapely)

Footprints `F` (polygon in the element's local XY) are rotated by the pose
yaw and translated: `F_g = R(ψ)F + p_xy`. Element height interval
`[z_min, z_max]` (global, from pose z + local extents). Two elements collide
if `F_g,i ∩ F_g,j ≠ ∅` **and** height intervals overlap. Keepout = polygon
dilation by `k` mm (or explicit keepout polygons); required clearance `c`;
`clearance = dist(F_i, F_j)` (0 if intersecting); `deficit = c − clearance`.
Plate boundary: footprint ⊂ plate outline eroded by `edge_margin`. Beam
envelope: each chief-ray segment buffered by `κ·w` (κ = `beam_envelope_k`) in
plan, height band `z ± κ·w`; intersects a non-participating element if
footprint overlap and height overlap. Moving envelope: union of footprint over
actuator travel samples (translation: two extremes → convex hull sweep;
rotation about a pivot: `moving_envelope_rotation_samples` angles). Tool/cable
clearance: declared polygons must not intersect other footprints.

The footprint lives in the envelope's own plan-parallel frame (`frame_pose`, see NUMERICAL_CONVENTIONS §2).
Pairs are processed in order of bounding-box distance (a lower bound of the exact distance) and the search stops once that
bound exceeds every requirement and the current minimum clearance; results are emitted in sorted pair order.
Applicability: 2.5D only (each element one height slab). Failure modes:
invalid/self-intersecting polygons (`INVALID_FOOTPRINT`, repaired only if
`make_valid` is explicitly requested), zero-area polygons, sub-tolerance
sliver overlaps (tolerance `layout_geometry_tol_mm`).
References: rotated unit square by 45° has half-diagonal `√2/2`; two squares
separated by known gap; SAT/analytic distance between two disks (buffered
points) `= d − r₁ − r₂`; hull of swept translation area `= wL + w²`.

---

## P8. Control / recoverability

### Governing equations

State `θ ∈ ℝⁿ` (actuator DOFs), readings `s(θ) ∈ ℝᵐ` (sensor centroids, power,
…). Sensitivity by central differences with configured steps:
`Jᵢⱼ = [sᵢ(θ+hⱼeⱼ) − sᵢ(θ−hⱼeⱼ)] / (2hⱼ)` (one-sided at travel limits with
reported order). Residual `r = W(s − s*)`. Levenberg–Marquardt with Marquardt
scaling:
`(JᵀJ + λ diag(JᵀJ)) δ = −Jᵀ r`, step clamped to `max_step`, state clamped to
`[θ_min, θ_max]` (saturation recorded per DOF); `λ` adapts (×up on rejected
step, ×down on accepted). Observability: SVD of `WJ`; rank by
`σ > rcond σ_max`; condition number `σ_max/σ_min`; unobservable directions = right null space;
an actuator is reported unobservable when `‖P_null e_i‖² ≥ observability_participation_min` (0.25). Coarse capture: if the sensor has no signal
(lost beam) a deterministic raster/spiral scan of the designated capture DOFs
within travel finds signal, then LM refines. Monte Carlo: PCG64 raw stream →
own uniform/Box–Muller transform (bit-stable), explicit seed required.

### Assumptions / applicability

Local linearization; readings are deterministic engine outputs (centroids of
the chief ray on an observation plane, captured power: what an absorbing plane
detects -- its active area's share, nothing outside its acceptance cone -- or a
transparent plane's active-area share within the cone). Non-differentiable
loss of signal handled explicitly (no gradient through zero).

### Failure modes

Singular/near-singular `J` (reported, LM damping), zero-signal Jacobian,
saturated DOFs, limit cycling, conflicting targets.

### Independent references

* Tilting a flat mirror by `δ` moves a beam by `2 L δ` at distance `L`
  (Jacobian analytic `∂x/∂δ = 2L`).
* Linear translation of a lens perpendicular to the axis shifts the beam by
  `−(x_lens)/f·L_after` (analytic small-signal).
* LM on a linear system converges in one damped step to the least-squares
  solution (`np.linalg.lstsq`).
* Identical seed ⇒ identical Monte Carlo output.

---

## P8a. Extended emitters: sampled incoherent rays

An `EXTENDED_EMITTER` source (`ExtendedEmitterSourceV1`) is a finite planar emitter: a rectangle or disc centred on the local origin
in the local XY plane, uniform over its area, emitting incoherent, unpolarized light. It emits but does not stop, reflect or absorb
light.

**Normalization.** The authoritative quantity of each spectral line is its total radiant power `P` in W into the law's whole domain.

| `angular_law` | Domain and measure | Law | Derived and reported |
|---|---|---|---|
| `LAMBERTIAN` | hemisphere around local +Z, projected solid angle `cos(θ) dΩ`, size `π` sr | `d²P = L cos(θ) dΩ dA` | radiance `L = P / (π A)`, W/(m² sr), `A` in m² |
| `ISOTROPIC` | whole sphere, both sides of the surface, solid angle `dΩ`, size `4π` sr | `d²P = P / (4π A) dΩ dA` | intensity `I = P / (4π)`, W/sr |

The two laws are different sources: from one geometry they collect different fractions.

**Sampling.** Each line is sampled by `sample_count` rays of equal weight `P / sample_count`: the first points of the 5-dimensional
Sobol sequence (Joe-Kuo direction numbers, Gray-code order) with a digital shift derived from `seed`. Two coordinates place the ray
uniformly in area, two give its direction uniformly in the law's measure (`sin²θ = u`, `φ = 2π v` for Lambertian; `cos θ = 1 - 2u`,
`φ = 2π v` for isotropic), and the fifth picks one of two orthogonal linear states, so the ensemble is unpolarized. The sequence uses
integer arithmetic and libm-free trigonometry: the rays are the same bits on every platform.

**Propagation.** Each sample is one chief-only ray with no Gaussian state, through the same interaction handlers as a beam. A clear
aperture takes or passes it whole. An observation plane counts a ray that lands on its active area (the whole plane without one) inside
its acceptance cone. An absorbing plane stops a ray on its active area, and a splitter that traces its reflection splits a ray.

**Composition.** Rays have no phase relation, so their powers add. No interference, coherent sum or fringe exists between them, between
two lines, or between them and a Gaussian beam on the same plane. A Gaussian beam's results keep their own meaning beside them. A
fluorescent or scattering re-emission law is not part of this source: power is emitted once, at the emitter.

**Accounting.** Emitted power = the sum of the ledger's sinks for each line (`PowerBudgetLine.conserved`). Light a ray leaves behind at an
interface whose reflection is not followed is booked `UNTRACED_REFLECTION` or `UNTRACED_TRANSMISSION`; it is not followed per ray, so it
is stated on every plane as light that may reach it (`SAMPLED_RAY_BRANCH`, `power_complete` false, `power_upper_bound_W` on an absorbing
plane; the plane's `value_basis` entry names the surfaces where it was left behind). A plane no ray of a line is counted on reads that line
as a sampled zero: a `sampled_light` entry with `power_W` 0.0 and no counted crossing, and one ray's weight (`emitted_W / sample_count`) in
the plane's `power_sampling_resolution_W`. Its power is `KNOWN`, never `KNOWN_ZERO` (a finite number of rays does not show it dark); a
`power_W` limit within the resolution of the value is not decided, and more rays resolve smaller fractions.

**Limits.** `sample_count` is a power of two from 16 to 65536 per spectral line. A ray meets at most `max_interactions_per_path` physical
surfaces. Rays and their branches share a budget of `max_interactions_per_path` steps per declared sample, reached only by partially
reflecting surfaces that send rays back and forth. When it is spent the ray being traced stops, the samples not yet launched are not
launched, and their power is booked `PATH_LIMIT` with one `MAX_PATHS` issue; `samples_traced` says how many rays stand behind the
results, which stay relative to the emitted power. `max_paths` and `max_trace_steps` bound beams and do not apply to these rays. Tracing
takes about 0.3 ms per ray-surface step in CPython. The first 32 rays of each line are kept as paths for drawing.

**Unavailable for rays.** Beam radius, waist, Rayleigh range, `M²`, Jones state, phase, wavefront, first-order transfer and every
pair metric describe one beam: a target on one of them reads `OUT_OF_MODEL_SCOPE`. `power_W` adds the rays' counted power to the beams'.
The linear optical network holds amplitudes of Gaussian beams and has no extended emitter. Exposures report the power a surface receives
from the rays, without a beam size. A design with an extended emitter is bought as optics only: its Optical Layout Drawing
draws beam paths for Gaussian sources only; a laboratory layout of it is refused (`EXTENDED_EMITTER_NOT_RESOLVABLE`).

**Validated against closed forms** (`tests/test_extended_source_radiometry.py`, 4096 rays, absolute tolerance 0.004 set before the first
run): a small Lambertian disc to a coaxial aperture (`F = sin²θ`), a finite Lambertian disc (coaxial disc-to-disc view factor), a Lambertian
rectangle (aligned identical rectangles), an isotropic emitter (`Ω / 4π`) and, at 16384 rays, `cos⁴θ` irradiance on a parallel plane.

## P8b. Sampled sources: intensity patterns, sampled complex fields, speckle

Three source kinds state their emission on a grid of the source's local XY plane (`SourceGridV1`: `column_count` x `row_count`
points at `pitch_mm`, centred on the local origin; arrays `[row][column]`, row 0 at the most negative y; up to 256 points per axis
written in the Project, up to 1024 for a generated speckle field).

**`SAMPLED_INTENSITY`, an arbitrary intensity pattern** (`SampledIntensitySourceV1`). An incoherent, unpolarized emitter whose radiant
exitance follows `intensity_map.values`: `PIECEWISE_CONSTANT` (each value over its cell; the emitting area is the grid's extent) or
`BILINEAR` (between the grid points, over the rectangle through the outer points). Normalization: `RELATIVE` values give the shape and
`power_w` / the line powers the scale; `W_PER_MM2` values are the exitance and their integral (cell sum, or the trapezoidal rule of the
bilinear surface, exact for it) is the power. Angular laws, the same at every point: `LAMBERTIAN`, `ISOTROPIC` and `COSINE_POWER`
(intensity `cos^n θ`, P8a), `UNIFORM_CONE` (uniform radiance within `cone_half_angle_deg`: `sin²θ = u sin²θ_c`), `COLLIMATED` (every
ray along local +Z: a geometric idealization without diffraction spread). Positions: the extended emitter's Sobol coordinates through
the inverse of the map's cumulative distribution (the row from one coordinate, the column in the row from the other, the place in the
cell from what remains of each: uniform in a piecewise-constant cell, by the inverse marginal and conditional distributions of the
bilinear surface between grid points). Each map is a monotone transform of each coordinate, so the sequence stays stratified, every ray
keeps the weight `P / sample_count`, and the half-sample estimate of P8a applies. Results are those of P8a (`sampled_light`, the binned
irradiance, `sampled-crossings`); `results.sampled_sources` adds `pattern`, `exitance_mean_W_per_mm2` and `exitance_peak_W_per_mm2`.

**`SAMPLED_FIELD`, a sampled complex field** (`SampledFieldSourceV1`): `U = amplitude exp(i phase_rad)` at the grid points, coherent and
monochromatic at `wavelength_nm`, fully polarized (`polarization`), propagating toward local +Z (`exp(-i w t)`, `exp(+i k z)`). `|U|^2`
is W/mm^2 (`SQRT_W_PER_MM2`) or scaled to `power_w` (`RELATIVE`); the power is `sum |U|^2` times the cell area.

**`SPECKLE`, a speckle field** (`SpeckleSourceV1`): an illumination of flat phase (`UNIFORM_DISC`, `UNIFORM_RECTANGLE` or `GAUSSIAN`, its
power on the grid `power_w`) through `realizations` random phase screens of Gaussian statistics and Gaussian correlation (rms
`phase_rms_rad`, autocorrelation `σ² exp(-r²/l_c²)`), realization `m` from `seed + m`. A screen is integer arithmetic (SplitMix64 words,
four uniform deviates per point summed to unit variance) convolved along each axis with `exp(-2 t²/l_c²)` truncated at `3 l_c` and
scaled by `σ` over the kernels' root sum of squares: Gaussian to the central-limit approximation, and the same screen on every platform.
Each realization is one coherent field; different realizations are mutually incoherent (their irradiances add: a moving diffuser over
an exposure).

**How a run traces a field** (geometric optics, both kinds): `ray_sampling.sample_count` rays drawn in proportion to `|U|^2` (constant
over each cell, the same inverse distribution), each leaving along the field's local wavevector: transverse direction `∇φ / (n k0)`
(the eikonal; `∇φ` from the wrapped differences to the neighbouring grid points, central inside, bilinear between grid points; for a
speckle field ray `k` reads realization `k mod realizations`), optical path starting at `φ / k0`, the field's Jones vector in a basis
whose first axis is local +X projected off the ray. The rays add their powers on every plane: the run gives the field's geometric
transport (collected power, sampled irradiance), never its interference. Validity, checked when the scene is built for every
realization: the wrapped phase step between neighbouring lit grid points (`|U|^2` at least `1e-6` of the peak, and the brightest points
holding all but `1e-3` of the power, so at most that share has unchecked steps) is at most `π/2`, else `SOURCE_FIELD_UNDERSAMPLED`; the
transverse sine stays below 1 at every grid point a ray's direction is interpolated from (the points of every cell with nonzero
`|U|^2` and their neighbours: every ray's sine is then below 1), else `SOURCE_FIELD_EVANESCENT`. A field outside either is not
traced: the issue (`OUT_OF_MODEL_SCOPE`, at the source) says why, the source's power is booked `SOURCE_NOT_TRACED`, the planes read
`UNKNOWN` where its light may arrive (never a known zero), the run completes and every other source, plane and target is evaluated;
the field analysis of that source is refused with the same code. A screen's rms step is about `√2 σ p / l_c`, so a strong diffuser needs a correlation length of many pitches.

**The field itself: `os-optics results field`** (`results/field.py`, `FieldRequestV1`, analysis `FIELD_PROPAGATION`). The source's field is
built again from the run's request and propagated as a scalar field to an observation plane:

* `ANGULAR_SPECTRUM`: when the source sees the plane through free space (no element on the source axis before it, one medium) and the
  plane is parallel to the source plane with its axes along the source's: `H = exp(i 2π z sqrt((n/λ0)² - fx² - fy²))`, evanescent
  frequencies removed, band-limited (Matsushima and Shimobaba 2009) on the grid zero-padded to `padding` times its extent. Exact scalar,
  not paraxial. Conditions: the band limit keeps at least 0.99 of the power, and the window holds the light (below).
* `PARAXIAL_TRANSFER`: the generalized Collins integral of the first-order ray-transfer matrix of the source axis to the plane (a probe
  beam along the axis, traced through the system's elements: the run's real-ray finite-difference matrix), in canonical coordinates
  (positions; `n` times direction cosines), separable along the plane's axes:
  `U2(u) = sqrt(1/(iλ0 b)) ∫ U1(x) exp(iπ (a x² - 2 x u + d u²) / (λ0 b)) dx`, discretized as the factorization
  `[[1, 0], [c/a, 1]] [[a, 0], [0, 1/a]] [[1, b/a], [0, 1]]` (transfer function, magnification, lens phase; `λ0 |b/a| <= N p²`) or the
  direct sum at `N` output points of pitch `λ0 |b| / (N p)` (`|a/b| 2 x_max p / λ0 <= 1`), the first that is sampled. The elements'
  power transmission along the axis scales the field. Conditions: no coupling of the axes beyond rounding (a rotation by a multiple of
  90 degrees is a relabelling), symplectic within 1e-3, a sampled discretization, the field's angular spread (the sine holding 99 % of
  its angular spectrum) within 0.2, the window holding the light (below), and the field's rays (traced again; only those that reach the
  plane through the axis's own interactions): at least 0.999 of the launched rays reach it (the apertures pass the field; they are
  checked against the rays, not applied), their power over the axis-transmitted power lies in `[0.999, 1.001]` (the elements transmit the field's rays as they do
  its axis; the probe beam whose transmission scales the field is launched along the axis with the field's rms size about its centroid,
  and where a clear aperture cuts that Gaussian's tail the cut is undone, each aperture's share of the probe's arriving power: a flat-top
  field that fills an aperture a Gaussian of its rms size overfills is not refused for the probe's profile), and first order describes the system over the field
  (below), else `ABERRATED_BEYOND_FIRST_ORDER`. The apertures' diffraction and the elements' action on polarization are not modelled.

* `AUTO`: the angular spectrum where it applies and all its conditions hold, else the paraxial transfer.

**First order over the field** (paraxial transfer, realization 0). Rays spread over the field's phase space as its amplitude (positions
distributed as `|U|`, directions the local wavevector plus a Gaussian diffraction offset of `√2` times the amplitude's rms angular
spread, so a small waist's diffraction cone fills the elements as its light does) are traced through the elements. Each ray's landing
point and direction against the first-order matrix's prediction, carried back to the source plane by the inverse matrix, is its
position and direction error `(dx, da)` there; free space over `s` (least squares over the rays: where the position errors are
smallest) turns them into `(dx + s da, da)`, and at that plane the aberration is a phase with `∇φ = k0 n da`, fitted by a polynomial
of degree 8 at the rays' first-order points. The field with `exp(iφ)` applied at that plane (free space there and back on a grid twice
as wide: the Fresnel transfer function, or the chirp-Fourier form when the plane is far) is propagated by the same transfer, and over
the region the plane's values may change by at most the request's `value_tolerance` (default 1 %; `aberration_value_change`): window and region power, mean and peak
irradiance, rms widths and contrast counted twice (so at most 0.5 % by default: within about a Rayleigh range of an image waist the fit predicts
as little as 0.35-0.5 of their change, below), the centroid relative to the rms width once (its shift is predicted exactly). The phase must be sampled where the light is (at most `π/2` between
neighbouring samples, `aberration_phase_step`), and what it leaves out (the fit's residual gradient, the position errors left at `s`)
bounds the field's relative rms error by at most 0.01 (`aberration_unmodelled`, the Poincaré inequality of a Gaussian weight). With
`include_field` the complex field over the region may also change by at most 0.1 relative rms, up to a constant phase
(`aberration_field_error`): away from a focus the phase can be aberrated while the irradiance is not. That condition is the exported
field's alone: where it fails the field arrays are left out (`arrays.field_withheld`) and the plane's values are stated, the same as
to a request without `include_field`. The change is the aberration's effect at that plane, not a bound from a Strehl ratio: near a focus a
phase aberration moves the peak in first order (a Gaussian of `w` 1.25 mm through a 50 mm plano-convex lens: 12.7 % 0.3 mm before the
paraxial focus, 4.4 % at it, where its Strehl ratio is 0.958).
Calibration against an independent reference (exact real-ray trace of the beam's wavefront normals through a plano-convex lens,
exit-face field by the ray mapping, exact angular spectrum; its first-order twin by the same transport): Gaussians of `w` 0.5-1.25 mm
through a 50 mm lens and 1-2.5 mm through a 100 mm lens, and waists of 10-80 um imaged by the 50 mm lens from 100 mm (their own rays
leave parallel; only their diffraction fills the lens), each at nine planes within three Rayleigh ranges of the image waist
(135 planes), and a second reference (exact 3-D vector-Snell trace of a 2048 x 2048 ray grid launched along the beam's wavefront,
ray-mapped exit field, exact band-limited angular spectrum by direct transform) for 405, 633 and 1064 nm, beams of 0.4-0.8 mm,
decentred and tilted lenses: far from the image waist and at it the change is 0.87 to 1.06 times the first-order model's error of
the peak and width; within about one Rayleigh range of the waist it is 0.35 to 0.7 of it (a focus shifted by the spherical
aberration moves the peak in first order: `w` 0.5 mm, 633 nm, 0.5 Rayleigh ranges off the waist: change 0.34 %, error 0.71 %;
1064 nm, `w` 0.7 mm, 0.6 Rayleigh ranges: 0.78 %, 1.10 %), the error exceeding the change by at most 0.47 % of the value. Counted
twice, every accepted plane of both scans is within 1 % of the references (largest 0.74 %); the decentred and tilted cases' centroid
shifts are the check's change to 1 %. The margin is calibrated over 405-1064 nm; the worst accepted plane's error grows with the wavelength
(0.77 / 0.80 / 0.92 % at 405 / 633 / 1064 nm, 0.97 % at 1310 nm) and passes 1 % at 1550 nm (`w` 0.72 mm, 2.5 mm past the paraxial focus of the
50 mm lens: change 0.9994 %, true peak error 1.03 %). A lens decentred by 4 mm (coma, `w` 0.68 mm): the centroid's shift, 0.08 rms widths at the
focus, is the check's change to 1 %.

**The window holds the light** (both methods, realization 0). Each computation is periodic over its output window (a circular
convolution; the chirp route's periodic sum over `λ0 |b| / p`): light that leaves the window folds back into it with its power
unchanged, so the power in the window shows nothing. The field is propagated again on a window twice as wide on the same output
lattice (twice the padding; the chirp route on the input interpolated to half the pitch, band-limited, zero beyond the grid). Within
the band limit (angular spectrum) or the sampling condition (factored route) light moves at most half its padded window, so a field
that starts on the grid lands inside the wider window. Conditions: at least 0.99 of the wider computation's power lies inside the first
window (`window_power_fraction`, else `WINDOW_LOSES_POWER`), and the plane's values over the region (window and region power, mean and
peak irradiance, rms widths, contrast; the centroid relative to the rms width) change by at most 1 % (`window_refinement_change`, else
`WINDOW_NOT_CONVERGED`). Example: a Gaussian of 0.15 mm waist at 5 Rayleigh ranges, grid 1.28 mm, padding 2: the band limit cuts its
spectrum where the amplitude is still 5.6 %, and the peak comes out 3.8 % high; the wider window changes it by that much, so the angular
spectrum is outside its conditions there and `AUTO` takes the paraxial transfer (peak within 2e-4 of the Gaussian beam).

A plane outside the chosen model's conditions gets no values: every plane quantity is `OUT_OF_MODEL_SCOPE` with the failed condition,
and the models list each condition with its observed value. Reported (over `region`, default the output window, which is the
method's: the padded grid for the angular spectrum, `λ0 |b| / p` or the magnified padded grid for the paraxial transfer, so the mean
irradiance and the contrast over the default region change when `AUTO` falls back from one to the other: state a region to compare
them; mean over the realizations): power in the window and the region, mean and peak irradiance, centroid, rms width (a Gaussian beam's 1/e² radius is
twice it), the contrast `std(I)/mean(I)` of one pattern (`speckle contrast` for a speckle source; one standard error as a `SAMPLING`
interval), the contrast of the average of `M` patterns and `mean(C_m)/√M` (what `M` independent patterns give), the power of the
field's rays on the plane (the geometric transport), the averaged irradiance map (block means, at most `map_points` per axis) and,
with `include_field`, the complex field of realization 0 over the region (`--field-out` writes it as a `pupil-field` `FIELD`
illumination). A speckle source also gives its screens' rms phase over the illumination and the share of the power they leave
unscattered (`|<exp(iφ)>|²`, `exp(-σ²)` for a Gaussian screen over many correlation cells) and states the pattern fully developed when
`exp(-σ²) <= 0.01` (`σ >= 2.15` rad): a weaker screen's pattern holds the unscattered beam and its diffraction by the illumination's
edge, whose ripple the contrast also counts (a 1 mm disc at 200 mm with `σ = 0.3` rad has contrast 0.81, nearly all of it that
ripple). The rms widths need 8 grid points of the region along their axis and the contrasts 8 along both (else `NOT_APPLICABLE`,
`TOO_FEW_POINTS`). Convergence: half-sample over realizations; a single pattern's values carry the window refinement (the window twice
as wide, over the same region).

**Validated** (`tests/test_sampled_sources.py`; tolerances set before the first run): a uniform map gives the uniform rectangle's
points (1e-14 mm); a two-level map sends 3/4 of the rays to its bright half (2/4096); a bilinear ramp's mean and cumulative
distribution (2e-3); a uniform cone's projected-solid-angle shares (2e-3); a Gaussian map's binned irradiance against the exact cell
integrals and the Engine's Gaussian beam source at the same plane (3 % on every cell above a tenth of the peak, 65536 rays); a linear
phase `2πx/Λ` turns every ray by the grating equation `sin θ = λ/Λ` (1e-12) and moves the angular-spectrum centroid to `z tan θ`, the
paraxial transfer's to `z sin θ` (1e-5); a Gaussian field through a 100 mm lens near its focus has the Gaussian beam's 1/e² radius and
peak irradiance (the Engine's q propagation through the same lens, 1 %); the field at a plane as a pupil-field illumination focuses to
`2P/(π w_f²)` (1 %); the aberration check against an exact ray-traced diffraction reference (`_pcx_reference`: refused at
12.5 % and 4.4 % with the change equal to the error to 15 %, a 20 um waist's 11 % likewise; a 0.5 mm beam accepted within 1 % at
the focus and 1.5 Rayleigh ranges either side; a 4 mm beam's rms width 20 and 60 mm behind a 100 mm lens within 1e-3 of the
real rays' second moments, its field refused); an off-centre field through a window keeps `(1 - R)²` of its power (1e-6); a screen's rms phase (5 %) and its autocorrelation at one correlation length (`e^-1`, 0.05); fully developed speckle
(rms phase 3 rad, 1 m of free space, 16 realizations) has one pattern's contrast 1 (0.1) and the average's `1/√16` (0.05).

## P9. Quantities a run does not evaluate

A run evaluates the targets on the metrics of the result layer (P10a). A target whose metric is in the table below is not
evaluated by the run: it is graded `OUT_OF_MODEL_SCOPE` (assessment `INDETERMINATE`) with the table's text as its
message, which names the command that computes the quantity where one does; a metric name the Engine does not know is
graded `UNKNOWN` (`UNSUPPORTED_METRIC`). The table is generated from `src/os_optics/validation/scope.py`
(`OUT_OF_SCOPE_METRICS`, `ANALYSIS_METRIC_ROUTES`); a metric is matched without regard to case
(`SPECTRAL_RESOLUTION_NM` is `spectral_resolution_nm`).

<!-- BEGIN SCOPE METRICS (generated by docs/engine/generate_docs.py; do not edit) -->

| Metric | Not modelled, or what computes it |
|---|---|
| `fresnel_propagation` | Fresnel (near-field diffraction) propagation is not a target of a run; 'os-optics results field' propagates a sampled-field or speckle source to a plane (angular spectrum, or the paraxial transfer of the source axis) |
| `fraunhofer_propagation` | Fraunhofer (far-field diffraction) propagation is not a target of a run; 'os-optics results pupil-field' computes the scalar low-NA focal field of a pupil |
| `angular_spectrum_propagation` | angular-spectrum propagation is not a target of a run; 'os-optics results field' propagates a sampled-field or speckle source to a parallel plane through free space by its angular spectrum |
| `complex_amplitude_field` | full-field complex-amplitude grids are not a target of a run; 'os-optics results field' gives the field of a sampled-field or speckle source at a plane, 'os-optics results pupil-field' the scalar low-NA focal field of a pupil |
| `field_amplitude_map` | full-field complex-amplitude grids are not a target of a run; 'os-optics results pupil-field' computes the scalar low-NA focal field of a pupil |
| `fdtd` | FDTD is not modelled by this Engine |
| `rcwa` | RCWA is not modelled by this Engine |
| `high_na_vector_diffraction` | a high-NA vector focal field is not a target of a run; 'os-optics results focal-field' computes the vector focal field of an ideal aplanatic objective |
| `doe_response` | arbitrary diffractive optical element response is not a target of a run; 'os-optics results pupil-field' computes the scalar low-NA focal field of a thin pupil mask |
| `metasurface_response` | metasurface full-wave response is not modelled by this Engine |
| `speckle_contrast` | speckle contrast is not a target of a run; 'os-optics results field' gives the speckle contrast of a SPECKLE source at a plane, of one pattern and of the average of its realizations |
| `phase_retrieval` | phase retrieval is not modelled by this Engine |
| `stray_light` | general stray-light Monte Carlo (scattering, every non-sequential path) is not modelled by this Engine; 'os-optics results ghosts' gives a run's interface-reflection ghost images with a bound on the ghosts it does not list |
| `rough_surface_scattering` | rough-surface scattering is not a target metric of a run; a surface's diffuse scatter (scatter on a refracting, reflecting or absorbing surface, LAMBERTIAN or COSINE_POWER) is traced as rays and adds to each plane's power and sampled_light |
| `pinhole_far_field` | exact far-field propagation behind a pinhole is not modelled by this Engine |
| `far_field_pattern` | diffraction far-field patterns are not a target of a run; 'os-optics results pupil-field' computes the scalar low-NA focal field of a pupil |
| `psf` | point spread functions are not a target of a run; 'os-optics results imaging' computes them for a run's image plane, 'os-optics results pupil-field' the scalar low-NA focal field of a pupil |
| `mtf` | MTF is not a target of a run; 'os-optics results imaging' computes the diffraction and geometric MTF of a run's image plane |
| `partial_polarization` | depolarization (an element whose Mueller matrix is not a Jones matrix) is not modelled by this Engine; a partially polarized source is (source polarization PARTIAL or UNPOLARIZED, metric degree_of_polarization) |
| `birefringent_double_refraction` | double refraction is traced in uniaxial crystals with plane faces (material kind uniaxial): each beam leaving the crystal is an arrival of its own, read with the usual metrics; this name is not a target metric. Biaxial crystals, optical activity and curved crystal faces are outside the model |
| `spectral_resolution_nm` | a spectrometer's instrument function (slit, dispersion onto the detector, pixel sampling) is not modelled; 'os-optics results grating-orders' gives the direction of every diffraction order of a grating, and a sample's emission lines are discrete lines of stated wavelengths |
| `cavity_circulating_power_W` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `circulating_power_W` |
| `cavity_transmitted_power_W` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `transmitted_power_W` |
| `cavity_reflected_power_W` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `reflected_power_W` |
| `cavity_transmission` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `transmission` |
| `cavity_reflection` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `reflection` |
| `cavity_power_build_up` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `power_build_up` |
| `cavity_resonant_power_build_up` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `resonant_power_build_up` |
| `cavity_free_spectral_range_Hz` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `free_spectral_range_Hz` |
| `cavity_transverse_mode_spacing_Hz` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `transverse_mode_spacing_Hz` |
| `cavity_linewidth_Hz` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `linewidth_fwhm_Hz` |
| `cavity_finesse` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `finesse` |
| `cavity_mode_matching` | the cavity analysis (`os-optics project cavity`, `os-optics results cavity`), quantity `mode_matching_efficiency` |

<!-- END SCOPE METRICS -->

Calculations of their own, outside a run: the scalar low-NA focal field of a pupil (P10e), the vector focal field of an
ideal aplanatic objective (P10g), the interface-reflection ghost paths of an image plane up to a stated bounce order with
a bound on the rest (P10i, ghost paths), and the scalar propagation of a sampled-field or speckle source to a plane
(angular spectrum in free space, the Collins integral of the source axis's first-order transfer through elements) with its
irradiance and speckle contrast (P8b). Traced in a run: the diffuse scatter of a surface as rays (P10i, diffuse surface
scatter) and partially polarized sources as two incoherent Jones components (P5). Not modelled: FDTD, RCWA, metasurface
full-wave response, phase retrieval, the exact far field behind a pinhole, biaxial crystals, optical activity,
depolarizing (non-Jones) elements and general stray-light Monte Carlo.

### P9b. Weak-signal measurement chain and nonlinear conversion: the boundary

Each item is in model scope and supported, or out of model scope with an external route (`validation/scope.py`,
`MODEL_SCOPE_BOUNDARY`; every item with its disposition and the refused model kinds are `capabilities()["model_scope_boundary"]`).

| Item | Disposition | What it is, or what a calculation needs that the Engine does not model | Route |
|---|---|---|---|
| Extended emitter | supported | | `source_model` `EXTENDED_EMITTER` (ISOTROPIC / LAMBERTIAN seeded rays), `os-optics results sampled-light` (P8a) |
| Weak bulk absorption | supported | | a declared material's `absorption_coefficient_per_mm` (Beer-Lambert with the Fresnel losses at its faces) |
| Camera response | supported | | `os-optics results detector` (P10e): photons, electrons, ADU, saturation, noise and SNR per pixel; with `electrical`, photocurrent, noise current and SNR per pixel |
| Static acousto-optic deflection | supported | | `deflector.acousto_optic` (P10e): the tones as orders of one grating, each tone's frequency shift; order powers from a stated table |
| Fluorescence emission | supported | a thin sample excited by the run's own light: absorbed share, quantum yield, emission lines (P9c) | `source_model` `FLUORESCENCE`; `os-optics results sample-emission` |
| Raman emission | supported | spontaneous Stokes scattering of a thin sample: shifts, cross-sections, number density, path length (P9c) | `source_model` `RAMAN`; `os-optics results sample-emission` |
| Chopper modulation | supported | square-wave chopping or sinusoidal modulation of named sources' light, read at its fundamental (P9d) | an observation plane's `detection.lock_in.modulation`; `os-optics results detection` |
| Lock-in detection | supported | demodulation at the fundamental, RC output filter of 1 to 4 sections, noise-equivalent bandwidth, SNR (P9d) | `detection.lock_in`; target metric `lock_in_snr` |
| Detector bandwidth and noise | supported | bandwidth, noise-equivalent power or dark current and load resistance, shot noise (P9d); not lag, crosstalk, nonlinearity, 1/f or ADC quantization noise | `detection.detector.electrical`; target metrics `detector_bandwidth_Hz`, `detector_snr`; per pixel `os-optics results detector` |
| Feedback and modulator timing | out of scope, external | control loops, acousto-optic switching times, RF phases and crosstalk | the static deflector gives directions and frequency shifts; timing is external |
| Spectrometer instrument function | out of scope, external | slit image, dispersion onto the detector, pixel sampling | `os-optics results grating-orders`; `os-optics results detector` for a stated irradiance |
| Second-harmonic generation | supported | | declared model `NONLINEAR_CRYSTAL` (P10l): undepleted-pump conversion of a TEM00 beam, the harmonic traced as a new spectral line; `os-optics results nonlinear`, targets `shg_conversion_efficiency`, `shg_power_W`. Pump depletion, sum-frequency mixing of several lines and other nonlinear processes: `os-optics interop show nonlinear-ultrafast-communications` |

How a request on an item outside the model is answered:

* Target metric `spectral_resolution_nm` is accepted and reported `OUT_OF_MODEL_SCOPE` with the reason and route
  above; the target is `INDETERMINATE`, never met. `detector_bandwidth_Hz`, `detector_snr` and `lock_in_snr` are
  computed on a plane that states `detection` (P9d); on any other plane a target on them is not assessed (`UNKNOWN`,
  `TARGET_INVALID`), naming the `detection` it needs. The second-harmonic metrics are computed (P10l).
* A declared optical model `model_kind` `CHOPPER` or `LOCK_IN` is refused before the Project is read:
  `SIMULATION_MODEL_UNSUPPORTED` at the optical model, with `details.boundary_item`, `details.disposition`,
  `details.missing_model` and `details.route` (`local/boundary.py`): the route declares them, since a modulation and a
  lock-in are part of an observation plane's `detection`, not optical elements. Any other unknown kind keeps the
  Project schema's refusal.
* A source of two spectral lines (a fundamental and its harmonic with powers you state) traces dichroics and filters at
  both wavelengths. Its harmonic is a stated input: the run computes no conversion for it. A harmonic the run generates
  comes from a `NONLINEAR_CRYSTAL` (P10l).

Nothing outside the supported items is estimated: no spectrometer or timing quantity is reported.

### P9c. Sample emission: fluorescence and Raman scattering of a thin sample

A `FLUORESCENCE` or `RAMAN` source is a sample whose emission follows from the light the run delivers to it
(`local/sample_emission.py`). Its `excitation` names an observation plane at the sample (across the excitation beam,
usually just before it) and the excitation's vacuum wavelength `lambda_ex`.

1. The Project without the samples' emission and without targets is traced (the excitation trace, identified by
   `excitation_input_hash`). The power `P_ex` of `lambda_ex` arriving at the plane is read from `power_by_wavelength` (every
   source that emits `lambda_ex` adds; the other wavelengths do not excite the sample). It must be a known quantity: complete,
   with no beam of it out of model scope, bounded only or partly beside the active area. Otherwise the Project is refused
   with `SAMPLE_EXCITATION_NOT_KNOWN` (`details`: `observation_id`, `wavelength_nm`, `quantity_state`): the emission is never
   guessed. A source that emits no `lambda_ex` is refused the same way; light that does not reach the plane is a known zero,
   and the sample then emits nothing.
2. Photon numbers: the excitation photon rate is `N_ex = P_ex lambda_ex / (h c)` (exact SI `h`, `c`).
   * Fluorescence: the absorbed share is `a = 1 - 10^-A` from the decadic `absorbance` `A`, or `absorbed_fraction` as stated.
     The model holds for a thin, optically dilute sample: `A` at most 0.1, `a` at most `1 - 10^-0.1` (0.2057), the usual
     limit of quantitative fluorescence (absorbance below 0.1, IUPAC Technical Report, Brouwer, Pure Appl. Chem. 83 (2011)
     2213). Across such a sample the excitation falls by at most 21 %, so the whole sample emits; a denser sample emits from a
     thin layer at its entrance and reabsorbs its own emission (inner-filter effects), which the model does not follow: it is
     computed by the same formula, and the record's `validity` holds the check (`absorbance` or `absorbed_fraction`,
     `maximum`, `observed`, `within: false`). The sample emits `quantum_yield` photons per absorbed photon (above 0: a sample that does not
     fluoresce is left out of the Project, as an absorbance of 0 is refused), line `i` the share `photon_fraction_i` (fractions
     within 1e-9 of 1 in sum are used as written, others divided by their sum; every line longer than `lambda_ex`). Line `i` carries `N_i = N_ex a QY f_i` photons per second and
     `P_i = P_ex a QY f_i lambda_ex / lambda_i`: each photon has the energy of its own wavelength, the Stokes shift's energy
     stays in the sample.
   * Raman: band `j` of Stokes shift `s_j` (cm^-1) lies at `lambda_j = 1 / (1 / lambda_ex - s_j 1e-7)` nm. With `N` scatterers
     per cm^3 along `L` cm and the stated differential cross-section taken the same in every direction
     (`sigma_j = 4 pi (d sigma / d Omega)_j`), it scatters `N_ex N sigma_j L` photons per second, `P_j = P_ex N sigma_j L lambda_ex / lambda_j`.
     The cross-section is photon-counting: scattered photons per second into a unit solid angle over incident photons per
     second per cm^2 (an energy-weighted cross-section differs by `lambda_ex / lambda_j` and is not what the field states).
     The model holds while the sum of `N sigma_j L` stays at most 0.01 (single scattering of an undepleted excitation); above
     it the same linear form is computed (it overstates the scattered power by about half the scattered share) and the
     record's `validity` holds the check (`scattered_fraction`, `maximum` 0.01, `observed`, `within: false`); a sum of 1 or
     more is refused.
3. The emission is traced as an incoherent, unpolarized extended emitter (P8a) on `emitting_surface`, `ISOTROPIC` (Raman, and
   fluorescence of randomly oriented emitters) or `LAMBERTIAN` (a scattering fluorescent surface), with exactly those line
   powers and `sampling`. The derivation is the sample component's `sample_emission` in the Engine request; `os-optics
   results sample-emission` (`RunResults.sample_emission()`, json_api `run_results` kind `sample_emission`) prints it with
   the emitter's collection at every plane.
4. The run with the emission must deliver the same excitation to the plane (relative 1e-12); a trace budget that cuts the
   excitation short, or anything else that changes it, refuses the run with `SAMPLE_EXCITATION_NOT_KNOWN`.

Conditions stated with every record (`model`): linear excitation (no saturation, no photobleaching), no reabsorption of the
emission, emission following the excitation instantly (a lifetime short against any modulation period), no polarization
memory, no depolarization ratio or polarization dependence of the Raman cross-section, anti-Stokes lines not modelled; the
sample inside the thin-sample limits (above; `validity` states an exceeded one). The
excitation beam is not attenuated by the sample: past a transparent plane it goes on with its full power (make the plane
absorbing, or model the sample's body with an absorption coefficient, where the absorbed share matters downstream). A sample's
emission does not excite another sample.

Validated (`tests/test_weak_signal_chain.py`): the line powers and photon rates against the photon-number arithmetic above
(relative 1e-12); the Raman wavelengths and powers against the wavenumber shift and `N 4 pi (d sigma / d Omega) L`; the
collected power through an ideal lens of clear radius `a` with the sample at its focus against the isotropic solid-angle
fraction `(1 - f / sqrt(f^2 + a^2)) / 2`, within four times the sampling's reported sigma; a second laser of another wavelength
does not excite the sample; a laser that misses the plane leaves the sample dark; an absorbance of 6 (or a stated absorbed
share of 0.5) is refused, 0.1 accepted.

Sampling: the emission is sampled like any extended emitter (P8a): `sampling.seed` digitally shifts one stratified Sobol point
set, so two seeds give estimates that differ by the stratified sampling error -- on the chain of
`tests/test_weak_signal_chain.py`, within 1.1/N of the analytic collection at N = 1024 and 4.1/N at N = 16384, against a
reported `collection_fraction_sigma` of 3.3/N and 12.6/N -- and coincide where the result depends on one or two sampled
coordinates. That sigma (the spread of independent random rays) is an upper guide.

### P9d. Detection: photocurrent, noise, bandwidth and lock-in

An observation plane that states `detection` (`ObservationV1.detection`) is a detector's sensitive face: it reads every watt
the run counts on the plane, each source at each wavelength (`ObservationResult.detection.contributions`), and reports every
quantity with its status: `MODELED`, `EXTERNALLY_CHARACTERIZED` (a stated parameter) or `UNAVAILABLE` with `reason` and
`missing` (`validation/detection.py`; `os-optics results detection`, json_api `run_results` kind `detection`).

* Responsivity `R(lambda) = QE(lambda) q lambda / (h c)` A/W, from the stated quantum-efficiency table (linear between rows,
  unavailable outside them: `WAVELENGTH_OUTSIDE_QE_TABLE`). Photocurrent `I = sum R(lambda_k) P_k`.
* Bandwidth (`electrical.bandwidth_Hz`): `FIRST_ORDER_3DB`, the -3 dB frequency `f_c` of a single-pole response, whose
  noise-equivalent bandwidth is `B = pi f_c / 2`; or `NOISE_EQUIVALENT`, `B` itself.
* Noise (one-sided densities, A^2/Hz, white): the light's shot noise `2 q I`; the dark floor, stated once -- a noise-equivalent
  power `NEP` at `nep_wavelength_nm` (current density `NEP R(nep_wavelength_nm)`; it holds every noise but the signal's shot
  noise), or a dark current `I_d` (`2 q I_d`) with, optionally, the thermal noise `4 k T / R_L` of a load or feedback
  resistance. `detector_snr = I / sqrt((2 q I + floor^2) B)`, an amplitude ratio, of the light as traced: a chopped source
  counts in its open state (the photocurrent and `detector_snr` are those while the chopper is open; the time-averaged
  current is `D` times the chopped part), a SINE-modulated source at its mean. `noise_equivalent_power_W_per_rtHz` refers the
  floor to the light's responsivity `I / P`.
* Lock-in (`detection.lock_in`): the light of `modulation.source_ids` is modulated, and a sample's emission carries the share of
  its excitation those sources deliver (`modulated_fraction`). `SQUARE` chopping of duty `D`: the traced power is the open-state
  power, the fundamental's peak amplitude `(2 / pi) sin(pi D)` of it (2/pi at D = 1/2); `SINE` of depth `m`: the traced power
  is the mean, the fundamental `m` of it. The output `X` is the fundamental's rms value with the reference in phase. Its noise
  is the input density through the output filter's noise-equivalent bandwidth: `n` identical RC sections of time constant
  `tau` give `ENBW = Gamma(n - 1/2) / (4 sqrt(pi) Gamma(n) tau)` = `1/(4 tau)`, `1/(8 tau)`, `3/(32 tau)`, `5/(64 tau)`. The
  shot noise of the modulated current is weighted by the reference over a period: `2 q (I_steady + w I_mod)`, `w = D + sin(2 pi D)
  / (2 pi)` for SQUARE (= D at D = 1/2), `w = 1` for SINE; steady light adds only noise. `lock_in_snr = X / sqrt(S^2 ENBW)`.
* The detector's response at the modulation frequency `f`: a `FIRST_ORDER_3DB` response passes the fundamental with
  `|H(f)| = 1 / sqrt(1 + (f / f_c)^2)` (`detector_response_at_modulation`). `lock_in_signal_A` and `lock_in_noise_A` are then
  those at the detector's output in photocurrent units, `X |H(f)|` and `sigma_X |H(f)|`: every modelled noise (shot, dark,
  the load's thermal noise) enters with the photocurrent and passes the same response, so `lock_in_snr` is that of the input.
  A `NOISE_EQUIVALENT` bandwidth states no shape: the signal and noise are those at the input.
* Conditions (`validity`, each with its value, limit and meaning):
  * `modulation_frequency_within_detector_bandwidth`: `f <= f_c` (`FIRST_ORDER_3DB`) or `f <= 2 B / pi` (`NOISE_EQUIVALENT`:
    the -3 dB frequency of a single pole of noise bandwidth `B`; a Butterworth response of noise bandwidth `B` loses less
    there). Within it a single-pole, real-pole or Butterworth response loses at most 3 dB (a peaked response may lose more;
    the shape is not stated) and the input-referred white noises set the lock-in's noise. Above it
    the attenuated signal meets noise added after the response (amplifier voltage noise, the lock-in's own input noise),
    which the model does not hold: `lock_in_noise_A` and `lock_in_snr` are `UNAVAILABLE`
    (`MODULATION_ABOVE_DETECTOR_BANDWIDTH`), and so is `lock_in_signal_A` where the response's shape is not stated; a target
    on `lock_in_snr` is `UNKNOWN`. A `SQUARE` chop above it never lets the detector reach the open state the photocurrent is
    stated for (its output follows the mean, `I_steady + D I_mod`, with a residual ripple): `shot_noise_current_A`,
    `total_noise_current_A` and `detector_snr` are `UNAVAILABLE` too (a target on `detector_snr` is `UNKNOWN`), while
    `photocurrent_A` (the open state's) and `noise_equivalent_power_W_per_rtHz` stand;
  * `modulation_periods_per_time_constant`: the output filter averages over at least 10 modulation periods per time
    constant, or the lock-in's noise and SNR are `UNAVAILABLE` (`LOCK_IN_AVERAGING_TOO_SHORT`);
  * `peak_photocurrent_within_linear_range` (where the detector states `electrical.saturation_current_A`): the peak
    photocurrent -- the open state of a chopped source, `I + m I_m` for a SINE of depth `m` -- is at most the stated linear
    range; above it every quantity that reads the photocurrent is `UNAVAILABLE` (`PHOTOCURRENT_ABOVE_LINEAR_RANGE`). Without
    a stated saturation current the photocurrent is taken as linear at any power, and its `basis` says so.

  The light-dependent quantities are computed from the traced light the plane counts: on a plane whose power is `KNOWN` or
  `KNOWN_ZERO` they are the model's values; on a lit plane whose power is `BOUNDED` or `UNKNOWN` (light the trace does not
  count may add, part of a beam is taken before the plane, an annular beam is counted whole) each carries
  `value_basis["value"]`: the plane's entry merged with the detector's own. The power, the photocurrent, the noise currents,
  `detector_snr` and the lock-in's modulated current, signal and noise rise with the light and keep the kind of the plane's
  power (`LOWER_BOUND` where light may add); `lock_in_snr` is an `ESTIMATE` (added light raises it when modulated, lowers it
  when steady), and so is `noise_equivalent_power_W_per_rtHz` when the run launches more than one wavelength. With a stated
  `saturation_current_A`, the peak photocurrent with the light that may add (at a quantum efficiency of 1 at the longest
  launched wavelength) must stay within it; where it does not, or nothing bounds that light, the quantities that read the
  photocurrent are `ESTIMATE` with the check `peak_photocurrent_A`. A target on `detector_snr` or `lock_in_snr` is graded on
  the interval the quantity lies in where that light is bounded. `detector_snr`: its lower end is the traced value on a
  plane of the model's value or one whose power is a lower end (light may add); on a plane whose power is an upper end (a share taken before it, an
  annular beam counted whole) and whose fate is complete, it is the SNR of the least power the plane's entry allows -- the
  taken shares and the annular misses removed, at the largest responsivity -- and 0 where that entry omits or assumes
  anything else; there the interval up to the traced value holds and is graded with or without light left out as
  negligible, which only widens its upper end. Its upper end is the SNR with that light's largest photocurrent (a quantum efficiency of 1 at the longest
  launched wavelength); there is none where nothing bounds that light or where a check it may leave open applies
  (`peak_photocurrent_A`, a `SQUARE` chop above the band), those checks carried in the target's `validity`.
  `lock_in_snr`: from that photocurrent taken as steady to it taken as modulated, on a plane of the model's value or one
  whose power is a lower end; on a plane whose power is an upper end it is graded with its entry, which carries light left
  out as negligible as the stated truncation (`BELOW_NEGLIGIBLE_LEVEL`, how far that light may move the value either way)
  beside the plane's cause, with `FAINT_LIGHT_BOUNDED` among its codes.
  `PLANE_POWER_NOT_KNOWN` stays where no traced light of known power is on the plane while light may arrive, where the
  power is out of model scope and where the plane bounds did not run. Not modelled: 1/f noise, internal gain and
  excess noise (APD, PMT), amplifier voltage noise, ADC quantization, the shape of the photocurrent's saturation and
  nonlinearity, the settling of the output filter, harmonic or square-wave references, heterodyne (beat-note) detection and
  balanced detection.
* Sampled light: where the plane's light holds a sample's sampled rays, every quantity that reads them carries
  `sampling_relative_sigma`, `sum_k |d ln Q / d P_k| sigma_k` with `sigma_k = collection_fraction_sigma x emitted_W` of each
  sampled line (to first order; the lines' errors added linearly, the lines of one emitter being the same rays). Like that
  sigma it is an upper guide; targets are graded on the value, as every sampled power is.
* Targets: `detector_bandwidth_Hz` (the stated bandwidth), `detector_snr` and `lock_in_snr` read the target plane's detection
  (all its light: a target on them selects no source or wavelength). A quantity that is not available makes the target
  `UNKNOWN` with `DETECTION_QUANTITY_UNAVAILABLE` and the reason; a plane without `detection`, or `lock_in_snr` without a
  lock-in, is not assessed (`UNKNOWN`, `TARGET_INVALID`).

Validated (`tests/test_weak_signal_chain.py`): shot-noise-limited `SNR = sqrt(I / (2 q B))` (relative 1e-12); the first-order
noise bandwidth and each filter order's ENBW against numerical quadrature of `|H(f)|^2` (1e-9); the square wave's fundamental
and noise weight against a sampled wave's discrete Fourier transform; a time-domain lock-in simulation (chopped current of duty
1/4 with cyclostationary white shot noise and a floor, in-phase reference, one RC section): its output mean and standard
deviation match `X` and `sigma_X` within their statistical error, which also rejects the weight of the mean current alone;
the connected chain -- excitation laser, fluorescence, collection lens, detector, chopper and lock-in -- against the closed
forms from the traced line powers (1e-12) and from the analytic collection (within the sampling sigma), its signal and noise
through `|H(f)|` of an analog single pole (scipy) and its `sampling_relative_sigma` against finite differences of the closed
forms; 1 MHz chopping on a 1 kHz first-order detector: the signal `|H| = 1e-3` of the input's, noise and SNR `UNAVAILABLE`
and the SNR target `UNKNOWN`; the `2 B / pi` limit of a noise-equivalent bandwidth against the quadrature of single-pole and
Butterworth (order 2 to 4) responses; 1 W on a detector of 10 mA linear range, and a SINE crest above the range, refused.

---

## P10a. Result layer: what the exported quantities mean

The result layer (`os_optics.results`, `docs/engine/RESULTS_AND_EXPORT.md`) adds no tracer physics; it states and derives.

* **Frames.** Design frame: right-handed, mm. Observation plane: origin, unit `u`, `v`, `n = u x v`; plane coordinates
  `u_mm = (p - origin) . u`; direction cosines `l_u, l_v, l_n`. Gaussian `x`/`y` = the beam's own axes `e1`/`e2`.
* **Signs.** `waist_distance = Re(q) > 0`: waist upstream. Two-beam `optical_path_difference_mm = OPL(a) - OPL(b)`.
  Stokes `s3 = -1`: right circular (clockwise facing the source). Wavefront OPD: ray minus chief (positive = lags).
* **Power ledger.** `input = sum of sinks + residual`; monitors observe (`OBSERVED`) without consuming unless absorbing;
  `UNTRACED_*`, `TERMINATED_OUT_OF_MODEL` and `UNMODELLED_RESPONSE` (a tabulated thin lens's `1 − T`) are power the
  model does not follow, not lost by physics.
* **First-order matrix.** `d(u, v, l_u, l_v)/d(x0, y0, a0, b0)` by central differences of eight differential rays traced
  with the chief ray (steps `first_order_position_step_mm`, `first_order_angle_step_rad`).
* **Polarization.** Jones (a beam is fully polarized, degree of polarization 1); Stokes and ellipse derived from it. A
  partially polarized source's light on a route is the Stokes sum of its two components (`polarization_mixtures`), with
  its degree of polarization; the single-beam polarization metrics read that sum.
* **Time-domain delays.** `group_delay_ps` = group optical path `Σ n_g L` / c from the source's origin plane (the
  envelope's arrival; dispersive, not the phase delay; pulse shape and dispersion broadening not modelled);
  `phase_delay_ps` = `Σ n L` / c; `time_delay_ps` = `(G_a − G_b)` / c of a pair (positive: a later). A pass without a
  stated group delay is left out: the value is the delay of the modelled path, an `ESTIMATE` that is not graded
  (`GROUP_DELAY_UNDEFINED`, `value_basis` `UNSTATED_GROUP_DELAY_ZERO` at the passes); coatings' group-delay bounds give
  an interval; a medium without a group index leaves no value;
  two sources have no common time origin. References: a retro mirror moved by `Δx` moves the pair delay by `2Δx/c`,
  an N-BK7 plate adds `(n_g − 1) t / c` (Sellmeier by central differences), the network export's `group_delay_ps`
  (`tests/test_time_delay_targets.py`).
* **Geometric wavefront.** Pupil rays start on the source plane along the normals of the source's Gaussian phase front;
  the phase-front term `W0 = n (sigma . rho) / (sqrt(1 + |sigma|^2) + 1)` makes a round Gaussian launch an exact
  spherical wave about its centre of curvature. OPD to a plane / sphere through the chief hit point, piston removed.
  Undefined after an ideal or tabulated thin lens (`RAY_OPL_UNDEFINED`). References: flat window, diverging launch (closed form),
  plano-convex singlet against an independent meridional trace and the Seidel `W040` (`tests/test_results_layer.py`).
* **Irradiance map.** Paraxial Gaussian envelopes with plane-wave phases, whole plane, coherent within a coherence group.
* **Exposures and power ratings.** At a glance:
  * **What the bounds count.** Every beam the trace follows; beams of one source and line add their fields with open
    phases, so a surface's exposure is a range. With power ratings also the light the trace does not follow (ghost
    reflections, a coating's transmission, light passing an aperture or an element), followed to every surface, and the
    light left out as negligible or of unknown fate whose power is known: at the diffraction limit of its in-phase power,
    times its open paths' crossings and its returns between surfaces facing each other.
  * **Closed returns.** Left-out light that may circulate through three or more surfaces of separate elements (a ring, a
    folded or coupled cavity, a periscope's windows through its folds) leaves every surface it may reach without an
    upper bound: its rating comparison and `power_rating_utilization` target stay undecided, one
    `EXPOSURE_BOUND_WITHHELD` notice of the run names the return's surfaces (`closed_return_surfaces`) and the surfaces
    without a bound (`surface_ids`), and the traced hits and incident power are still reported. A stack of parallel planes (windows, plates, flat mirrors on one axis) keeps a bound that counts all
    its returns where a series of them converges, and is treated like a ring where none does.
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
    passes only with a large rating margin. An exceedance in W/cm² or J/cm² is decided at a much smaller margin, since
    that light, of small power, cannot undo a traced beam over the area it covers; in W/cm the margin is larger, since a
    thin sheet of that light meeting a face at a shallow angle keeps its P/d along a strip of the face. Coatings of low
    reflectance on the faces (declared or from the catalog) and fewer surfaces that may reflect the light make the
    bound smaller; where that light alone keeps a rating undecided, the run follows it deeper by itself.

  In detail: at every physical surface a traced beam meets: incident power `P`, both Gaussian 1/e²
  radii `w_x`, `w_y`, chief-ray AOI, peak irradiance `I0 = 2P/(π w_x w_y)` (transverse plane; `I0 cos AOI` on the surface),
  `P/d` (`d = 2 min(w)`), pulsed peak fluence `2E/(π w_x w_y)` with `E = P/f_rep`. "Peak" is spatial: for a pulsed
  source `I0` is the spatial peak of the average irradiance. The pulse train enters only the fluence, the pulse energy,
  the pulse-count condition and the coherence revivals below: no dispersive broadening of a pulse and no group-velocity
  walk-off between its lines, coatings at the line's
  wavelength, no group-delay dispersion of a mirror. The exposure is the intensity of the summed fields arriving
  from either side. A peak fluence is the energy of one pulse per area. A pulse holds every line of its source that
  crosses the surface along one path (a pulse keeps its duration: its lines travel together); pulses of different
  sources, of one source along different paths, or of members of one class need not arrive together. Its lower bound is
  that of one pulse: per crossing, the sum over the line classes of `max(v_k − 2√v_k (Σ_{j≠k} γ_kj √v_j + A),
  (√v_k − Σ_{j≠k} √v_j − A)²)` for that crossing's member `k`, and the largest over the crossings; its upper bound keeps
  every pulse together (reference `tests/test_pulse_fluence_events.py`: two 10 Hz lasers each within a rating alone, one
  source with two lines). A source's lines partition its average power, so the energy of one pulse along one path is
  `P / f_rep` however the spectrum is split. The window bound (light of unknown place) is taken per line class and not
  summed over a pulse's lines: valid, and looser than it could be for a multi-line source. Light of different lines or coherence groups adds (no interference term in the time average); beams
  of one coherence group and line add their fields with open phases, also when they cross the surface in opposite
  directions (a standing wave; the incident power, a flux, has no cross term between the two sides): between
  `max(0, √v_k − Σ√v_j)²` (and `Σv − 2Σγ√(v_i v_j)`) and `Σv + 2Σγ√(v_i v_j) ≤ (Σ√v)²`, `γ` the degree of coherence
  of the Gaussian spectrum of the declared coherence length after the two fields' relative *group* delay `τ` (the
  difference of their group optical paths `Σ n_g L`, `n_g = n − λ dn/dλ`, not of their phase paths) and group-delay
  dispersion (`Σ k₂ L`, `k₂ = λ³/(2π) d²n/dλ²`): `(1 + b²)^(−1/4) exp(−(τ/L_c)²/(1 + b²))`, `b = 2B/L_c²` (higher orders
  of the dispersion not included; the same in the plane analyses, `group_path_difference_mm`), at its largest over the
  surface (`τ` changes across it with the beams' tilt, offset and wavefront curvature, and through a plate of glass with
  the rays' angles: collinear beams keep about their chief rays' value, beams crossing at an angle are fully coherent
  somewhere in their overlap; 1 when none is declared, the path difference is undefined, a beam crossed glass other
  than a plate -- a lens, a prism, a cube --, whose envelope delay across the beam is not followed, or the two beams
  crossed passes of undetermined group delay differently: an element without a thickness in the model (an ideal or
  tabulated thin lens, a splitter plate, a polarizer, a retarder, a Faraday rotator, a diffraction grating, an attenuator, an aperture that attenuates) or a coating
  whose phase is not stated constant over wavelength (also met beyond the critical angle), listed per beam in `group_delay_unknown_at`, with the
  beam's polarization state where the delay may differ between polarizations (a coating met obliquely, a retarder, a
  polarizer passing light along both axes: beams in different states there have no known relative delay; reference: a
  polarization interferometer through a common half-wave plate or 45 deg coated fold, `tests/test_undetermined_group_delays.py`) --
  the two-beam coherence metrics then are stated at no added delay at those passes, with an `ESTIMATE` entry
  (`UNSTATED_GROUP_DELAY_ZERO`) that decides no target; a pulsed source's pulses may be mutually coherent (a mode-locked train), so two of its fields
  count as coherent again near every multiple of its period length `c / f_rep` -- the bounds take the larger coherence of
  the delay and of its distance to the nearest non-zero multiple, and a coherence metric has no value where such a revival
  could change it (reference: a Mach-Zehnder whose arms differ by one period length, `tests/test_pulse_train_coherence.py`); reference: a Mach-Zehnder balanced for its group delay, coherent at its white-light fringe with
  its splitters' phases stated and of unknown coherence without, `tests/test_group_delay_and_returns.py`; a thin filter
  or retarder or an attenuating aperture in one arm, a coating without a stated phase, `tests/test_undetermined_group_delays.py`) --
  Minkowski's inequality for the time-averaged field, valid for any spectrum. Upper bounds at the beams' peaks (as if they coincided), lower bounds at
  the traced beams' centres through each beam's Gaussian profile and, in peak irradiance and peak fluence, the window
  bound: light of unknown place (a ghost of unknown size, the light left out as negligible) may be at its bound at a
  beam's centre but cannot be strong over the whole area the beam covers -- a traced beam's mean against a window
  `exp(-s ρ²)` over its area is lowered by light of power `P` crossing the surface from one side by at most `H √P`,
  whatever its directions (`max |E| ≥ |∫ψ*E| / ∫|ψ|`, Cauchy-Schwarz against the flux, `H` the window's overlap with plane
  waves weighted by `1 / cos`), for a beam of one Gaussian mode on a plane or a sphere (its wavefront phase, the sphere's
  sag and the clear aperture's edge bounded); in `P / d` the same window with that light as Gaussian beams no more focused
  than the diffraction limit crossing the surface from one side, each lowering the mean by at most its `P / d` over the
  strip of the window it covers (its narrow axis crosses the surface at a slope of at least `θ / √(2 + θ²)`, `θ = w₀ / w`;
  a thin sheet at a shallow angle covers a strip about `w² / w₀` wide, so `P / d` gains less), for any traced beam on a
  plane or a sphere; reference: fields of known flux and beams crossing from one side against the bounds on a fine grid,
  an AR-coated two-singlet expander far over its rating in W/cm² and W/cm, `tests/test_window_bound.py`; equal bounds (one beam; incoherent beams with
  coincident peaks) make the exposure exact. Two equal interferometer branches: 4 I in phase, 0 in antiphase (reference:
  `tests/test_coherent_exposure.py`, the balanced Mach-Zehnder's bright and dark ports against the hand Gaussian and the
  Engine's two-beam analysis; a window's rear face against its own etalon; a retro-reflected beam on a splitter coating,
  the standing wave). `UNKNOWN`, never 0, without a valid Gaussian
  state (including the observation planes' aberration check); `OUT_OF_MODEL_SCOPE` behind a tabulated thin lens whose
  `supported_metrics` do not cover the exposure (the utilization, or `power_W` with a beam-size metric). With power
  ratings, untraced light (ghost reflections, coating transmissions, light passing beside an aperture or element) is
  followed by probe traces to the surfaces (the rule of the observation planes, with no exemption for a surface its
  line's traced beams also meet), each with the pupil rays of two whole rings of the beam it leaves and checked for
  aberrations like a traced beam (an aberration-dominated probe arrival counts with the diffraction limit of its power):
  it raises each surface's upper bound (`untraced_hits`) and, coherent with the traced beams of its group, may lower the
  lower bound. A crystal's face (`UniaxialMaterial` on a side) is read over the crystal's two indices: its reflectance bound
  is the largest s Fresnel coefficient over every pair of indices in the two sides' ranges (1 where one of them puts a
  critical angle within the angles; a positive crystal's extraordinary wave reflects more than its ordinary index gives),
  a cemented face between two cuts of one crystal reflects and turns light whatever their ordinary indices, the cones of
  light refracted there hold the refractions at the ratio's two ends widened by the walk-off angle
  `atan(|n_o² − n_e²| / (2 n_o n_e))`, and between parallel crystal faces the momentum windows and the walk of the returns
  are taken with the ray within that angle of its wave normal (reference: `tests/test_crystal_index_readers.py`).
  Light left out as negligible (the branches at most `negligible_power_fraction` x their line's input and the
  paths ended at the power floor) and light of unknown fate whose power is known (the trace's or the probes' budget, a
  path ended out of model scope; `exposures.faint_light`; its pieces are mutually coherent, so `P` is their power together
  in phase, `(Σ√p)²`, `in_phase_power_W`) is bounded by its diffraction limit `2πn²P/λ²` (the most a
  field of power `P` crossing a plane from one side reaches: Cauchy-Schwarz over its plane-wave spectrum; the Gaussian
  peak at waist `λ/(πn)`; for a broad spectrum times the mean of `(λ/λ')²` over it), times the most its open paths'
  crossings bring onto a surface or across a plane, `c = √2 m Σ_{j=0..N} a_j`, at least 2 (the light once
  from either side): between two reflections the light meets each lens and each refracting surface between media of
  different index at most once and travels in at most `L = T + 1` straight legs (`T` those lenses and refracting
  surfaces, which may turn it), each meeting any other surface at most `K` times, as often as a straight line can (a
  plane once -- a mirror, a splitter, a refracting plane between media of one index, an observation plane --, a sphere
  or a conic twice, an asphere at most twice the highest power of `ρ` in its sag, that power itself on a paraboloid or a
  flat base, twice that with an axicon term) -- so it meets a lens or a refracting surface between different media at
  most `m = 1` time and any other surface at most `m = K L` times, also where a prism or a lens turns it back across a
  plane or onto a mirror or a splitter, plane or curved --; after `j` reflections it is made of pieces from different
  sequences of `j` of the `N` surfaces that may reflect it (no surface reflects it twice on an open path), mutually
  coherent, whose fields add: `a_j` is the smaller of the sum over those ordered sequences of the products of `m_s √R_s`
  (`R_s` a surface's reflectance bound over every angle and side) and the square root of their number times the power
  they share (`√2` for the light split between the two sides each time), so `c` is 2.26 on the faces of four lenses with
  0.25 % coatings and 5.41 on an uncoated window's faces (a bare face counts 1), and grows quickly where many surfaces may
  reflect everything (reference: two splitters whose pieces recombine in phase, `tests/test_open_path_crossings.py`); on a
  transparent observation plane its power is at most `m Σ_{j=0..N} min(1, R^j)` (`R` the sum of the reflectance bounds,
  each times its `m`), at
  least 2, times the pieces' total power, on an absorbing one that power (reference: an open path through two fold
  mirrors crossing a plane three times, a window's ghost that one or two prisms or a thin lens turn back across a plane,
  a curved splitter and a refracting plane between media of one index that prisms turn the light back across, a
  spherical and an aspheric splitter that one straight leg crosses twice and four times, the most points a straight line
  meets a sphere, a conic, an asphere or an axicon at, `tests/test_open_path_crossings.py`). It grows where two surfaces facing each other return it (`validation/resonance.py`): each round trip
  between two reflecting surfaces that face each other -- a window's, a lens's or a cube's faces, a window and a mirror,
  the mirrors of a linear cavity, also with lenses or plates between them -- keeps at most `g = r_a r_b` of its field,
  the two amplitude reflectances bounded over every angle at which the pair's light can meet them (along legs between the
  surfaces' apertures; between parallel planes with nothing turning light between them, nor off their hull on a way
  from one to the other -- a refracting plane parallel to them keeps the light's momentum along them and counts in the
  walk, while a tilted plate or a wedge, which shifts the light sideways, turns it (reference: a tilted plate and a wedge
  pair between two offset parallel mirrors, `tests/test_return_pairs.py`) --, from the momentum along them that
  the rest of the scene brings in, which they keep, coming back
  onto the planes at most `min(ρ₁/s, ρ₂/s + 1)` times, `ρ₁ ≥ ρ₂` their apertures' radii and `s`
  the walk per pass; a thin lens turns light by at most `2 atan(ρ/2|f|)`, `ρ` the largest height of its aperture above its
  axis, and light after it meets a surface it straddles from either side; the surfaces between them are those that meet
  the two's hull and every lens or refracting surface two of them lie on opposite sides of), so all its returns bring at
  most `g/(1 − g)` times one crossing more. What each return sends out of the pair (through its surfaces, past an
  aperture's edge, reflected by a surface between them) is at most that return's field and may reach any surface: every
  surface and plane, from either side, gets `f = 1 + Σ g/(1 − g)` over the scene's pairs (`faint_light_return_factor`,
  `FAINT_LIGHT_RECIRCULATES`), coherent with every beam of its line on every surface (`FAINT_LIGHT_BOUNDED`); a rated
  surface no other light reaches lies between 0 and that bound, its ratings compared with that light's lines (regime,
  wavelength and each line's pulse train). From an interface's
  denser side only angles below the critical angle count, so at a bare interface a range reaching it counts the
  reflectance just below it, 1, and a coated face counts its coating's values over the angles below it (a catalog or
  Project coating holds its published values beyond its published data, as above); a coating an Engine request states
  with a validity range counts 1 outside it, and its pair bounds nothing; a record that states an air-equivalent angle range,
  met from a medium denser than air, takes the use of the left-out light (the coating angle reference, below). An interface takes only light in
  its own medium on each side (indices compared). A pair whose gain bound is 1 (two surfaces that reflect everything, a coating met outside its data, the faces
  of an uncoated lens, which light inside it can meet up to the critical angle), and light at the per-path interaction
  limit (light that may be stored in a cavity the traced light shows), bound nothing: no upper bound on any surface, and
  no lower bound for its lines (reference: a window's faces, a two-mirror cavity only left-out light enters, the light a
  cavity sends out through its mirror onto a plane and a window behind it, a cavity closed through a fast thin lens, a
  decentred one and one far off the line between the mirrors, parallel mirrors of different apertures, a coating met
  outside its data, a coated and an uncoated lens, `tests/test_return_pairs.py`).
  Closed returns through three or more surfaces of separate elements (a ring, a folded or coupled cavity, a periscope's
  windows through its folds) withhold the upper bound of every surface they may reach, with one
  `EXPOSURE_BOUND_WITHHELD` notice of the run naming the return (`closed_return_surfaces`) and those surfaces; a stack of parallel planes keeps an all-orders series
  of its returns where one converges (`1/(1 − A)` with `A = Σ m_s r_s < 1`, else the alternating series of its
  reflections along the planes' normal) and is withheld the same way where none does (reference: rings, folded and
  coupled cavities, plate stacks, `tests/test_closed_returns.py`, `tests/test_closed_return_stacks.py`).

  When the return analysis runs, and its work budget. The analysis of a wavelength (its pairs, then its closed returns)
  runs the first time a bound reads it: only a source line that left light out of the trace at that wavelength needs it,
  so a run whose lines leave nothing out never runs it, and the values are the same whenever it runs. Its work is counted
  in deterministic units (`src/os_optics/validation/return_work.py`): one per cone test of the closed-return search, 1000
  per pair of reflecting surfaces the pair screen examines, and the fold-solid analysis' cell-pair tests, cone comparisons,
  states and iterations of its sums (`fold_cell_work`, `FOLD_ENTRIES_PER_UNIT`; about 4 µs per unit on a desktop CPU). The
  limit is the run's: the wavelengths draw on it in the order the run first reads them (the observation planes in Project
  order, each plane's source lines in the trace's order, then the rated surfaces), each record stating the units the
  earlier ones spent (`work_before`) and what that left it (`work_limit`), so the whole run's analysis does at most
  `max_return_analysis_work` units however many lines leave light out; a wavelength that finds nothing left stops before
  its first unit (`INCOMPLETE`, `NOT_STARTED`). The pair screen grows
  with the square of the reflecting surfaces; the closed-return search can grow exponentially with them (a scene of 128
  reflecting surfaces: 8128 pairs, about 50 s, then a search that does not end in minutes). An analysis that reaches
  `max_return_analysis_work` units (the run's; `compute_budget.max_return_analysis_work`, default 4,000,000) stops:
  the pairs or the closed returns it did not finish are not known, so every upper bound that reads them is withheld --
  a plane's `power_upper_bound_W` null (its state and value unchanged: the left-out light is the stated truncation of the
  value), a rated surface without an exposure upper bound and with rating status `UNKNOWN`. The execution stays
  `COMPLETED`: the analysis is auxiliary, and its budget is listed in `outcome.execution.budgets.exhausted_by_phase`
  (phase `return_analysis`). One `RETURN_ANALYSIS_BUDGET` notice (status `NOT_APPLICABLE`) names
  the wavelengths, the work done, the run's limit, the stage it stopped in (`pairs`, `solid_folds` or `closed_returns`) and
  exactly the bounds it withheld: the power upper bounds of the planes whose power reads the returns (a transparent plane;
  an absorbing plane takes each watt once and keeps its power upper bound, `power_bound_observation_ids`), the
  coherent-power and irradiance upper bounds of the planes whose field reads them, and the surfaces' exposure upper
  bounds; quantities that
  do not read the analysis (traced powers, the power budget, exposures of traced light) keep their values (reference:
  `tests/test_return_analysis_budget.py`).

  The closed-return search examines trails of directed legs shortest first (every trail of 4 surfaces, then of 5, ...)
  and stops at the first closed return it cannot rule out, which is the one the issues name (other closed returns may
  remain: blocking it and running again names the next if one does). A trail never takes the same
  directed leg twice but may meet a surface again (a fold mirror on a folded return), so rings, folds and returns that pass
  a surface twice stay detected. The search holds one step per leg of the trail it is extending, at most one per directed
  leg between the screened surfaces, whatever the number of trails. Each wavelength analysed is recorded in
  `outcome.execution.budgets.return_analysis` with three answers: `COMPLETED_NO_RETURN` (every trail examined, none
  closes: the open paths' count bounds the left-out light), `POTENTIAL_RETURN` (`closed_return_surfaces`; bounded by
  `closed_series_bound` where an all-orders series of parallel planes applies, else by nothing; a plane-faced solid whose
  internal folds have no certified sum, `validation/solid_folds.py`, is one too, named by its faces, whatever the trail
  search found) and `INCOMPLETE` (the work limit came first, `termination` `BUDGET_EXHAUSTED`, or `NOT_STARTED` where the
  pair screen or the fold-solid analysis used it). `closed_return_kinds` says what each named return is: `path` (a closed
  trail, its surfaces in order), `parallel_stack` (a stack of parallel planes, which the trail search does not run on:
  `termination` `PARALLEL_STACK`, its surfaces as a group, bounded by `closed_series_bound` where a series applies) or
  `solid_folds` (a fold solid's faces). An `INCOMPLETE` screen bounds nothing, as a potential return does not: it is never read as a
  screen that found no return, and it names no return (a fold solid's included). The record
  also gives where the analysis ended (`ended_in`), its `work_done` (`solid_fold_work` of it in the fold-solid analysis),
  `work_before` and `work_limit`, the longest trail examined
  (`search_depth`) and its storage (`frames_peak` of `frames_limit`). A run cancelled during the analysis has no result
  (reference: `tests/test_return_analysis_budget.py`). Measured on a scene of three splitter cubes, a fold mirror and a
  sample mirror (17 reflecting surfaces in one group): a potential return of three surfaces found in about 46,000 units of
  search after about 511,000 units of pair screen, about 1.2 s per wavelength on a desktop CPU.

  A pair whose gain bound is 1, or a closed return no series bounds, leaves a plane without a power upper bound, and the run names each such plane with `EXPOSURE_BOUND_WITHHELD`, with or without a rating. A pair at gain bound 1 also leaves every surface of a rated run without an exposure upper bound, and the run names each with it. The notice has status `NOT_APPLICABLE`: no bound, comparison, target or status moves, and the manifest's `warnings` lists it. Stating a coating whose reflectance bound is below 1 on the pair's surfaces, or blocking the closed path, obtains a bound. Which code each case carries:

  - a pair at gain bound 1, rated surface: `EXPOSURE_BOUND_WITHHELD` at `surface:<id>`, `cause` `RETURN_PAIR_GAIN_BOUND_1` with `return_pair_surfaces` and, per pair in the same order, `return_pair_saturation` (why each of its two surfaces has a reflectance bound of 1: `CRITICAL_ANGLE_IN_RANGE`, `GRAZING_INCIDENCE_IN_RANGE`, `COATING_DATA_RANGE`, `COATING_EVALUATION_ISSUE`, `INDEX_UNKNOWN`, `REFLECTS_EVERYTHING`, `REFUSED_COATING_USE`) and `return_pair_angles_deg` (the angles of incidence its light meets each surface at); `return_pair_nominal_surfaces` / `return_pair_nominal_gain` name the pairs whose records state a round-trip gain at their centre-line incidence, with that gain; `return_scope` `SCENE_WIDE_NOT_LOCALIZED` says the pair's factor applies to every surface and plane (the surfaces its light can reach are not narrowed); `suggested_action` follows the cause;
  - a pair at gain bound 1, any plane with `power_upper_bound_W` null: `EXPOSURE_BOUND_WITHHELD` at `observation:<id>`, `quantity` `power_upper_bound`, the same `cause`;
  - a refused coating use, rated surface: `EXPOSURE_BOUND_WITHHELD` at `surface:<id>`, `cause` `COATING_USE_REFUSED` with `coating_record`, `incident_medium`, `incident_index`, `valid_aoi_deg`, `extent_in_medium_deg`, `air_equivalent_deg` and `coating_refusal` (`OUTSIDE_DECLARED_VALIDITY`, `OUTSIDE_DECLARED_WAVELENGTH`, `NO_AIR_EQUIVALENT_ANGLE`, `EXTENT_NOT_BOUNDED`); a surface held only through the pair a refusal leaves at gain bound 1: `COATING_USE_REFUSED_ON_FACING_SURFACE`; planes keep their bound and have no notice;
  - a closed return, a rated run: one `EXPOSURE_BOUND_WITHHELD` at `exposures`, `quantity` `exposure_upper_bound`, `cause` `CLOSED_RETURN_NOT_BOUNDED`, with `closed_return_surfaces`, the surfaces and elements it leaves without a bound (`surface_ids`, at most 12, with `surface_count`; `element_ids`), the surfaces of every group the trail search screens (`closed_return_candidates`: closed returns between separate elements run only among the surfaces of one group, so the list says where any other closed return can be), the round gain bound of the trail found (`closed_return_gain`: the product of its surfaces' reflectance bounds) and, for a solid whose folds have no certified sum, why (`solid_fold_reasons`: `HELD_LOSSLESS_STATE_WALK`, `CONES_NOT_CLOSED`, `ROUNDS_NOT_SETTLED` or `SUM_NOT_CERTIFIED` -- what the fold analysis could not certify, not a statement that the light circulates); any plane with `power_upper_bound_W` null: `EXPOSURE_BOUND_WITHHELD` at `observation:<id>`, `quantity` `power_upper_bound`, `cause` `CLOSED_RETURN_NOT_BOUNDED`, with the same facts (a dark plane also `OBSERVATION_NOT_REACHED`, `UNKNOWN`);
  - light at the interaction limit: `MAX_INTERACTIONS`, status `UNKNOWN`, at `path:<id>` (a dark plane also `OBSERVATION_NOT_REACHED`).

  A surface notice has `element_id`, `surface_id`, `quantity` `exposure_upper_bound`, `wavelength_nm` and `suggested_action`, and names an undecided `power_rating_utilization` target of its element in `affected_targets`. The run's closed-return notice names the undecided `power_rating_utilization` targets of its `element_ids`. A plane notice has `observation_id`, `wavelength_nm` and `suggested_action`, and names no target.

  **Coating angle reference.** Every coating record states the medium its angle range refers to: 629 of 707 records (catalog release 2026-10-28.1) declare an air-equivalent angle, 78 the incident medium. Met from a medium denser than air, an air-equivalent record is evaluated at the bounded angular extent of the light the trace leaves out, converted by `asin(n sin theta)`; a use outside the record's declared angle or wavelength range, a use above the critical angle (no real air-equivalent angle) or a use whose extent cannot be bounded is refused rather than approximated, and that face has no exposure upper bound (`EXPOSURE_BOUND_WITHHELD` names it, with the record, the medium, the extent and the air-equivalent angle). The refusal concerns the exposure upper bound of the refused face and of a pair it leaves at a gain bound of 1; a plane's power bound and every other surface read the record's stated value. The reference-design lens coating `AR_400-1100R0.01T0.99_AOI0-10` is published for 0-10 deg air-equivalent. In `focus` and `folded` the left-out light meets it from the glass within 9.82 and 9.80 deg air-equivalent: the value holds and the bounds stand. In `beam_expander` and `relay_4f` faint ghosts fill the lenses, their extent is not bounded below the record's range, and the four faces of each design have no exposure upper bound; the planes keep their power upper bound. At the 5.715 mm clear radius the face's normal alone gives 19.6 deg air-equivalent on R 25.8 mm and 9.65 deg on R 51.5 mm; the refusal stands on the ghosts' own rays, which reach 22-75 deg air-equivalent on those faces. To obtain a face's bound, use a coating whose record states those angles, or keep the beam well inside the clear aperture. A glass-side reflectance at high angles needs per-side or angle-resolved data. The extent covers left-out light from the reflection that leaves it out up to its next reflection, through refractions, plates and lenses, at the line's own level (`negligible_power_fraction` x its input); each later reflection of that light takes the coating's stated value at a field of at most `sqrt(R)` per reflection. Air-side incidence of an air-equivalent record, `INCIDENT_MEDIUM` records and uncoated surfaces read their stated values. The traced light's power reads the coating's value at the chief ray (a chief ray outside the record is `COATING_AOI_OUTSIDE_RANGE`).

  How the extent is found (`validation/coating_use.py`). Quantity: per coated dense side, surface and wavelength, the range of
  incidence angles, in the medium the light arrives from, of the left-out light's rays there. A left-out branch leaves a
  followed hit (trace or probe, every order above the line's level): the reflection or transmission with `P R <= level`, or any
  branch of a hit whose line the probes did not finish. It runs on an open path, through refractions, plates and lenses, to
  its next reflection. Each axis of a Gaussian beam fills an ellipse in `(x, x')` with radius `w`, wavefront radius `R_c`,
  far-field half angle `theta_far` (`M^2` included): any linear function of `(x, x')` over the light is Gaussian, with the
  share past `K` times its 1/e^2 half width `0.5 exp(-2 K^2)`, `K` taken from the followed hit. To first order a ray's
  incidence differs from the chief ray's by `x' - c x`, the normal turning by `c x` across the footprint: half width
  `D = K max_axis sqrt((w (1 / R_c - c))^2 + (w_0 theta_far / w)^2)`, `K theta_far` on a plane; on a sphere `c = +-1 / R`
  and `1 / (R cos i)` in the plane of incidence, the larger counting. Extent: `[asin(sin i - D), asin(sin i + D)]`. A hull is
  carried too, and is the bound where the Gaussian state does not hold: half width `x = K w` on the clear aperture, half
  angle `a = K theta_far`, `x -> x + t tan a`; normal turn `phi` 0 on a plane, `asin(sin i + x / R) - i` on a sphere, at
  most what the cone of normals over the clear aperture leaves on a sag surface; reflection `a -> a + 2 phi`, refraction
  `a -> M (a + phi) + phi` with `M` Snell's largest angular magnification over the angles met, thin lens `+ atan(x / |f|)`.
  Where both apply the narrower holds. A hit without a Gaussian state that describes it (invalid, centrally obstructed,
  aberration dominated) has the clear aperture and the cone of the leg from the surface before it. A branch in air whose
  envelope reaches an element its chief ray misses leaves that element's dense sides not bounded. A use not bounded is
  refused (reference: `tests/test_coating_use.py`).

  **Internal folds inside one plane-faced solid.** A plane-faced solid (beamsplitter cube, polarizing cube, penta prism, right-angle prism) whose left-out light can return to a face through two or more of its other faces has those returns counted. The directions inside are the mirror images of the cones the light enters with (from every surface outside the solid that redirects light and every source's beam, refracted in); a step from one face to the next exists only where a straight segment of that cone joins the two clear apertures, and carries the face's reflectance bound over the step's angles, 1 where they reach the critical angle (total internal reflection); a step that meets an air-equivalent coating record beyond its stated angles reads the record's stated value, and that face has no exposure upper bound (`EXPOSURE_BOUND_WITHHELD` names the record), as for the first run of the left-out light; the closed walks at each face are summed with a certified bound, and the open paths' counts add every walk that closes a loop other than an alternation with a face the pair bounds already count. The closed walks back to an entrance face sum to 0.0076 on the catalog beamsplitter cube, 0.0102 on the polarizing cube and 0.0049 on the penta prism (its fold, 0.005 x 0.99^2); exact ray trees traced inside these solids never exceed them. The faces' exposure upper bounds rise by 0.03-1.2 % on the cubes, whose decisions stay, and by 3.9-10.3 % on the penta prism, whose S4 then reads `CONDITIONS_NOT_VERIFIED`. A solid whose closed walks have no such bound (a lossless return, more than 192 direction cones) is a closed return: no surface the left-out light reaches has an exposure upper bound, and an `OUT_OF_MODEL_SCOPE` issue names its faces. A solid whose light meets each face once between the returns of one pair (a window, a plate, a wedge, a dove prism) keeps the pair bound; there the pair model leaves out total internal reflection at a face met beyond its critical angle (a wedge's walk toward its apex). A solid with a crystal body, a non-convex body or a curved face keeps the pair and open-path bounds without its internal folds. Which solids are analysed: the plane faces of one element with one isotropic body index, convex (every corner of a polygon face, as a mesh solid's planar patches are, and every other face's aperture extent on the body's side of each face's plane) and with no source inside: catalog prisms and cubes, and mesh solids (`FACET_SOLID`) alike, each cell of an arrayed mesh member its own solid. A closed mesh solid of glass (a corner cube, a penta, rhomboid or dove prism, a Fresnel rhomb, a light guide) has no certified sum: the analysis keeps, per direction cone and face, one part of the face's aperture for all the visits there, so light that advances along faces met beyond the critical angle or silvered closes a lossless walk of states although the light itself moves on and leaves (a rhomboid's or a light guide's zigzag between two parallel faces, a dove prism's between its base and top, a penta's two silvered faces and a face met beyond the critical angle), and a corner cube's cones, imaged again by its partly reflecting entrance face, do not close within 192 (nor do those of a mesh whose rounded coordinates leave its faces slightly off right angles); such a solid is a closed return (`POTENTIAL_RETURN`, no exposure upper bound where its left-out light reaches). The analysis does not bound it; it has found no lossless return of the light: ray trees traced inside these meshes over the light the model admits find every circulation passing a partial reflection, and finite per-face sums of a ray's visits in amplitude (a corner cube's back face 1.25 with a bare entrance face, 1.053 with an AR coating of reflectance 0.0025; a penta's 1.25; a rhomboid's 3.75 and a light guide's 8.5 at the faces of their zigzag, a few lossless visits per pass). A mesh cube of index 1.05 lit across a corner (no face near its critical angle) is bounded, and the traced walks stay within its sums. The open paths' counts leave out the walks whose only loops are runs of alternations with one paired face, which the pair factor counts; that factor counts one such run per walk and a run with one partner, so a walk whose loops are alternation runs at two or more faces of one open path, or a run alternating with two partners at one face, is counted by neither. Its share is second order in the runs' sums `A - 1`: at most `prod A - 1 - sum (A - 1)` over a solid's faces, 1.1e-3, 1.5e-3 and 3.7e-3 of the solid's whole closed-walk product on the catalog beamsplitter cube, polarizing cube and penta prism (0 on the right-angle prism, which pairs no face); exact ray trees inside these solids have not exceeded the bounds.

  **Light held by total internal reflection in a lens body.** The exposure upper bound on a lens face counts the single return between the face and its pair; light that enters the glass body and reaches the total-internal-reflection range is not counted. In the reference designs (focus, folded, beam expander, 4f relay, three-lens AR train) the internal angle reaches at most 0.9 deg against a critical angle of 41.4 deg (N-BK7 at 780 nm), no ray enters the TIR range, and the omitted term is zero (K = 1.000000). That angle is the entering beam's: ghosts that fill the same lenses reach 39.8 deg in glass, without reaching the TIR range, and the coating angle reference below states what they do to the exposure bound. A curved lens body has no bound on consecutive total internal reflections over arbitrary light inside it: in the 0.25 %-AR biconvex singlet (R 101.3 mm, 4 mm thick, 12.7 mm clear radius), rays started inside it in random directions meet its faces beyond the critical angle up to 12 245 times in a row before a partial reflection, so no series over its reflections converges for such light (this does not show that light the lens receives is held: at the reference designs' feeds none reaches the total-internal-reflection range); a bound on that light needs a limit on the power it holds, which the exposure model does not state. For wide entry cones the held power grows: at a 75 deg cone the 1 %-AR catalog lens holds 2.6 % of the power (4.6 % of rays ever reach TIR; amplitude factor 1.27, returned-power ratio 4.3); near grazing it holds 13.0 % (amplitude factor 2.36, returned-power ratio 16.0, up to 5 consecutive TIRs), and a 0.25 %-AR singlet reaches 16.9 % held (amplitude factor 4.53, returned-power ratio 69.6, up to 30 consecutive TIRs). A power-based bound on TIR-held light is needed only for such wide or grazing feeds; the reference designs are unaffected.

  **Returns among refracting surfaces.** The pair model, the open-path count and the `A < 1` series count a refracting surface at most once between two reflections; light that refraction turns back onto a surface it already passed before being reflected again is not counted. A bounded search (14 executed fixture runs and 540 configurations of a two-singlet strong-curvature train, reflections traced with an explicit budget) produced no traced path that meets a refracting surface twice between two reflections; refracting-surface returns themselves are traced (0.25 %-AR faces return 0.99 % of the incident power to a plane behind the source). The search did not cover plane-faced prisms and retroreflectors, ball and hemispherical refractors, trains of three or more elements, or catalog products. The rule: this return class is outside the exposure bound.

  Without ratings
  any unknown fate removes the upper bound. Where it alone keeps a comparison from being decided, those lines' light is followed deeper (to the level
  the decision needs, at most four passes, not below the power floor) and the exposures computed again (reference: the
  window behind pick-offs of 1e-7, 1e-5 and 1e-14 and at full power, `tests/test_coherent_exposure.py`; a faint branch
  focused onto a broad beam, 50 faint reflections, a faint p branch on an s surface, `tests/test_faint_light_bounds.py`; a
  cavity only left-out light enters, a window's etalon factor, an aberrating lens behind a leak, a 10 fs pulse's
  spectrum, `tests/test_group_delay_and_returns.py`). A beam whose chief ray misses a rated surface while its Gaussian footprint reaches
  the support (in the surface's vertex plane, a sphere's rim plane; transverse distance `s |cos AOI|`, a lower bound)
  bounds it by the profile there (`beside_hits`), so a tail is never a known zero. A comparison is
  `WITHIN_RATING` only when the upper bound is within the rating, `EXCEEDS_RATING` when the lower bound is above it, in
  the directions the rating's applicability allows. Every condition a rating can state is recorded
  (`RatingComparison.conditions`: verified, differs with its effect on the threshold, not verifiable with its reason, not
  stated): its regime, the wavelength within 2 %, pulsed, the duration (a rated duration read as a FWHM) and the
  repetition rate within 10 % (a test beam diameter, rate, count, polarization or value definition the rating does not
  state: not verifiable, `RATING_INFORMATION_MISSING`; otherwise
  no established relationship; followed light of a line the rating is not for that carries at most 1 % of the most light
  the surface can receive, in peak irradiance and every quantity that gives a comparison its direction, counts in the
  exposure but not in these conditions -- the rule for left-out light and minor beams; reference: a weak 532 nm beam on a
  1064 nm-rated window, on its axis and beside it, `tests/test_rating_conditions.py`); the test beam on the sample against every beam's 1/e² size on the surface -- verified
  within 5 % of a stated `1/E2` or `FWHM` diameter, a threshold at least the rated value for a beam whose footprint is no
  larger than the test beam (a smaller spot meets fewer defects, and heats less at a given fluence or irradiance), at most
  the rated value for a beam narrower nowhere than it (small test spots overstate the threshold of larger beams); a
  diameter without its definition is read as a 1/e² diameter from the stated value to 1/0.5887 = 1.70 times it (FWHM
  0.589, 1/e and ISO effective-area 0.707, second-moment and 1/e² 1 of a Gaussian's 1/e² diameter: the standard widths;
  a model assumption, a width taken at a lower level would fall outside), and a beam between
  those readings is not verifiable; beams of both directions, a beam whose axes straddle a stated diameter, a pulsed
  rating with a test beam on a metallic reflector (the multi-pulse threshold of metal mirrors was found to rise with the
  spot size) unless it is a single-pulse rating applied to one pulse per site, and a `W/cm` rating with a larger beam
  (the linear power density removes only the first-order thermal dependence on the size) have no established
  relationship; light of unknown size bounded by its diffraction limit withholds a verified or smaller-beam direction
  beyond 1 % of the most light the surface can receive, and a beam with no known value there always does, while the
  weakest beams that carry, with the followed light of unknown size, at most 1 % (a faint ghost of another size) are left
  out of the comparison (the left-out light is then checked with them: it never makes a faint ghost decide the size) --
  both shares in every quantity that gives the size its direction: the rating's own (peak irradiance, P/d or peak
  fluence: a micro-spot at a low repetition rate may carry little of the irradiance and most of the fluence) and, for a
  CW rating, the power and P/d (a broad beam's heating), for a pulsed one the peak fluence and the pulse energy, a beam
  whose chief ray misses the surface with its tail bound `exp(-2 d²/w²)` in each (`d` the miss distance, `w` its larger
  1/e² radius); the pulse count `S` against the pulses one site receives -- each crossing of a source's light brings that
  source's `pulse.total_pulses` (`UNLIMITED`: more than any rated count; the lines of one source along one path are one
  crossing), crossings that together carry at most 1 % of the most light (in the quantities above) do not count, a site
  receives from the fewest one counted crossing brings (`low`) to all of them (`high`; unbounded where left-out light may
  carry more than 1 % or light may be stored) --, compared per tested site only for a single-pulse rating (one pulse on
  each tested site) or a count stated as applied at one level to each tested site (`test_protocol` `S_ON_1`): damage is
  irreversible (a site damaged within `S` pulses at a fluence is damaged within more; a site that survives `S` survived
  its first fewer), so `low = high = S` is verified, `low ≥ S` gives a threshold at most the rated value, `high ≤ S` at
  least the rated value, `low < S < high` is `PULSES_PER_SITE_UNDETERMINED`, a source that states no `total_pulses`
  `DESIGN_INPUT_MISSING`; a rating that states no count, or a count above 1 without its protocol (one level, ramped,
  raster -- a ramped test's value may lie above or below the one-level value), is `RATING_INFORMATION_MISSING` whatever
  the totals; continuous light differs (reference `tests/test_rating_evidence_corrections.py`,
  `tests/test_pulse_count_input.py`:
  one and two crossings, several sources, a weak source within the share); the test polarization for the light as a whole (s / p in each beam's
  plane of incidence, one linear axis, one circular state, unpolarized: every beam within 1 % of one common state, so any
  mixture is -- beams of another state that carry at most 1 % of the most light count in the exposure, not in the state;
  it fails only where the light at a traced centre violates it for certain; reference
  `tests/test_rating_polarization.py`, Fresnel `t_s`, `t_p` written out, crossed incoherent beams) -- a polarization in
  other words, s / p at normal incidence, beams in different states whose mixture is not decided, or left-out light of
  unknown state that may carry more than 1 % of the most light the surface can receive (in peak irradiance or in the
  rating's own quantity, whichever share is larger) is not verifiable, and left-out light of a line the rating is not for
  beyond that share leaves each condition the line differs in (its regime; else its wavelength, pulse duration, repetition
  rate) not verifiable -- on a surface only left-out light reaches, where light may be stored between surfaces facing each
  other (nothing bounds its share), whatever its power, the stored light's lines included; a `polarization_condition`
  that says more than its `polarization` field (an orientation, a handedness) is not verifiable either, and one that names another state than the field (`TM` beside `S`, `circular (RHC)` beside
  `LINEAR`) is refused as input. The applicability: `NOT_APPLICABLE` (no established relationship, or opposite
  directions: `CONDITIONS_MISMATCH`), `NOT_VERIFIED` (`CONDITIONS_NOT_VERIFIED`, its exposure -- exact, or its bounds -- still reported), at
  least rated (decides only `WITHIN_RATING`), at most rated (only `EXCEEDS_RATING`), `AS_RATED` (both; reference
  `tests/test_rating_conditions.py`: fixture diameters, the catalog's micro-spot, sampler, CW cube, silver-mirror,
  single-pulse and 10000-pulse ratings by product id). A `W/cm` rating without a diameter definition is compared with P/d
  from the 1/e² to the FWHM diameter (the standard width definitions of a Gaussian beam lie in that range). The rated
  value as a threshold for the peak exposure (`rated_threshold_low` / `_high`): `PEAK`, the value; `TOTAL_PER_1E2_AREA`
  (the total energy or power over `π (d/2)²`, `d` the 1/e² diameter: `E / (π w²)` against the peak `2E / (π w²)`), twice
  it; not stated -- every catalog rating -- none: the `value_definition` condition is `RATING_INFORMATION_MISSING` (test
  reports use several definitions, among them the energy over a FWHM-diameter area, whose peak threshold is `R ln 2`, and
  no set is known to be exhaustive); a `W/cm` value is its own threshold; a `TYPICAL` value bounds none (reference
  `tests/test_rating_evidence_corrections.py`, `tests/test_rated_value_reading.py`).
  Fluence and irradiance are compared in the beam's transverse plane; no rating states its test angle or plane. A rating not verified may apply as stated or with a threshold at most its value, so a surface is
  within only when the exposure's upper bound meets it too and a rating of its quantity that applies bounds it (its
  lowest threshold no larger than this one's highest); a surface's utilization (exposure / threshold: lower bounds against
  `rated_threshold_high`, upper bounds against `rated_threshold_low`) interval runs from the largest lower bound of its as-rated
  and at-most comparisons to, when an as-rated or at-least comparison exists, the largest upper bound of those and the
  ones not verified; an at-most comparison leaves no upper bound unless an as-rated or at-least comparison of its
  quantity has a rated value no larger (the threshold then lies between the two; where all have larger values the ratings
  disagree: the lower bound stands, no upper bound); an element's from the largest of its surfaces' lower bounds to the largest of their upper
  bounds (none while a rated surface has none; reference `tests/test_rating_decisions.py`). No scaling laws, no thermal or
  damage model. An exceedance is a notice
  (`POWER_RATING_EXCEEDED`, status `NOT_APPLICABLE`): it never changes the run's status; a `power_rating_utilization`
  target makes it a requirement. Light whose power is out of model scope (`ArrivalResult.power_out_of_scope_at`: behind a
  crystal whose second-harmonic generation is out of model scope -- a pulsed source --, a variable attenuator or a
  transmission map that varies across the beam, an element or absorbing plane part of the beam meets beside its chief ray:
  with a bounded share that the element stops -- an absorber that does not scatter -- `incident_power_W` is an upper
  value, `value_basis`)
  has no exposure on the surfaces it reaches (`ExposureHit` `OUT_OF_MODEL_SCOPE`, `NONLINEAR_CONVERSION_OUT_OF_SCOPE` or
  `OUT_OF_MODEL_SCOPE`; no irradiance, fluence or pulse energy, no bound), and a `power_rating_utilization` target of their
  element is `OUT_OF_MODEL_SCOPE`. References: `tests/test_power_exposure_ratings.py` (radius at the surface by independent
  ABCD through a thin lens and a flat interface; `2P/(πw²)`, `P/d`, `2E/(πw²)`) and `tests/test_exposure_bounds.py`
  (untraced light on a rated element, partial ratings, the response-model scope, the run status with an exceedance).
* **Geometric pair metrics.** `centroid_separation_mm`, `normalized_centroid_separation`, `angular_separation_rad`,
  `geometric_overlap_ratio` (`∫√(I_a I_b) dA / √(P_a P_b)` of the footprints on the plane) and
  `waist_position_separation_mm` compare two beams of any wavelengths (`wavelength_nm` / `wavelength_nm_b`): geometry of
  the chief rays and the Gaussian intensity envelopes only, no phase or coherence (`tests/test_geometric_pair_metrics.py`:
  offset and tilted sources, two lines focused by an achromatic thin lens).
* **Not modelled.** Diffraction beyond the Gaussian beam (PSF, MTF, diffracted fields), the Gouy phase in `phase_rad`,
  partial polarization / depolarization, scattering, untraced ghosts, coherent sums beyond the paraxial superposition; a
  pulse's spectral width against a coating's band, dispersive pulse broadening, mirror group-delay dispersion; how far a
  threshold falls with more pulses than a rating's count, or with a repetition rate other than its own.

---

### Exposure records

The JSON schema of a result states, for each exposure record, what the record is, when a value of it is null and where
its model is documented; the model itself -- how beams combine on a surface, what the bounds count and leave out, how each
rating condition is judged, with the derivations and the cases -- is the record's documentation in the Engine
(`help(os_optics.contracts.<Record>)`, the class docstring in `os_optics/contracts/results.py`), one statement per record:

| Record | What it is |
|---|---|
| `ExposureReport` (`exposures`) | every surface a traced beam meets and, with power ratings, every surface untraced light reaches; whether the trace accounts for all launched power |
| `ElementExposure` | one element: its surfaces' exposures and its worst rating utilization |
| `SurfaceExposure` | one surface: the traced beams on it, their combined peak values and bounds, and the rating comparisons |
| `ExposureHit` | one traced beam at one surface: power, Gaussian radii, angle, footprint and peak values |
| `RatingComparison` | one power rating against one surface's exposure: conditions, exposure in the rating's unit, utilization, status |
| `RatingCondition` | one condition of a rating against the light: verified, differs (with the effect on the threshold) or not verifiable |

`tests/test_schema_text_budget.py` holds the schema's text to that form: no description beyond a stated length, and a
stated total.

## P10b. Neutral-density attenuators (`NeutralDensityInteraction`)

A thin plane: power transmittance `T = 10^-OD(x, y)` with the optical density of the element's law at each ray's hit point
(`FIXED`; `LINEAR_ROTARY`: OD linear in the hit point's azimuth about the element axis minus the wheel rotation, from
`od_start` at `arc_start_deg` to `od_end` at `arc_end_deg`; `STEPPED_ROTARY`: constant OD per sector; `LINEAR_GRADIENT`:
OD linear along local X minus the shift), `coated_inner_radius_mm` for a coating on an annulus around an uncoated hub (a hit
point closer than that radius to the law's centre is uncoated; the centre is the element axis, for a gradient shifted with
the filter), `uncoated_transmittance` where the law leaves the plane uncoated (without it the chief ray's light there passes
on with a power that is not computed, `ND_UNCOATED_REGION` with `power_upper_bound_W`), a constant `reflectance` R traced as a branch (phase convention of
an ideal reflector, π for s and 0 for p) and `1 - R - T` absorbed (ledger `ATTENUATOR_ABSORPTION`); `R` plus the highest
`T` (lowest OD, or the uncoated transmittance) cannot exceed 1. The packet's power and Gaussian state use the chief ray's
`T`; the OD range over the beam's 1/e² footprint (32 points of the ellipse projected along the chief ray, plus its centre;
with a hub also the footprint point nearest the hub centre, or the hub centre itself when the footprint covers it, so a
beam across the hub's edge counts the uncoated density) -- or, for a beam without a valid Gaussian state (behind an axicon,
an annulus), over its live pupil rays -- beyond `ND_FOOTPRINT_OD_TOLERANCE` = 0.2 is `ND_DENSITY_VARIES_ACROSS_BEAM`
(`OUT_OF_MODEL_SCOPE`; the Gaussian state is withheld from there on, the beam's power is out of model scope:
`ArrivalResult.power_out_of_scope_at`, and power-derived targets on it are `OUT_OF_MODEL_SCOPE` with no value). For a linear gradient of `ΔOD` across the 1/e²
diameter the beam-averaged transmittance exceeds the chief-ray value by `exp((ln 10 · ΔOD / 2)² / 8) − 1` (0.66 % at the
tolerance) and the transmitted centroid moves by `ln 10 · ΔOD / 8` beam radii (6 %). References:
`tests/test_neutral_density.py` (each law against `10^-OD` written out, the wheel rotation / gradient shift, the footprint
range against `OD(x ± w)`, the averaging formula against quadrature) and `tests/test_neutral_density_annulus.py` (the
annulus law and the hub's uncoated transmittance, a footprint straddling the hub's edge, a hub inside a large footprint).

## P10c. Faraday rotators (`FaradayRotatorInteraction`)

A thin plane whose field axis is the plane's normal `z` (local +Z of a `FARADAY_ROTATOR` model). The field vector turns by
`theta(lambda)` right-handed about `z`, whichever way the light travels. In the ray's right-handed transverse basis
(`e1 x e2 = d`):

    J_out = sqrt(T(lambda)) [[cos b, -sin b], [sin b, cos b]] J_in,   b = sign(d . z) theta(lambda).

A reciprocal rotation or retarder has no such sign: its matrix does not depend on `d . z`, so going through it and coming back
undoes the turn, while a Faraday rotator doubles it. `theta` and `T` are the stated rows of the model: linear in the vacuum
wavelength between rows (`LINEAR`, over the rows' span) or at the rows' exact wavelengths (`POINTS_ONLY`, and a one-row
table). A wavelength outside that domain has no response: the light passes the plane straight on with a power and
polarization that are not computed (`MODEL_WAVELENGTH_OUTSIDE_BAND`, `OUT_OF_MODEL_SCOPE`, with `power_upper_bound_W`; power
ledger `TERMINATED_OUT_OF_MODEL`, an unknown fate, once), so no power of that line exists behind it, only its bound and
geometry; other lines keep theirs. Nothing is extrapolated or held from a nearby row, and no
Verdet constant, magnet field or temperature law is applied: the rotation at a condition other than the stated
`operating_conditions` is not computed. `valid_aoi_deg`, `operating_conditions` and `evidence` are optional statements of
the model: without `valid_aoi_deg` no angle of incidence is checked, and a table without `evidence` is an assumed response
(`element_models[].response.data_kind` `ASSUMED`).

`theta` applies unchanged at every chief-ray angle of incidence inside `valid_aoi_deg` (for a uniform field along the plate
normal the path `L / cos t` and the field along it `B cos t` cancel); outside it `ELEMENT_AOI_OUTSIDE_RANGE`. The loss `1 - T`
is ledger `ELEMENT_TRANSMISSION_LOSS`; a lost share that depends on polarization (a dichroic rotator) is not modelled. The
element has no thickness in the trace: the Gaussian beam passes unchanged, the body's length and index act on nothing, and
it states no optical path. So `ray_opl_defined` is false behind it (no phase or optical path length in the handoff, wavefront
`RAY_OPL_UNDEFINED`), the pass is recorded as of undetermined group delay with the beam's polarization state (two beams that
do not share it have no stated phase difference or coherence, section P6), and a linear optical network through it is refused
(`PHASE_UNDEFINED_FARADAY_ROTATOR`).

An isolator is a path, not a property of the element: polarizer, rotator, analyzer, read with one source launched from each
end, per source (`PowerBudgetLine.observed_W`, sources are independent launches). With reference planes on the laser side
(`IN`) and the load side (`OUT`) of the path: insertion loss `10 log10(P[FWD, IN] / P[FWD, OUT])`, isolation `10 log10(P[REV,
OUT] / P[REV, IN])`, for the launch polarizations stated. With a polarizer `T1, ER1` at 0 deg, an analyzer `T2, ER2` at 45 deg
and rotation `theta`: forward `T1 T (T2 cos^2(theta - 45) + (T2 / ER2) sin^2(theta - 45))`, reverse `T2 T (T1 cos^2(theta + 45)
+ (T1 / ER1) sin^2(theta + 45))`. That number holds for those launches and that path: reflections from other surfaces, a load
that returns another polarization, and a measured device's own isolation are other statements.

References: `tests/test_faraday_rotator.py`: the Jones matrices of both directions and the port powers against the formulas
above derived by hand in the test (relative 1e-12), the round trip against a half-wave plate that sends the forward beam to the
same output, the same isolation path with that plate (isolation = insertion loss), a polarizer / quarter-wave plate / mirror
trap.

## P10d. Diffraction gratings (`DiffractionGratingInteraction`)

A thin periodic plane of period `L`, normal `n`, unit grating vector `K` in the plane (local +X turned by
`grating_vector_angle_deg` toward local +Y; the grooves run along `g = z x K`). An order `m` leaves with

    n2 t2 = n1 t1 + m (lambda / L) K,    t = d - (d . n) n,    d2 = t2 + s sqrt(1 - |t2|^2) n,

`lambda` the vacuum wavelength, `n1` the index on the arriving side, `n2` that side's index for a reflection order and the far
side's for a transmission one, `s` sending the light back on the arriving side or on across the plane. This is the general
vector grating equation (Loewen and Popov, *Diffraction Gratings and Applications*, ch. 2): the part along `g` is conserved, so a
conical mount needs no rule of its own. In the plane of `K` and `n`, with an angle positive toward `+K`:
`n2 sin(theta_m) = n1 sin(theta_i) + m lambda / L`; an order `+1` at normal incidence turns toward `+K`. Every pupil and
differential ray takes the same equation with its own direction, so the first-order transfer and the rays' spread come from
the order's real map. Where `|n2 t2| >= n2` the order is `NONPROPAGATING` (it has no direction, is reported in `grating_orders`, and
is not traced): a followed order that is not `all_propagating` is also a `GRATING_ORDER_NONPROPAGATING` warning.

Gaussian beam. The grating adds a constant tangential wavevector: a transverse offset maps by `M` (along `d1` onto the plane, then
across `d2`), and slopes by `(n1 / n2) M^-T`, so `q' = (n2 / n1) s^2 q` along each singular direction of `M`, `s` its stretch: in the
plane of incidence `s = cos(theta_m) / cos(theta_i)`, across it 1. In a conical mount those directions are neither: an astigmatic
beam off them is `GAUSSIAN_TWISTED_ASTIGMATISM`, a round one turns onto them.

Each order is a branch path (`branch_kind` `grating_order_+1`), the arriving path ends `GRATING_ORDERS`, and the rest of this section
is what each quantity of an order means. **Direction** is always available. **Power** comes from the stated response:

* `response` absent, a transmission grating met against +Z, a conical angle beyond `max_conical_angle_deg`, a wavelength or angle
  outside its domain (nothing is extrapolated), a `RELATIVE` table, an `UNPOLARIZED` table (the traced beam is fully polarized:
  at most twice the mean, at most the incident power, is what it says): `GRATING_RESPONSE_UNAVAILABLE` (`UNKNOWN`), the incident
  power goes to the ledger's `UNMODELLED_RESPONSE` once, and every order is followed with its power `null` and
  `power_upper_bound_W`. Such an order books nothing further, no power floor ends it, and it follows every branch of its own
  through a splitter; a plane reads its position and direction, and its power is `UNKNOWN` (transparent: it may be crossed more
  than once) or `BOUNDED` (absorbing: at most the power of unknown fate). A target on a position or direction is decided; a power
  target is graded on the interval.
* an `ABSOLUTE`, `TE_TM` table where it holds: with `f_TE`, `f_TM` the beam's power fractions along the grooves and across them, an
  order carries `P (f_TE eta_TE + f_TM eta_TM)`: the two add incoherently, whatever their relative phase, because they are the
  grooves' eigenpolarizations in a planar mount. Stated `absorption` goes to `ELEMENT_TRANSMISSION_LOSS`; the rest of the
  incident power -- orders the table does not represent or the run does not follow -- goes to `UNMODELLED_RESPONSE`.
  A followed order the table does not list is bounded by what the listed orders leave.

**Polarization** is known behind a `TE_TM` table with purely TE or TM light (within `POLARIZATION_MATCH_FRACTION`), and behind a table that
states TE = TM for every order at that wavelength and angle: the response then does not depend on the polarization, and an order keeps the state
that arrives, as a deflector's table holds whichever polarization arrives. Otherwise efficiencies state no phase, so a mixed state's relative phase is open and `beam.polarization` is `null`. A surface whose share depends on it
(a polarizer, a coated reflector or splitter, an interface off normal incidence, another grating) then opens its power too: from there
on it is bounded by what it was. Such a beam forms no pair: a pair target whose selection may hold it reads `UNKNOWN`, `PHASE_UNDEFINED`, not a missing pair. **Phase** is not stated: `phase_rad` and the optical path length have no value behind the
element, two beams' phase difference and coherence follow the rule of a pass of unstated delay (`delay_key`, with its order and hit
point), and network exports are refused (`PHASE_UNDEFINED_GRATING`). A response table's `data_kind` (`MEASURED`,
`INDEPENDENTLY_COMPUTED` with `evidence`; `ASSUMED`) is reported with every order; product names and line densities establish
nothing about efficiency. Not modelled: resolving power, diffraction by the finite aperture, pulse dispersion, the thickness and
index of the plate.

## P10e. A pupil mask, its focal field and a detector, stage by stage

Two post-processes (`os_optics/results/pupil_field.py`, `os_optics/results/detector.py`), each a separate model that traces no ray and changes no run. The chain they form is

```
generator (a mask, or a run's beams) -> pupil -> focal field -> detector
```

and every quantity says how it is known: `MODELED` (computed here), `EXTERNALLY_CHARACTERIZED` (a stated value, used as stated, with its source), `BOUNDED` (the known light, with the range the light a map leaves out allows, `RESULTS_AND_EXPORT.md`) or `UNAVAILABLE` (with a reason; what
does not need it keeps its value). A stage that cannot run reports why and passes nothing on: no independent Gaussian takes the place of a missing field, and a detector parameter that
is missing leaves the field as it is.

### Focal field (`pupil_field`, request `PupilFieldRequestV1`)

**Conventions.** Field `U exp(-i w t)`, propagation `exp(+i k z)`, `|U|²` in W/mm². A mask multiplies `U` by `A exp(+i φ)`, so `φ = 2π x sinθ / λ` tilts toward `+x`. The pupil lies in
the front focal plane of an ideal thin lens (`exp(-i k r² / (2f))`), which leaves no quadratic phase and no inversion on the back focal plane:

```
U_f(u, v) = 1 / (i λ f) ∫∫ U(x, y) exp(-i 2π (x u + y v) / (λ f)) dx dy      (constant phase exp(i 2 k f) left out)
```

The integral is the forward FFT (kernel `exp(-i 2π k m / N)`) of the zero-padded pupil grid: `u_k = k λ f / (N_x dx)`, `k = -N_x/2 .. N_x/2 - 1`, and `+x` maps to `+u`. The pupil grid is
`x_m = (m - (n_x - 1)/2) dx`, with phases referred to its centre.

**Pixels.** Without `fill_factor` the grid samples a continuous pupil. With it, each cell holds one value over a rectangle `a = dx sqrt(fill)`, and the exact transform is the FFT times
`a_x a_y sinc(a_x u / (λ f)) sinc(a_y v / (λ f))`. The window `|u| < λ f / (2 dx)` holds the principal diffraction order of the pixel grid; the replicas beyond it are `power.higher_orders_W`,
by Parseval's theorem on the continuous field. Light on the gaps between pixels is `power.dead_area_W` and is not propagated. A beam or a target spot beyond the window is refused: it would
fold back into it. A pixelated ramp beyond it folds back as the device does.

**Power.** `power` is the ledger incident -> on the pupil (`clipped_W` beyond the opening) -> active (`dead_area_W`) -> after the mask (`absorbed_by_mask_W`) -> delivered (`not_delivered_W`,
of a stated device efficiency `η`, which scales the field by `sqrt(η)`; that light has no modeled place on the focal plane) -> window (`higher_orders_W`). The discrete Parseval relation holds
for every transform and is reported (`parseval_relative_residual`). Without a stated efficiency, `device_efficiency` is `UNAVAILABLE` and the power is that of an ideal device.

**Sampling.** The transform has at least twice the illuminated span, so the irradiance `|U_f|²`, whose bandwidth is twice the field's, is fully sampled; a smaller `fft_size` is refused.

**Beams.** A `BEAMS` or `HANDOFF_BEAMS` illumination puts one TEM00 field per beam on the pupil in the convention of `results handoff`: with the beam's transverse coordinates `(x1, x2) = M (x - x0, y - y0)` (`M`: its two Gaussian axes' components along the plane's `u`, `v`; the identity for a stated beam) the irradiance on the plane is `2 P / (π w1 w2) |det M| exp(-2 (x1²/w1² + x2²/w2²))` (`|det M|` the cosine of the tilt: the flux through the plane) and the curvature phase `k (x1²/(2 R1) + x2²/(2 R2))`, so an elliptical beam of any orientation is carried as it is. `COHERENT` fields add with their stated phases; one beam alone without a phase is computed with its constant phase taken as 0 (`phase_reference` `UNDETERMINED_CONSTANT`).
`INCOHERENT` beams add in irradiance, one transform per beam, with no focal phase: the time average over an exposure much longer than the inverse of their frequency difference, as for the tones
of an acousto-optic deflector.

**Spot array.** A `SPOT_ARRAY` mask is generated by weighted Gerchberg-Saxton iterations (Di Leonardo et al., Opt. Express 15, 1913 (2007)) with the pixel envelope in the field at each spot, from a
deterministic chirp of start phases, from which the iteration does not amplify rounding (a start linear in the spot index does). It is a generator, not a retrieval from measured intensities.

**Validity: the scalar paraxial domain, marginal-ray sine up to 0.2.** The marginal-ray sine is `pupil_na + field_sine`: the pupil's rim (down to 1e-4 of its peak irradiance) as seen from the
focus, plus the output grid's corner as seen from the lens. At 0.2:

| Effect left out | Size at 0.2 |
|---|---|
| Focal scale: `tan θ` against `sin θ` | `θ² / 2` = 2 % at the rim |
| Aplanatic apodization `sqrt(cos θ)` | 1 % in amplitude, 2 % in power at the rim |
| Longitudinal field of a linearly polarized pupil, share of the energy `<sin²θ cos²φ>` = `NA² / 4` | 1 % |

Above it `focal_irradiance` and `focal_phase` are `UNAVAILABLE` (`NA_ABOVE_SCALAR_LIMIT`), `status` is `PARTIAL` and the pupil stage and its ledger stay. A tightly focused spot is the
vector focal field of P10g. Polarization, aberrations, fringing fields, phase calibration and the device's response in time are not modeled
(`not_modeled`).

**Validated against closed forms** (`tests/test_pupil_field.py`; tolerances set before the first run and stated there): a uniform circular pupil against the Airy pattern (first zero `1.22 λ f / D`, 1 %;
the pattern `(2 J1(x) / x)²`, 0.2 %; the peak `P A / (λ f)²`, 1e-9; encircled energy `1 - J0² - J1²`, 1 %); a blazed ramp of `q` pixels per period (the spot at `u = λ f / (q p)`, its peak `sinc²(1/q)`, 1e-9;
the window power `fill sinc²(sqrt(fill) / q)` of the active power, 1e-3 at 1024 pixels; the replicas summed directly against the ledger, 2 %); a binary 0-π grating (first order `(2/π)²` = 40.5 %,
third `(2/(3π))²`, 1e-6; no zero or even orders); field phases (`U_f(0) = -i sqrt(P A) / (λ f)`, 1e-9 rad; the shift theorem); a Gaussian (`w_f = λ f / (π w)`, peak `2 P / (π w_f²)`, 1e-3); a clipped Gaussian
(`1 - exp(-2 a² / w²)`, 0.2 %); coherent and incoherent beams; Parseval across the pupil and the focal window (1e-12). A 5 x 5 Gerchberg-Saxton array of 0.3 mm pitch from 640 pixels of an 8 mm pupil gave
power uniformity 0.992 (`1 - (max - min) / (max + min)`) and 77.7 % of the power within the spots' Airy radii, where a perfect spot holds 83.8 %. The generator does not amplify rounding: a shift of the targets by 1e-13 mm moves the mask by less than 1e-8 rad. Not validated: any mask or device against a measurement.

### Detector response (`detector_response`, detector `DetectorModelV1`)

An irradiance map (`W/mm²`) of a run (`irradiance_map`) or of a focal field, and what the detector states, give for one wavelength over one exposure `t`:

| Quantity | Formula | Needs |
|---|---|---|
| `photon_flux_density` | `I / E`, `E = h c / λ` | nothing stated |
| `pixel_power_W` | `I` integrated over each light-sensitive rectangle (`pitch sqrt(fill)` wide) | pitch, fill |
| `photons` | `pixel_power t / E` | + exposure |
| `signal_electrons` | `QE(λ) photons`; QE linear between stated rows, never extrapolated | + QE |
| `dark_electrons` | `i_d t` | dark current, exposure |
| `shot_noise_variance_e2` / `dark_noise_variance_e2` / `read_noise_variance_e2` | `signal` / `dark` / `σ_r²` | as above |
| `total_noise_variance_e2`, `snr` | their sum; `signal / sqrt(total)` (null where saturated or where the variance is 0) | all of them |
| `saturated` | `signal + dark ≥ full well` | + full well |
| `adu`, `adc_clipped` | `min(signal + dark, well) / gain`, limited to `2^bits - 1`: the mean count above the black level, unrounded | + gain, bit depth |
| `photocurrent_A`, `electrical_noise_current_A`, `electrical_snr` | `R(lambda) P_pixel` with `R = QE q lambda / (h c)`; `sqrt((2 q I + floor^2) B)` with the `electrical` floor and noise-equivalent bandwidth (P9d); their ratio (null where the noise is 0) | pitch, fill, QE and `electrical` (a detector that does not state it gives none of these); no exposure |

A map of grid points is read as Keys' cubic convolution, integrated exactly, with `pixels.integration.relative_error_estimate` its difference from the bilinear reading at the brightest pixel; a map of bins is constant
in each bin. An Airy peak sampled just at its Nyquist step under a pixel as wide as the step reads 3.2 % low; at 1.6 times that density, 0.8 %. A pixel array reaching beyond the map is `PIXEL_GRID_OUTSIDE_MAP`,
never read as dark. Pixel crosstalk, lag, nonlinearity and ADC quantization noise are not modeled.

**Validated against closed forms** (`tests/test_detector_response.py`): the photon count of a known power, wavelength and exposure (1e-12); the integral of a Gaussian over pixels against erf, within the trapezoid rule's
bound `h²/12 |f''/f|`, with the error falling at least as `h²`; the noise sums and the SNR (1e-12); a run's map against the source's power.

### Acousto-optic deflector (`deflector.acousto_optic`)

A generator that is a Project element: one thin transmission grating of period `v / df`, `df` the greatest common divisor of the RF tones, each tone its integer order `m f_RF / df`, so that (grating relation of P10d)
`sin θ = sin θ_in + m λ f_RF / v`. The direction is the Bragg relation of the momentum a phonon gives a photon and does not depend on the crystal's index: Snell's law at its parallel faces cancels it. `m` is `+1` or `-1`; the order's optical frequency shifts by `m f_RF`,
which `design.tones` reports (a run traces no frequency). Each order's power is `EXTERNALLY_CHARACTERIZED`, from a stated table for this set of tones, or `UNAVAILABLE`, its direction still traced; the zero order stays. Tone phases, timing, RF crosstalk, intermodulation, the crystal's length
and Bragg selectivity are not modeled (`design.quantities`). Tones differ in optical frequency, so a detector integrating much longer than `1 / Δf` sees their irradiances add: `pupil_field` takes them `INCOHERENT`, and a run states no phase to add them with.

**Validated** (`tests/test_acousto_optic.py`, 1e-9): positions on a screen against `L tan(asin(λ f / v))`, tone spacing `λ Δf / v`, the zero order at the axis, the period and orders, the tones' powers against a stated table.

### The chain

`tests/test_modulator_chain.py` carries a worked example with hand-derived numbers. 852 nm, `f` = 200 mm, an 8 mm pupil on 12.5 µm pixels, a camera of 6.5 µm pixels, 0.1 ms: 1 nW in one spot has a peak of `1.7311e-6` W/mm²
and `7.034e-11` W on the peak pixel (`1 - (π D / (λ f))² p² / 24` = 0.9617 of the peak value), that is 3.017e4 photons, 1.810e4 electrons at QE 0.6, SNR 134.5, 3017 ADU at 6 e/ADU. A ramp of 8 pixels per period moves it
to `u` = 1.704 mm with `sinc²(1/8)` = 0.9496 of it. A 5 x 5 array of 25 nW reaches the camera with each brightest pixel on its target and between 0.8 and 1 of that count; 16 tones of 1 nW at 2 MHz steps (650 m/s) keep their
524.3 µm spacing, their summed power, and their common peak `2 P / (π w_f²)`. The values of that camera and that velocity are chosen for the example, not a product's. `tests/test_acousto_optic.py` runs the same with the deflector in a Project: 16 tones of 25 to 55 MHz, 20 nW at
5 % each, read from the run's beams at the pupil plane, reach `u` = 6.554 mm and on at 524.3 µm, with the peak `5.410e-8` W/mm² and 9.80e3 photons in 1 ms on a 6.5 µm pixel.

## P10f. Linear cavities: round trip, eigenmode, resonances (`cavity_analysis`)

An analysis of a stated cavity (`os_optics.local.cavity`, `os-optics project cavity`), written as an analysis document
(`AnalysisResultV1`, analysis `CAVITY`) bound to the Project's Engine input hash and its settings; it changes no run.

### Governing equations

* Placement: the paraxial analysis's (P1 reflection, the unfolded axis): the elements listed once from end mirror `M1` to end
  mirror `M2`, the light leaving `M1` towards the next element. `P` is the forward pass from `M1`'s vertex to `M2`'s vertex
  (free gaps and the elements between, plane folds the identity), `P_r = ((D, B), (C, A))` of `P` the same elements met in
  reverse order, and the round trip at the reference plane (after `M1`) `R = M1 P_r M2 P`.
* Stability: `m = (A + D) / 2`; stable for `|m| < 1`, marginal within `1e-9` of `|m| = 1`, unstable beyond; `g1 g2 = (1 + m) / 2`
  (`g_i = 1 - L / R_i` of two bare mirrors).
* Eigenmode: `1/q = (D - A) / (2B) - i √(1 - m²) / |B|`, `w² = -λ / (π Im(1/q))`, propagated element by element; the waist of
  each gap's beam at `-Re(q)` from the gap's start with `w0² = λ Im(q) / π` (a waist outside the gap is that beam's virtual one).
  A symmetric cavity (`M1 = M2`, `P` with `A = D`) takes `q` from the half round trip `H = M2 P`, whose square is the round trip:
  the symmetric confocal cavity has `R = -I` (every beam reproduces) and `H` gives its mode `w0² = λL / (2π)`.
* Gouy phase per round trip `θ ∈ [0, 2π)`: `cos θ = m`, `sin θ` of the sign of `B` (symmetric: `2 θ_H`). Two mirrors:
  `θ = 2 acos(±√(g1 g2))`, the sign of `g1`.
* Resonances: `FSR = c / L_rt`, `L_rt` the round-trip group path (twice the free gaps plus `n_g t` per medium of a transmissive
  element, `n_g` of P2's dispersion; free space `n = 1`). Transverse-mode spacing `FSR θ / (2π)` (modulo the FSR).
  Round-trip power fraction `ρ = Π R_i^{k_i}` (an end mirror `k = 1`, a fold `k = 2`), `g = √ρ`, finesse `F = π √g / (1 - g)`,
  Airy linewidth `Δν = (2 FSR / π) asin((1 - g) / (2 √g))`, defined for `(1 - g) / (2 √g) ≤ 1`.
* Mode matching of a TEM00 input beam stated at the reference plane: `η = Π_{x,y} 2 √(Im q_a Im q_b) / |q_a - q_b*|`.
* Crystals (a Project's `birefringence`) between the mirrors: the cavity has an ordinary and an extraordinary family of modes. The
  request's `polarization` (`ORDINARY` / `EXTRAORDINARY`) chooses one where every crystal's optic axis lies in its faces and the
  crystals' axes are parallel across the light. The ordinary wave's plate has the reduced length `t / n_o` and group path `n_g,o t`;
  the extraordinary wave's (wave-vector surface `k_c²/n_o² + k_⊥²/n_e² = k0²`) `t n_e / n_o²` in the section along the optic axis,
  `t / n_e` across it, and `n_g,e t`. Without `polarization` (`POLARIZATION_NOT_STATED`) or without a common linear eigenmode
  (`CRYSTAL_EIGENMODES_UNDEFINED`) every quantity that reads a crystal's index (stability, eigenmode, group path, FSR, mode spacing,
  linewidth, mode matching) is `OUT_OF_MODEL_SCOPE`; an extraordinary eigenmode is also where the axis lies at an angle to the
  section axes (`CRYSTAL_AXIS_OBLIQUE_TO_SECTIONS`) or the crystal's element has power (`CRYSTAL_IN_POWERED_ELEMENT`), its group path
  and FSR standing. Reference (`tests/test_crystal_index_readers.py`): a 10 mm calcite plate in a 200 mm cavity at 1064 nm, FSR
  725.41 MHz (ordinary) and 731.70 MHz (extraordinary) from the Sellmeier group indices, and both eigenmodes against NumPy round-trip
  matrices to 1e-9.

### Assumptions / applicability (checked; each quantity states its own state, reason and `value_basis`)

A quantity carries the number the model computes together with what that number is (`value_basis`, RESULTS_AND_EXPORT "Value
basis"): a `BOUNDED` quantity its value beside its proven interval (an upper or lower bound, or an estimate inside the interval);
a quantity outside a validity limit, or computed from a stand-in input, its value in state `OUT_OF_MODEL_SCOPE` / `UNKNOWN` with
kind `ESTIMATE` and the failed condition or the stand-in named. Only a `KNOWN` value or a proven interval decides a target. The
document's `execution` is `COMPLETED` for every answered request: the analysis has no budget.

* Stable cavity: else `CAVITY_UNSTABLE` / `CAVITY_MARGINALLY_STABLE`: no eigenmode exists, so the Gouy phase, mode radii, waists,
  mode spacing, mode matching and the injection quantities are `OUT_OF_MODEL_SCOPE` without a value.
* Paraxial: the mode's far-field half angle at most `gaussian_scope_divergence_rad`, its radius over a mirror's radius of curvature
  at most `gaussian_max_w_over_radius` (the Project's `numerical_settings`); else `MODE_OUTSIDE_PARAXIAL_LIMITS`.
* Apertures far outside the mode: every element's clear radius at least `aperture_over_mode_min` mode radii (request field,
  default 3: `exp(-18)` of a Gaussian's power beyond); else `MODE_CLIPPED_BY_APERTURE`.
  Under either of the last two the round-trip matrix still has its eigenmode: its quantities carry the matrix's values as
  `ESTIMATE`s whose `validity` rows state the observed value, the limit, the setting that holds it and the element
  (`mode_radius_mm.<id>` also that element's own aperture ratio), and `arrays.waists` is filled. A clipped mode adds a loss no
  input states (finesse, linewidth and the resonant quantities become bounds for any loss); no mode of injected light is resolved,
  so the seven powers are bounded from zero. A limit the caller set and the case meets is named beside a `KNOWN` value (`MODEL`).
* Coaxial elements: `coaxial_tolerance` `{lateral_mm, tilt_rad}` (request field; default 1e-6 mm and `1 - cos <= 1e-12`, about
  1.414e-6 rad, as in the paraxial analysis; a curved mirror only at normal incidence, whatever `tilt_rad` says, since a tilted
  one has no single 2 x 2 matrix). `arrays.elements` states each element's measured `decentre_mm` and `tilt_rad`; the
  analysis is the nominal-axis one (a tilt changes a matrix at second order in the angle, a decentre deviates the axis by the
  decentre times the element's power, which is not reported).
* Reflectances: the coating's published data at the wavelength and angle of incidence, or stated in the request
  (`EXTERNALLY_CHARACTERIZED`). Outside the bands or the angle range the record states, the value is the nearest published one,
  held -- the number a run traces with -- as an `ESTIMATE` (`REFLECTANCE_OUTSIDE_COATING_DATA`, `HELD_AT_NEAREST_STATED_POINT`),
  inherited by the round-trip power fraction, finesse and linewidth; the injection waits for a stated coefficient. A fold mirror
  whose coating reflects s and p differently reports `reflectance_s.<id>` / `reflectance_p.<id>`; `polarization` `S` / `P` chooses
  one where the tilted mirrors share one plane of incidence and no crystal, polarization element or Faraday rotator lies between the
  mirrors (else `REFLECTANCE_DEPENDS_ON_POLARIZATION`). A reflectance stated without its transmittance leaves `T` `BOUNDED` in
  `[0, 1 - R]`.
  Losses the analysis does not evaluate (elements inside the cavity, clipping, an unstable or marginal cavity) only lower `ρ`: the
  finesse is then `BOUNDED` above and the linewidth below (`PROVEN`), each with the mirrors' value and a basis that names the loss
  taken as zero (`NO_UNSTATED_LOSS`).
* Group path: an element whose model states no thickness (an ideal or tabulated thin lens, a polarization element, an attenuator, a
  Faraday rotator) adds none, as its paraxial matrix takes it at one plane; the group path, the FSR, the mode spacing and the
  linewidth are then `ESTIMATE`s that name the element (`THIN_ELEMENT_PATH`).
* Not modelled: ring and astigmatic cavities (a curved mirror off normal incidence is refused by the placement), a source's own
  higher-order mode content (M² > 1), a pulsed or broad-linewidth source (below), the coatings' own reflection phases (they shift
  every longitudinal mode alike). A plane-plane Fabry-Perot etalon is marginally stable: no Gaussian eigenmode, its eigenmode and
  injected build-up are `OUT_OF_MODEL_SCOPE` (`CAVITY_MARGINALLY_STABLE`) while its free spectral range, finesse and linewidth stand. A
  cavity shorter than 10 wavelengths is refused (`CAVITY_TOO_SHORT`). A reflectance `R = 0` is accepted (a round trip returning
  nothing: finesse 0, no half maximum).

### Injection and resonant build-up (`injection`)

Light injected through the first listed mirror `M1` (power `P`: a stated `power_w`, or what a Project source on the axis behind `M1`,
emitting along it, delivers to `M1`). `M1` and the last mirror `M2` have power reflectance `R`, transmittance `T` (the coating's at the wavelength and
AOI, or stated) and loss `A = 1 - R - T`; one pass between them keeps `F = Π R_fold · √(1 - L)` of the power (`L` the stated
`round_trip_loss`). With `b = √R2 F` and the round-trip field gain `a = √R1 b`, a mode carrying `p` at round-trip phase `δ` from
resonance (`local/cavity_build_up.py`):

* circulating (forward field leaving `M1`) `P_c = T1 p / ((1 - a)² + 4 a sin²(δ/2))`; at `M2` `F P_c`; transmitted `T2 F P_c`;
* reflected `P_r = p ((√R1 - (1 - A1) b)² + 4 √R1 (1 - A1) b sin²(δ/2)) / ((1 - a)² + 4 a sin²(δ/2))` (the input mirror reflects
  `-√R1` from outside, `+√R1` from inside and transmits `√T1` both ways: the Stokes relations);
* losses: `A1 (p + b² P_c)` in `M1`, `A2 F P_c` in `M2`, `(1 - F)(P_c + R2 F P_c)` at the folds and inside. Every channel has its
  own formula; their sum equals `p` (`power_budget.closure_residual_W`: rounding only).
* On resonance (`δ = 0`) a mode-matched beam: build-up `T1 / (1 - a)²`, transmission `T1 T2 F / (1 - a)²`. The reflected field is
  `√R1 - (1 - A1) b` per unit incident field: under-coupled where `√R1 > (1 - A1) b`, over-coupled where it is smaller, no reflection
  (critical coupling) where they are equal: `R1 = (1 - A1)² R2 F²` with the input mirror's own loss `A1` held
  (`input_loss_matching_reflectance`), `R1 = R2 F²` for a lossless `M1` (`impedance_matching_reflectance`);
  `arrays.injection.coupling.regime`.
* Detuning: stated, `detuning_Hz` from the nearest TEM00 resonance (`δ = 2π Δν / FSR`), or `length_offset_nm`, the cavity's length less
  that resonance's (`δ = 2π · 2 ΔL / λ` reduced to `[-π, π]`). A Project's nominal positions do not fix it: the round-trip phase turns by
  `2π` per `λ/2` of length, far below a placement's precision, so a detuning computed from them would be arbitrary. Without either, the
  detuning is `UNKNOWN` (`DETUNING_NOT_STATED`) and each power is `BOUNDED` over every detuning: each channel of each mode is a ratio of
  two functions linear in `sin²(δ/2)`, monotonic in it, so its extremes are on resonance and half-way between resonances. The modes
  resonate at different detunings (their Gouy shifts): the bounds of the sum over the resolved modes come from branch and bound over the
  TEM00 round-trip phase (`cavity_build_up.swept_extremes`: on an interval each mode lies between its values at the interval's
  smallest and largest `sin²` of half its phase; intervals are halved and pruned against the best midpoint value until the bound is
  within `1e-9` of it, or 256 intervals would be followed at once -- a broad minimum of many modes can stop there, within about
  `1e-4` of the extreme), never looser than the sum of each mode's own extremes; the unresolved light's bounds are added.
  `arrays.injection.at_tem00_resonance` lists the resolved modes' channels with TEM00 on resonance (`not_resolved_W`: the power in the
  modes not resolved, which they leave out).
* A Project source (`source_id`, beam `SOURCE`): its light is traced from the source to `M1` (`validation/pipeline.py`, the source
  alone); `P` is the power the trace delivers to `M1`'s coating (after `M1`'s own substrate surfaces, e.g. the Fresnel loss of an
  uncoated back face). How the light reaches the mirror is a condition of the injection quantities alone: the cavity's own
  quantities are those of the request without the source. With anything else on the way (a stop, a lens, a window; an observation
  plane only records), a source off the axis or one that is not a single-line Gaussian beam, the beam at the mirror is not the
  source's (`CAVITY_SOURCE_UNSUPPORTED`): mode matching and the mode fractions are `OUT_OF_MODEL_SCOPE`, the injected power is the
  traced arriving power, and the seven powers are `BOUNDED` from zero by it as light not resolved. Where the light does not reach
  the mirror, or one of the elements' model conditions fails on the source's path (`local/validity.py` on its pupil rays): part of
  the beam meeting an element its chief ray misses (`BEAM_PARTIALLY_INTERCEPTED`, the message stating the share of the equal-power
  pupil samples it takes), light leaving an element through its edge -- the chief ray's power counts the whole beam there --, the
  arriving power is not known either and the injected power and the seven powers are `OUT_OF_MODEL_SCOPE`.
  `arrays.injection.traced_arrival` records the emitted and arriving power, the path, the elements met and the conditions found.
  Open: carrying the beam through coaxial elements before the mirror. That test reads
  the pupil samples, with their detection limit (P1, partial interception: an edge beyond the outermost ring, about 1.14 beam
  radii from the chief ray at the default 8 x 12, takes up to about 1 % of the beam unseen; a stop 1.2 / 1.5 beam radii from the
  axis, 0.8 / 0.13 % of the beam, leaves the injection as without it).
  The injected beam is the source's own TEM00 beam (waist and position from its model; an astigmatic one with its axes along the
  sections) carried through `M1`'s substrate by the paraxial matrix of its spherical faces (gaps `t / n`, faces `(n2 - n1) / R`) to the
  reference plane; its mode matching is reported (`mode_matching_efficiency`) and its HG content used. A stated `input_beam` must equal
  that beam (waist within 1e-6 relative, position within 1e-6 of the larger of 1 mm and the Rayleigh range), else it is refused.
* Polarization with a crystal in the cavity (`polarization` `ORDINARY` / `EXTRAORDINARY`): the source's Jones vector, in the section
  frame, divides its power between the crystals' eigen-polarizations: the fully polarized part `|E · c|² / |E|²` along the optic axis
  `c` (extraordinary), `|E × c|² / |E|²` across it (ordinary); a partially polarized source adds half its unpolarized part to each
  (`polarization_family_power_fraction`). The family analysed carries that share; the rest is `other_polarization_W` in the power
  budget, so the two families' requests together never exceed the input. A stated `power_w` is the power in the family analysed.
* Transverse content: a coaxial TEM00 beam `q_a` against the eigenmode `q_b` of one section couples `η = 2 √(Im q_a Im q_b) / |q_a -
  q_b*|` into `HG_0`, `η C(2k, k) 4^-k (1 - η²)^k` into `HG_2k` and nothing into odd orders (the overlap integral of `H_2k` with a
  Gaussian; completeness gives `|μ|² = 1 - η²`); `HG_mn` takes the product of the sections and resonates at `δ00 - m θx - n θy`; a
  round beam in a round cavity has `LG_p0` fractions `η2 (1 - η2)^p` (`η2 = ηx ηy`), the sum of its `HG` group of order `2p`.
* Resolved orders: `N = m + n ≤ mode_order_max` and every element's clear radius at least `3 √(N + 1)` mode radii (three classical
  radii `w √(N + 1)` of the order), its far-field angle `√(N + 1) θ0` and its radius over a mirror's radius of curvature inside the
  paraxial limits. The rest is bounded for any phase and any extra loss: circulating `[T1/(1 + a)², T1/(1 - a)²]`, transmitted
  `[0, T1 T2 F/(1 - a)²]`, reflected `[0, min(1, ((√R1 + (1 - A1) b)/(1 - a))²)]` of its power; the totals are then `BOUNDED`
  (`PROVEN`), a transmitted, reflected or lost power never above the power followed. Its fraction is `1 - Σ listed` (the listed
  fractions summed exactly, `math.fsum`) where that is at least 1e-6, else the direct sum of the series (each section's tail bounded
  geometrically): never above 1. The same bounds cover the whole beam where the intracavity losses are not known.
* Three groups, each with its own inputs. The resonant quantities and `arrays.injection.coupling` read the mirrors' coefficients
  alone; `detuning_Hz` is the stated one; the seven powers read the source and the beam too. A coefficient that is not known
  withholds only its readers (`missing` names the request field). With the detuning and the losses stated a power's value is the
  resolved modes' sum: `KNOWN` where the light not resolved is at most `negligible_power_fraction` of the injected power (the
  proven interval beside it, `convergence.resolution` the most that light can add), else the `LOWER_BOUND` of a `BOUNDED`
  quantity. Without a stated `round_trip_loss` for the elements between the mirrors the values are those without that loss (upper
  bounds for the build-up, the transmission and the matching reflectances on resonance; estimates for the powers), inside
  intervals that hold for any loss. Without a detuning no value exists and the channels with TEM00 on resonance are the
  quantities `at_tem00_resonance.<channel>`.
* A failed source or beam condition selects the bound that still holds (the condition is the powers' `reason`, its numbers a row
  of the injection model's `validity`): a pulsed source, or one whose linewidth is above `injection.source_linewidth_fraction_max`
  (default 0.01) of the cavity's (`SOURCE_NOT_SINGLE_FREQUENCY`): bounded over every detuning, each spectral component sitting at
  some detuning; a source with `M² > 1` (`SOURCE_NOT_FUNDAMENTAL_MODE`) or an injected beam whose radius at the reference plane is
  above `1 / aperture_over_mode_min` of `M1`'s clear radius (`INPUT_BEAM_CLIPPED`): all the light is light not resolved, bounded
  from zero.
* Out of the model: an unstable or marginal cavity (no eigenmode: `OUT_OF_MODEL_SCOPE` with its reason); a tilted or offset input
  beam. Refused: a detuning (or scan bound) at or above the light's frequency `c / λ`.

Targets. The request's `targets` (`quantity_id` with `minimum` / `maximum` / `target_value` ± tolerance) and the Project's targets of
a cavity metric (`validation/scope.py` `CAVITY_METRICS`, e.g. `cavity_transmission` -> `transmission`) are assessed in `arrays.targets`:
`MET` / `UNMET` from a `KNOWN` value or a `PROVEN` interval wholly inside / outside, else `INDETERMINATE` (an interval straddling the
threshold, or a quantity not known); `NOT_APPLICABLE` for a Project target of another component, wavelength or configuration;
`INDETERMINATE` with `TARGET_UNIT_MISMATCH` for a Project target whose `unit` is not its metric's (`CAVITY_METRIC_UNITS`: `W`, `Hz` or
`1`; the Project check warns of it, as for a run's metrics). Their
thresholds form `identity.requirements.targets_hash`; the physical identity does not move with them. A run reports the same targets
`OUT_OF_MODEL_SCOPE` with the route to this analysis (`os-optics results cavity RUN`, json_api `run_results` kind `cavity`, which
analyses the run's Project and binds the document to the run's Engine input and result hashes).

The run beside the analysis. The trace follows single passes; the analysis does not enter the run's result or its hashes. A beam
meeting the back of a mirror whose coating transmits ends with `OUT_OF_MODEL_SCOPE` (cause `MIRROR_TRANSMISSION_NOT_TRACED`) (path termination
`BLOCKED_BY_BACK_SIDE`, its power the ledger's `BLOCKED_BACK_SIDE`), and light circulating between the mirrors stops at
`max_interactions_per_path`. The exposure model's return pair of the same two mirrors (P8, `validation/resonance.py`) has the round-trip
field gain `g = r_a r_b`, each `r` the amplitude reflectance bounded over the angles the pair's light meets: `g ≥ a`, so its power
factor `(1 + g/(1 - g))² = 1/(1 - g)²` bounds `P_c / (T1 p)` at every detuning, and equals the on-resonance build-up for an
angle-independent coating (`tests/test_cavity_injection.py`, `test_the_exposure_model_s_return_bound_for_the_same_mirrors_covers_the_resonant_build_up`).

Independent references (`tests/test_cavity_injection.py`): the enhancement `T1 / (1 - √(R1 R2))²` and each loss channel; impedance
matching; the detuning curve against the coefficient-of-finesse form `1 / (1 + 4a/(1 - a)² sin²(πν/FSR))` and the half maximum at
half the linewidth; complex field sums for the reflection; the round-trip phase from the exact rational number of wavelengths; the HG
content of waist, position and combined mismatches against overlap integrals by quadrature (Hermite polynomials) and the closed form
`(2 w0 w / (w0² + w²))²`; the LG content against radial overlap integrals (Laguerre polynomials); Gouy-shifted resonances; the
confocal cavity's even orders resonating together; bounds that contain the higher-order result. `tests/test_cavity_injection_scope.py`:
the power a stop passes `P (1 - exp(-2 a²/w²))` (the refusal's traced power), the Fresnel loss `((n - 1)/(n + 1))²` of the input
mirror's back face and the beam through its substrate by hand-written ABCD matrices, the source's mode matching by the closed-form
overlap of two Gaussian beams, Malus's law for the crystal families, the Airy range over every detuning, the unresolved fraction against
the listed fractions summed in rationals, the lossy input mirror's matching reflectance against complex field sums.

### Independent references

`tests/test_cavity_analysis.py`: the symmetric confocal cavity (`w0² = λL / 2π`, Gouy `π`, spacing `FSR / 2`, `F = π √R / (1 - R)`),
plane-concave (`w0² = (λ/π) √(L (R - L))`), half-symmetric, asymmetric, convex-concave (virtual waist) and near-concentric cavities
against the g-parameter formulas to `1e-11`; an unstable and two marginal cavities; a 45° fold equal to the straight cavity with the fold's
reflectance counted twice; a plane-lens-plane cavity against round-trip matrices, quadratic roots and a segment-by-segment Gouy sum
written in the test; the group path through an N-BK7 lens against a Sellmeier difference quotient; mode matching against the overlap
formula in radii and curvatures.

## P10g. Vector focal field of an ideal aplanatic objective

A post-process (`os_optics/results/focal_field.py` over the diffraction package `os_optics/diffraction/`) that traces no ray and changes no run. Its reply is an analysis document
(`AnalysisResultV1`, analysis `FOCAL_FIELD`): every quantity has a state (`KNOWN`, `OUT_OF_MODEL_SCOPE`, `NOT_APPLICABLE`), its model basis and its convergence evidence.

**Objective and pupil.** An ideal aplanatic objective of effective focal length `F` and numerical aperture `NA`, with focal medium index `n`: free of aberration at every field point used,
obeying the sine condition `h = F n sin θ` (pupil radius `F NA`, reference sphere of radius `f' = n F`). The pupil is lit in air by a stated uniform, annular or Gaussian field (optionally tilted:
direction cosines `l` focus at `F l`), a set of TEM00 beams (`PupilBeamV1`, coherent or incoherent), or the beams a run delivers to the pupil plane (`HANDOFF_BEAMS`, with each beam's Jones vector).
Polarization: linear, circular, a Jones vector, or unpolarized (two orthogonal halves added in irradiance). An optional pupil phase is a list of Zernike terms (unit-RMS normalization over the
pupil).

**Model (Richards and Wolf, Proc. R. Soc. A 253, 358 (1959)).** A pupil point at azimuth `φ` maps to the ray `s = (-sin θ cos φ, -sin θ sin φ, cos θ)`; its radial field becomes
`e_p = (cos θ cos φ, cos θ sin φ, sin θ)`, its azimuthal field stays `e_s = (-sin φ, cos φ, 0)`; the amplitude gains `sqrt(cos θ)` (energy conservation). The focal field is the Debye integral
`E(r) = -i f' / λ_m ∫ a(s) exp(i k s·r) dΩ`, `H` likewise from `s × a`. Units: a plane wave in the focal medium has `|E|²` equal to its irradiance, so `|E|²` and the axial Poynting flux
`S_z = Re(E_x H_y* - E_y H_x*)` are in W/µm² (positions in µm). A window is a plane-parallel plate normal to the axis between the objective and the focus: each plane-wave component gains the
phase `k0 t (n_w cos θ_w - n cos θ)` and its single-pass Fresnel transmissions `t_p`, `t_s` (or 1 for ideal anti-reflection surfaces; Török et al., J. Opt. Soc. Am. A 12, 325 (1995)); light
reflected inside the plate forms other, weaker foci and is not included.

**Three evaluations of one integral.** A rotationally symmetric pupil of one polarization (uniform, annular or centred Gaussian, untilted, radial Zernike terms only) is computed by the Bessel
reduction (`I0`, `I1`, `I2` by Gauss-Legendre quadrature in `θ`; `os_optics/diffraction/bessel.py`) and checked by the direct sum over a polar Gauss-Legendre x uniform-`φ` sampling of the sphere
(`debye.at_points`). Any other pupil is computed by the matrix Fourier transform over a Cartesian grid of direction sines (rim cells weighted by their area inside the pupil;
`debye.on_grid`, Leutenegger et al., Opt. Express 14, 11277 (2006)) and checked by the direct polar sum. Every value is computed again with every node count doubled (`convergence`); the
check by the other method is in `arrays.reference` (the relative difference of `|E|²` at the peak and at every reported width crossing). Node counts grow with the distance of the evaluated
region from each spot, so a tilted or multi-spot pupil is sampled finely enough.

**Quantities.** The power ledger: stated power, power inside the pupil (polar quadrature of the pupil irradiance), clipped by the rim, sent through the window; the flux through the focal plane
over one period of the grid transform, where Parseval's theorem holds exactly (`parseval_relative_residuals`, at the stated and doubled grid), and `energy_relative_residual` = (flux - transmitted power) / transmitted power,
which closes as `1 / grid_cells` because rim cells cut the pupil. Per spot: the peak `|E|²` and its position (alternating axial and lateral golden-section searches), the full widths at half
maximum along x, y (through the peak, in its plane) and z, the half widths at `1/e²` of the peak along x and y, the radii of the circles around the peak on its plane holding 50 % and 80 %
of the transmitted power (`S_z` integrated by Gauss-Legendre annuli), the power within two Airy radii, and the Strehl ratio: the peak against that of the same pupil without its phase
(Zernike terms and the window's phase; transmissions kept), 1 for the ideal objective without a stated phase, `NOT_APPLICABLE` for several spots. The share of `E_z` in the integral of `|E|²`
over a transverse plane follows from the sphere by Parseval's theorem (`f'² ∫ |a_z|² / cos θ dΩ` over `f'² ∫ |a|² / cos θ dΩ`); the grid transform's value is the check. The Gaussian waist
`λ F / (π w)` is the model's value for a centred Gaussian beam within the paraxial Gaussian scope (far-field angle `w / (n F)` at most `gaussian_scope_divergence_rad`) that the pupil does not clip
(pupil radius at least `2 w`); otherwise the quantity is `OUT_OF_MODEL_SCOPE` (`NON_GAUSSIAN_FOCUS`) and holds the formula's number as an `ESTIMATE` whose `value_basis.validity` lists the checks that failed
(parameter, limit, observed value); it decides no threshold, and the widths and encircled-energy radii describe the spot.

**Validity.** The Debye integral needs a Fresnel number `F NA² / λ` of at least 100 and a focal region within 0.1 `f'` of the focal point (`validity` of the model basis). Outside them the focal
quantities are `OUT_OF_MODEL_SCOPE` (`DEBYE_APPROXIMATION_INVALID`) and the pupil ledger stays; the analysis ran to its end (execution `COMPLETED`). A beam no field can be built from is refused: not TEM00
(`BEAM_NOT_TEM00`), not in air (`BEAM_NOT_IN_AIR`), several coherent beams one of which has no phase (`PHASE_NOT_DEFINED`; one beam alone is computed, `arrays.phase_reference` `UNDETERMINED_CONSTANT`), a beam with no Jones vector (`POLARIZATION_NOT_DEFINED`), no handoff for
`HANDOFF_BEAMS` (`HANDOFF_REQUIRED`); an invalid request is `FOCAL_FIELD_INVALID`. An `INCOHERENT` sum of handoff beams leaves the beams no field can be built from out and lists them (`arrays.illumination.excluded_beams`), while a `COHERENT` one refuses: a field left out could cancel the others.

**Validated** (`tests/test_focal_field.py`; tolerances set before the first run and stated there): an independent Richards-Wolf table at 852 nm (x-polarized Gaussian pupil, `w` = 10 mm,
`F` = 20 mm; NA 0.15, 0.30, 0.40, 0.50, 0.65) reproduced to 1e-4 in the 1/e² radii (NA 0.65: x 0.6984686 µm against 0.698480, y 0.6358356 against 0.635844); the Airy pattern at NA 0.05 (FWHM
`0.5145 λ / NA`, encircled energy `1 - J0² - J1²` at 50 % and 80 %, peak `P π NA² / λ²`, 3e-3); the Gaussian waist and its peak at low NA (1e-3); the exchange of the x and y widths between x and
y input and equal widths for circular and unpolarized input (3e-9); `E_y` odd in x and y and `E_z` odd in x for x input; the longitudinal share against a one-dimensional quadrature of the
sphere (3e-9); the clipped-Gaussian pupil power (2e-9) and the energy balance (1e-3, Parseval 1e-12); a tilted input at `F l` with the on-axis widths (2e-3) and incoherent foci carrying their
beams' powers (1e-3); a window's `(1 - R)²` transmission at low NA (1e-3), its focus between the paraxial `t (1 - 1/n)` and the marginal-ray shift, and its Strehl ratio at NA 0.3 against the
Maréchal estimate of the plate's residual phase (0.01); a Zernike spherical and astigmatic term against Maréchal (0.01). Not validated: any real objective, against a prescription or a
measured point spread function.

## P10h. Imaging analysis: point spread function and MTF from the trace

A post-process (`os_optics/results/imaging.py`, `os_optics/results/imaging_trace.py`, the kernel of `os_optics/diffraction/debye.py`) that changes no run; its reply is an
analysis document (`AnalysisResultV1`, analysis `IMAGING`).

**Rays.** Each field is a spectral line of a beam source of the run. Its light is traced again from the run's own Engine request with the source's pupil rays replaced by a
square grid over a launch disc (`SourceG.pupil_layout`: the tracer's pupil launch is the only place it is read; the untraced-light probes and interference are not run). The disc
starts at 2 beam radii and grows by 1.5 until the system's apertures cut rays (up to the request's `max_launch_radius_beam_radii`, default 16), then is fitted to the light that passes: the vignetted pupil
(`rays.pupil` `APERTURE`). Where no aperture cuts the disc within that reach the system's pupil is not bounded: the field is analysed on the disc of 2 beam radii, exactly as a request that
states `launch_radius_beam_radii` 2 (`rays.pupil` `LAUNCH_DISC`, `rays.pupil_search_radii`), and every number of the field says it stands on that stand-in pupil -- state `UNKNOWN` with reason
`PUPIL_NOT_BOUNDED` beside the value and a `value_basis` of kind `ESTIMATE` with the assumption `UNIFORM_PUPIL` (2 beam radii): it is the image of a uniformly lit disc of that size, not of the
system's pupil and not of the beam's own Gaussian profile. A stated `launch_radius_beam_radii` is the caller's pupil and its numbers are `KNOWN`. The pupil is
lit uniformly (a point object; at infinity for a collimated source). The same is done with twice the rays per axis: the values reported are that finer grid's, the coarser one is each quantity's
`convergence`. A request with `convergence` false traces each field once, on `launch_grid` rays: the values are that grid's (each value's convergence `NONE`: not checked),
and on that coarser pupil a diffraction quantity can leave its domain (`PUPIL_UNDERSAMPLED`) where the default computes it. One with `diffraction` false computes no point
spread function (the Strehl ratio, the MTF curves and the encircled-energy radii are `NOT_COMPUTED`, reason `NOT_REQUESTED`); a field without a wavefront makes it partial,
as it makes the full analysis. The grid is
deterministic: no random sampling, no seed. The rays read are those of the run's strongest path of that line at the plane, at its last crossing unless `crossing_index` says otherwise.

**Geometry and wavefront.** The image-space axis `w` is the rays' mean direction; the reference point is their centroid on the image plane (or the source axis's ray, `CHIEF`); the
spot, the point spread function and the MTF lie on the plane normal to `w` through it, with the tangential axis the projection of the image plane's normal (the plane's `u` axis
on axis) and the sagittal axis across it. The wavefront is each ray's optical path from the source's phase front (`results.rays`' launch term) less `n` times the projection of
its offset from the reference point on its direction, `W_j = OPL_j - n d_j . (p_j - r)`: the wavefront on the reference sphere at infinity about the reference point (direction
space), the phase the Debye integral below sums; piston removed, its RMS and peak-to-valley are in waves. The same RMS on the finite sphere through the last surface before the plane
is reported beside it (`rays.opd_rms_waves_on_last_surface_sphere`, null where a ray's line passes farther from the reference point than that radius): the two agree where the
radius is large against the spot and the focus offset (a plane defocused by about the last gap reads about twice the defocus wavefront on that sphere, and a spot wider than
the last gap misses it). Validated against the defocus wavefront `dz NA^2 / (4 sqrt 3)` RMS
(2 %) with the last surface 10 um and 0.5 mm before the plane.

**Point spread function.** `U(a, b) = Σ_j A_j exp(i k0 W_j) exp(i k (s_t,j a + s_s,j b))` over the rays, `A_j = sqrt(p_j ΔΩ_j) n / λ`, `p_j` the ray's share of the light: its irradiance where it crosses the plane relative to its launch (`|J_j|^2` of the field the trace carries -- a sample's amplitude, the interfaces' and coatings' s and p transmissions -- over that of the source's Jones vector), normalized over the rays; a partially polarized or unpolarized source's two components are mutually incoherent: each has its own `A_j` (its own irradiance, times its share of the line, normalized over both), its field is propagated to the image apart, and the point spread function is the sum of their irradiances, from which the Strehl ratio, the MTF and the encircled energy follow (the geometric spot, its centroid and the wavefront error weigh the rays equally),
`ΔΩ_j` the solid angle its tube fills in image space (the Jacobian of the launch grid's map to direction sines, from its grid neighbours, over `cos θ`), `W_j` its optical path
difference. The sum is separable in `a` and `b`: two matrix products for a map. It is normalized to unit integral over the plane (Parseval over the rays: `Σ p_j / cos θ_j`). Up to an
image-space sine of 0.2 the field is scalar; above it each ray carries the vector field of an aplanatic lens's polarization map (`e_p`, `e_s` about `w`), for unpolarized light (two
orthogonal pupil polarizations added in irradiance). From the map: the Strehl ratio (the peak, found by golden-section searches, over that of the same rays without their phase at the
reference point), the MTF (the Fourier transform of the line-spread functions along the tangential and sagittal axes over all the light -- the light beyond the map is missing only below `1 / (2 half width)`; values beyond the map's Nyquist
frequency are null) and the diffraction-limited MTF of the same rays without their phase, and the radii holding 50 % and 80 % of the light (Gauss-Legendre annuli over the map's cubic
spline) with the diffraction-limited 80 % radius. The geometric MTF is `|Σ_j p_j exp(-i 2π f (r_j - r_c))|` of the ray spot.

**Validity.** The diffraction quantities need the rays' optical paths (`RAY_OPL_UNDEFINED` after an ideal or tabulated thin lens, a Faraday rotator or a grating), an image (light
converging at the plane after a surface: `EXIT_PUPIL_UNDEFINED`), neighbouring rays at most π apart in phase and a map half width at most half the period `λ / (n Δs)` the ray
spacing `Δs` in direction sines gives the summed field (`PUPIL_UNDERSAMPLED`), and a map holding at least 95 % of the light (`PSF_WINDOW_TOO_SMALL`). The geometric MTF needs a
wavefront error of at least one wave RMS (`GEOMETRIC_REGIME_NOT_MET`): below it diffraction sets the MTF; where no wavefront is defined (`RAY_OPL_UNDEFINED`) whether it holds is
not known (`GEOMETRIC_REGIME_UNKNOWN`, `UNKNOWN`). In both cases the geometric curve of the traced spot is stated beside that state, with a `value_basis` that says what it is: an
`ESTIMATE` holding the regime check (`opd_rms_waves`, minimum 1, the observed value, `within` false or null); it lies above the diffraction MTF below one wave and decides no
comparison. Only where a source's two polarization components do not form one spot is there no curve. Outside its domain any other quantity is `OUT_OF_MODEL_SCOPE` with that reason;
the geometric spot, the numerical aperture and (where defined) the wavefront stay. Each field's model bases carry their checks (`models`).

**Validated** (`tests/test_imaging.py`; tolerances set before the first run): an aberration-free F/8 plano-hyperbolic singlet (conic `-n²`) against the Airy pattern -- numerical aperture
1/16 (2e-3), Strehl 1 (1e-3), the MTF `(2/π)(acos ν - ν sqrt(1 - ν²))` (0.01), the 80 % radius from `1 - J0² - J1²` (1 %). Also checked:
the double-Gauss fixture (about F/3) on axis and at 5° (scalar; the MTF below the same pupil's diffraction limit; equal tangential and sagittal MTF on axis, different off
axis), the Cassegrain of `tests/test_paraxial_mirrors_compounds.py` (its third, focusing crossing) and an F/1.2 plano-hyperbolic singlet (vector kernel, Strehl 1 within 0.01). A thin ideal
lens refuses the diffraction quantities and keeps the spot. Not validated: any design against an external lens-design program's MTF, or against a measurement.

**Not modelled.** Polychromatic point spread functions (one line per field), partial coherence, image simulation beyond the point spread function, a source beam whose
chief ray an aperture cuts (the tracer ends it there: such a field is `NO_ARRIVAL`; a beam that fills the stop around its chief ray arrives, P10 truncation), extended emitters as fields.

## P10i. Ghost paths: interface reflections up to a bounce order

A post-process (`os_optics/results/ghosts.py`, `os_optics/results/ghost_trace.py`) that changes no run; its reply is an analysis document (`AnalysisResultV1`, analysis
`GHOST_PATHS`; request `GhostRequestV1`, `os-optics results ghosts RUN --monitor IMAGE`).

**Trace.** The run's own Engine request is traced again by `GhostTracer`, the run's `Tracer` with four hooks (the refraction, the spawn of a branch, a power flow, an untraced
branch). At every refracting interface whose reflection the design does not follow (`trace_reflection` off: a lens or window face), the reflected part becomes a traced branch while
the branch meeting it has made fewer than `max_bounce_order` such reflections (default 2, at most 4). Its amplitude is the bare Fresnel `r_s`, `r_p` of the interface, or the
coating's stated reflectance (and phases) when the face is coated, at the chief ray's angle of incidence, applied to the Jones vector of every pupil ray as for the transmitted light;
its Gaussian beam is reflected by the face's local curvature (P4). A reflection the design follows (a splitter, a mirror, total internal reflection) is the design's light, not a
ghost bounce. Everything else is the run's tracer: the primary light is traced exactly as in the run, a ghost without a valid Gaussian state whose pupil rays an aperture cuts goes on
with the geometric pupil-sample power (P10), and the untraced-light probes and interference are not run. The trace is done with the sources' own pupil sampling and again with twice
the rings and twice the points per ring; the values are the finer trace's, the coarser one is each quantity's `convergence`. A request with `convergence` false does the
first trace only (each value's convergence `NONE`, its sampling resolution kept). The ghost trace has its own budget (`max_paths` 4096,
`max_trace_steps` 16384 by default; the run's `max_interactions_per_path`), strongest branch first.

**Per ghost path** (a path with at least one ghost bounce) and line: the bounce surfaces in order, the bounce order and its direction (`FORWARD`: an even order, on toward the
image side; `BACKWARD`: an odd order, back toward the source), its end, and the power its last bounce reflected over the line's input. On each crossing of the image plane:

* `relative_power = P_ghost / P_primary`, the primary being every path of the line without a ghost bounce on the same plane: a ratio of powers (incoherent: a ghost does not
  interfere with the primary light or with another ghost); `power_relative_to_line` over the line's input. No absolute radiometry is stated.
* `focus_offset_mm`: the least-squares point closest to the ghost's pupil rays (`results.rays._ray_focus`, equal weights), as a signed distance from the plane along its normal
  oriented with the primary light (positive beyond the plane). Parallel rays have no focus: `NOT_APPLICABLE` (`RAY_FOCUS_UNDEFINED`); fewer than 8 pupil rays: `UNKNOWN`
  (`TOO_FEW_RAYS`).
* `waist_offset_x_mm`, `waist_offset_y_mm`: the waists of the ghost's paraxial Gaussian beam along its chief ray, the same way; without a valid Gaussian state `OUT_OF_MODEL_SCOPE`
  with the state's reason code (`GAUSSIAN_SURFACE_TOO_STRONG`, `GAUSSIAN_HIGH_NA`, ...), while the ray focus stays.
* `spot_rms_radius_um`: the RMS radius of the pupil rays on the plane about their centroid.
* `irradiance_ratio`: the peak irradiance over the primary's, `2P / (π w_x w_y)` of both Gaussian beams where both are valid; else the estimate
  `relative_power (r_p² + d_p²) / (r_g² + d_g²)` from the RMS radii `r` and the diffraction radii `d = λ / (π NA)` of each bundle (NA the medium index times the largest sine of its
  rays about their mean direction). The estimate is not a bound.

**Completeness.** Every piece of a line's light the ghost trace does not follow is entered with its power where it leaves the traced beams: reflections beyond the bounce order
(`BOUNCE_LIMIT`), ghost reflections at or below `relative_power_floor` of the line's input (default 1e-9; `BELOW_FLOOR`), light the ghost budget leaves untraced, a whole source line
included (`PATH_BUDGET`), paths ended outside the model and source lines not launched (`OUT_OF_MODEL`), light a rough surface or a scattering medium scatters (`SURFACE_SCATTER`,
`BULK_SCATTER`: `SCATTERED`), light an element's response model does not deliver (`UNMODELLED_RESPONSE`), other untraced light (light beside an element, a coating's transmission:
`UNTRACED_<kind>`), and the enumerated ghost paths beyond `max_listed` per line (`NOT_LISTED`, at the power they were born with). A passive system never raises a beam's power, so
the sum (at most the line's input: the pieces are disjoint parts of its light) bounds the power every unlisted ghost brings to any plane (`completeness.unenumerated_power_upper_W`;
per line `unlisted_power_upper_relative_to_line` and `_to_primary`, `BOUNDED`, with the trace's own powers of the pieces as the nominal value). Each piece's bound undoes the
single-line approximations on its way. Past a geometric pupil-sample cut: the light past an aperture is at most the light that met it. At each ghost surface the trace applies the
chief ray's reflectance and transmittance to the whole beam, and the beam's mean reflectance exceeds the chief ray's where the reflectance is convex in the angle (1.1 to 2 times for
cones of NA 0.3 to 0.5 at 45 to 60 deg); the bound takes instead an upper bound on the beam's mean over its light. The paraxial Gaussian beam's position and direction are Gaussian
(its plane-wave spectrum has the far-field half angle `θ0` at every distance), so the light within `ρ` whitened radii of its centre meets a plane within `ρ θ0` of the chief ray's
angle of incidence, and a sphere of radius `R` at normals whose angle to the chief ray's line has its sine within `(ρ w + 2 ρ θ0 R) / R` of the chief ray's (`w` the beam radius).
Shell by shell (`ρ` = 0.25 to 6 in steps of 0.25) the bound adds the shell's share times the largest coefficient over those angles and both polarizations (a bare face's s reflectance
at the widest angle and 1 from the critical angle on; for the transmittance 1 less the smallest p reflectance, 0 where Brewster's angle is inside; a coating's largest reflectance and
transmittance over its table), and counts the light beyond whole. That bound is `PROVEN` under the stated model (every other interaction acts on the whole beam with its chief ray's
coefficients, as in the run). A beam without a valid Gaussian state, an annular beam or a sag surface has no stated distribution of angles: there the bound takes the mean, over its
live pupil rays (equal-power samples), of each ray's larger coefficient at its own angle -- a sampled value -- and every piece after such a face makes the line's interval `NOMINAL`
(not proven; `completeness.proven` false; each line's `completeness` array counts the ghost-surface meetings with sampled angles). It bounds power, not irradiance: a weak ghost may
focus. Light whose power the trace does not know (`POWER_NOT_KNOWN`, behind a grating without a response) is listed without a bound, and the bound is then not `proven`. The bound
covers every plane, so on an image plane it also counts light travelling back toward the source.

**Sampled powers.** A ghost or primary power on the geometric pupil-sample basis states its sampling resolution (`convergence.resolution`, in the quantity's units; a ratio's from
both of its powers) and its message says so; a ratio whose primary power is not resolved from zero is `UNKNOWN` (`SAMPLING_NOT_RESOLVED`).

**Execution.** `COMPLETED` when the enumeration ran to its end; `PARTIAL` (`BUDGET_EXHAUSTED`) when the ghost budget ran out. What the enumeration could not state is not an
execution state: a path that ended outside the model, scattered light, light an element's response model left undelivered and light of unknown power are listed in
`completeness.not_followed` and are in the bound; a field that is not a beam source, a primary that does not reach the plane and a power out of model scope are their
quantities' own states and reasons. The light beyond the bounce order and below the floor is what the request declares, and enters the bound.

**Validated** (`tests/test_ghosts.py`): an uncoated N-BK7 plate at normal incidence against Fresnel -- primary `T²`, the two-bounce ghost `T R R T` (relative `R²`, 1e-8), the back
reflections `R` and `T R T`, and the bound `T R³`, at least the power the exhaustive four-bounce enumeration of the same plate finds born beyond two bounces and within 1e-5 of
it (the reflectances over the beam's angular spectrum), covering the order-3 and order-4 light it reaches; a 45 deg plate met by a beam of far-field half angle 0.2 rad, whose
beam-averaged reflectance (an independent integral over its plane-wave spectrum, each plane wave with its own s and p shares) exceeds the chief ray's: the bound covers the average;
a plate of bulk-scattering glass (87 % of the line scattered) and a rough face: the scattered light enters the bound and is listed in `not_followed` with the reason `SCATTERED`; two windows with a stated 0.5 % coating -- the `N (N - 1) / 2 = 6` two-bounce ghosts at `R² T^(2 (i - j - 1))` (1e-9); a biconvex singlet's
two-surface ghost focus against the paraxial trace of the folded path (refract, reflect, reflect, refract; signed indices) within 0.02 mm of 36 mm, its Gaussian waists likewise; a cut
ghost budget (`PARTIAL`, `PATH_BUDGET`, its bound covering every ghost it did not list). The double-Gauss fixture (about F/3) on axis and at 5°: every two-bounce ghost listed with its
focus, order and relative power (the strongest about 0.3 % of the primary), the ghosts whose Gaussian state the steep inner faces invalidate keeping their ray focus. Not validated:
any design against an external ghost-focus program or a measurement.

**Not modelled.** The ghosts of scattered light (bounded, not enumerated), diffraction of a ghost at an aperture edge, coherence between a ghost and the primary light, polychromatic ghost images (one line at a time), the
ghosts of an extended emitter, absolute irradiance.

## P10i. Diffuse surface scatter and bulk scattering

Module `os_optics/optics/rays/scatter.py` (the events and the ledger split), `os_optics/optics/rays/scatter_trace.py` (the rays),
`os_optics/validation/scattered_light.py` (results).

**Surface scatter** (`SurfaceScatter` on a `RefractInteraction`, `MirrorInteraction` or `AbsorberInteraction`; a Project surface's
`scatter`, `SurfaceScatterV1`). Where a beam meets the surface on its active side, the share `f = fraction` of the power meeting it
leaves as incoherent, unpolarized light on `side` (`reflect`: back into the medium the light comes from; `transmit`: into the medium
behind), with the intensity law `I(θ) = P cos θ / π` (`LAMBERTIAN`) or `I(θ) = P (n + 1) cosⁿθ / (2π)` (`COSINE_POWER`, `n = exponent`)
about the surface normal on that side. The beam goes on with `1 - f` of its power (field scaled by `sqrt(1 - f)`; direction, Gaussian
state and polarization unchanged) and meets the surface's own interaction (Fresnel / coating, reflection, absorption). A reflector
scatters on its reflecting side, an absorber back only, a splitting surface not at all.

The scattered light of each beam is a secondary extended emitter: a point at the beam's chief-ray hit point, of power `f P`, sampled by
`sample_count` rays (default 1024; the emitter sampling of P8a with `cos θ^(n+1) = 1 - u` for the direction) and traced with the
same physics. Its rays belong to the beam's source line: their fates replace the event's `SURFACE_SCATTER` entry in that line's ledger,
so `input = Σ sinks` covers specular and scattered light together. Results: per plane one `sampled_light` entry per event
(`scatter_surface_id`, `parent_path_id`; collection fraction of the scattered power, its first-half estimate, the crossings' columns), its
counted power part of the plane's power; per event one `sampled_sources` entry (`shape` `POINT`, `scatter_side`, `scattered_fraction`,
`incident_power_W`, on-axis `intensity_W_per_sr`, `beam_radius_mm`). The run traces at most `max_scatter_origins` events (the
Project's `compute_budget`; 64 unless it states another), the strongest first.

**Bulk scattering** (`scattering_coefficient_per_mm` `μs` and `anisotropy_g` of a material). A traced beam's power decays as
`exp(-(α + μs) L)`; the loss splits exactly into `BULK_ABSORPTION` and `BULK_SCATTER` in the ratio `α : μs` (both remove power at the same
place along the path). The scattered light is not followed: `anisotropy_g` is validated and reported, not used.

**Accounting.** Scattered light the run does not trace -- bulk scattering, light a scattered or sampled ray scatters again, light a probe
trace scatters, events beyond the `max_scatter_origins` strongest, a calculation of the beams alone -- keeps its ledger entry (`SURFACE_SCATTER`,
`BULK_SCATTER`): power of unknown fate, which may reach any plane. Above the line's negligible level (`negligible_power_fraction` of
its input, summed with the line's other unknown-fate power except what a budget stopped) no plane of that source line reads a complete
power (`power_complete` false; a lit plane is `BOUNDED` from its traced value, a dark one `UNKNOWN`; an absorbing plane still states
`power_upper_bound_W`), and one `SCATTERED_LIGHT_NOT_FOLLOWED` notice (`NOT_APPLICABLE`) per surface and per medium states the power
(a surface's notice also the budget in force, `max_scatter_origins`, and how many of its events lie beyond it,
`origins_beyond_budget` with `power_beyond_budget_W`).
At or below that level the scattered power is left out as negligible: the planes keep their state, and their upper bounds include it. A plane no ray of a traced event is counted
on reads it as a sampled zero, as for an emitter.

**Model scope.** Geometric, incoherent, single scatter per beam: no diffraction, speckle or coherence of the scattered light, no
polarization of it. The scatter leaves from the chief-ray hit point: the footprint is not resolved (`beam_radius_mm` states it; the
description holds where it is small against the distances to what the light reaches). Measured BSDF tables, ABg / Harvey laws,
volume single scattering (Henyey-Greenstein emission along the path) and multiple bulk scattering are not modelled; a `bsdf` other than
`LAMBERTIAN` / `COSINE_POWER` is refused. The scattered rays are held to the published models of the elements they meet like an
extended emitter's (wavelength, coating and angle ranges, data ranges where they stop; not the edge and body checks, which a diffuse law
meets by design): a model-scope finding lists the event's line on the planes its rays cross (`affected_sampled_lines`), and so does a
finding on a beam the light scatters out of behind the point of the finding (`RESULTS_AND_EXPORT.md`).

**Validated** (`tests/test_surface_and_bulk_scatter.py`, 4096 rays, absolute 0.004 of the scattered power, set before the first run):
the share of a point scatterer's power on a coaxial disc of radius `a` at distance `h`, `1 - (h / sqrt(h² + a²))^(n+1)` (Lambertian:
`a² / (a² + h²)`), for `n` = 1, 3 and 0.5; the ledger closure over specular and scattered light; a rough mirror's specular power
`R (1 - f) P`; a plate of index 1.5 at normal incidence: `(1 - f)(1 - R)² exp(-(α + μs) L) P` (relative 1e-12) and the split of its
bulk loss.

## P10j. Thin complex-transmission samples (`ComplexTransmissionInteraction`)

Module `os_optics/optics/complex_transmission.py`; the tracer's handler `_do_complex_transmission`. A thin plane, the same medium on both
sides, with `t(u, v) = A exp(iφ)`: `A` = `amplitude` x the sampled map's amplitude; `φ` = `phase_rad` + linear ramps + steps + Zernike
terms (`NOLL` / `RMS_ORTHONORMAL`, `FRINGE` / `PEAK_UNIT`, evaluated as a Zernike sag is) + the map's phase (bilinear between nodes, the
unwrapped phase). In a Project: an `APERTURE_STOP`'s `sample` (`ComplexTransmissionSampleV1`): the opening holds the sample, the plate
blocks around it.

**Rays.** Each ray gains the optical path `φ / k0` and the tangential wavevector `∇φ`: `n d_t' = n d_t + λ ∇φ / (2π)` (the local grating
equation); its polarization is kept (basis transported). A ray with `|d_t'| ≥ 1` is evanescent: retired; the chief ray's path ends
(`TRANSMISSION_MAP_EVANESCENT`, `OUT_OF_MODEL_SCOPE`).

**Part of the beam evanescent.** Where the chief ray propagates but some live pupil rays, or nodes of the beam's Gaussian footprint
within the clear aperture (the footprint rule below, each node with the Gaussian's local ray direction: the chief ray's tilted by the
offset over the wavefront's radius of curvature), are turned beyond grazing, the beam behind the element is not a Gaussian beam
(`TRANSMISSION_MAP_EVANESCENT`; its Gaussian quantities are unavailable) and its power goes on the geometric pupil-sample basis (as behind an aperture that cuts the pupil rays of such a beam):
of the `N` live equal-power pupil samples, the turned ones' `Σ A² / N` leaves the beam as evanescent light of unknown fate
(`TERMINATED_OUT_OF_MODEL`: the planes of that line read a power that is not complete), the others' `Σ A² / N` goes on, and the
sampling resolution grows by the samples whose cell may straddle the evanescent edge and those of the outermost ring (their cells reach
the Gaussian tail, which the footprint may show turned though no sample is). With no sample left the path ends out of model scope. For
a Zernike defocus whose gradient reaches `k0` at the radius `r_c` of a beam of radius `w`, the thin-element share that propagates,
`1 - exp(-2 r_c² / w²)`, lies within the reported power's sampling resolution (`tests/test_complex_transmission_sample.py`).
Where no pupil ray is turned and only the far tail of the footprint is, the turned share is bounded by the Gaussian power beyond the
outermost footprint ring inside the innermost turned node, `exp(-2 ρ²)` in the beam's whitened frame (as far as the footprint's
rings and azimuths resolve the turned region). When that share of the incident
power is at most the line's negligible level (`negligible_power_fraction` × the line's input), it is left out as negligible light
(it bounds the planes' power as the other light left out does) and the beam keeps its Gaussian state and power; above it, the
pupil-sample basis applies. For the defocus above, `r_c` = 4 w (share 1.3e-14) keeps the beam's 1 W, `r_c` = 3 w (1.5e-8) does not.

**Power.** The beam's power is multiplied by the Gaussian-weighted mean of `A²` over its footprint: in the beam's whitened frame a
Gauss-Legendre rule in the radius (3 panels to `aperture_quad_extent_sigmas` beam radii) times a uniform azimuth rule, each node projected
along the chief ray onto the plane; the same mean from half the nodes per axis must agree within 1e-9, else the power behind is out of
model scope (`TRANSMISSION_MAP_VARIES_ACROSS_BEAM`). Without a Gaussian footprint the chief ray's `A²` is used and the power is out of
scope where `A²` varies across the live pupil rays. Pupil rays carry their own `A`.

**Gaussian beam.** A phase that is a plane across the footprint (uniform phase, linear ramp) steers the beam (`apply_grating`: the
anamorphic map of a constant tangential wavevector) and keeps it Gaussian. When the phase less its best-fitting plane varies by more
than 0.01 rad RMS over the footprint, the beam behind the element is not a Gaussian beam: its Gaussian quantities are unavailable
(`TRANSMISSION_MAP_PHASE_STRUCTURE`) and it goes on as rays with its power.

**Diffraction.** The element turns rays by the local phase gradient; diffraction by structure finer than the beam is not computed in the
run. The imaging analysis (P10h) and the wavefront read the rays' optical paths, which carry `φ`; its sampling checks (`PUPIL_UNDERSAMPLED`)
judge the map's structure against the ray spacing. Polarization-resolved maps and birefringent samples are not modelled; the linear
optical network refuses a path through a sample (`PHASE_UNDEFINED_SAMPLE`); paraxial analysis refuses a stop whose sample has a phase.

**Validated** (`tests/test_complex_transmission_sample.py`): uniform `a`, `φ`: power `a² P` (1e-12) and optical path `φ λ / (2π)` (1e-12 mm);
a linear phase at normal and 20° incidence: `sin θ' = sin θ + λ g / (2π)` (1e-12), sampled and analytic ramps equal; an amplitude `a0 + c u`:
`a0² + c² w² / (4 cos²θ)` at 0° and 30° (relative 1e-9); a Zernike defocus as the thin-lens phase of focal length `k0 R² / (4 sqrt(3) c)`: its
rays meet there (spot RMS below 1e-6 beam radii) and the beam is no longer Gaussian; a phase step across the beam; the recorded
homogeneous plate (index 1.6, 0.01 /mm, 5 mm at 632.8 nm: 0.852612768013 mW) with a sample behind it.

## P10k. Triangle-facet surfaces (`MeshSurface`, Project `FACET_SOLID`)

A surface of flat triangles (`vertices_mm`, `triangles` counter-clockwise from the positive side), with `approximation` (`FACETED_EXACT`, or
`TESSELLATED_FROM_CAD` with its `max_chord_deviation_mm` and `source`), `closed` (the facets bound a solid: `medium_positive` outside,
`medium_negative` inside), an `interaction` (refract, mirror or absorber) and optional `facet_groups` (triangles with their own interaction:
a coated, mirrored or blackened face). The contract checks the mesh: indices in range, no degenerate facet, every edge on at most two facets
that run it in opposite directions (a manifold sheet of one orientation), every edge on exactly two facets and a positive enclosed volume
when closed, at most 20000 vertices and 40000 triangles, each triangle in at most one group. A Project declares a solid of facets as the
model kind `FACET_SOLID` (closed, a declared medium inside, air outside, groups `TRANSMITTED` / `REFLECTED` / `ABSORBED` with optional
coatings); it compiles to one closed `MeshSurface` named `facets`. A `MeshSurface` may state a `placement` (a rigid pose of its vertices in
the owning frame); a `FACET_SOLID` member of a `COMPOUND` unit keeps its vertices as written and takes its member pose as this placement
(composed with one it has), so it has the faces it has placed directly.

Model (`optics/surfaces/facets.py`, `scene/lower.py`). The triangles are grouped into planar patches: edge-connected triangles of one
facet group whose corners lie within the coplanarity tolerance of the patch's plane. The tolerance is read from the coordinates as given
(`coplanar_tolerance`): their rounding `q` is half a unit of the last decimal any coordinate uses, or, where every coordinate is a
single-precision number (float32, as binary STL stores them), half the float32 spacing at the largest coordinate when that is coarser;
rounding places the corners of one flat face within `sqrt(3) q` of its plane, and the tolerance is `COPLANAR_QUANTA * q` = `4 sqrt(3) q`,
at least `COPLANAR_FLOOR_MM` = 1e-10 mm (coordinates of full double precision) and at most `COPLANAR_MAX_FRACTION` = 1e-4 of the diagonal of
the mesh's bounding box. A patch whose corners all lie within 1e-10 mm of its first triangle's plane is traced as that plane (exact data);
otherwise its plane is fitted to its corners (normal: the sum of its triangles' area vectors; offset: midway between the farthest corners on
either side) and `deviation_mm`, the farthest any corner lies from it (at most the tolerance), bounds the geometric error of tracing the
face as that plane. The faces are found from the vertices as written (before any `placement` or component pose moves them: a
rigid motion would turn rounded coordinates into full-precision ones and hide their rounding). A face fitted to facets whose corners lie
farther from it than `FITTED_FACE_NOTICE_FRACTION` = 1e-5 of the mesh's extent is stated in a notice (`FACET_FACE_FITTED`,
`NOT_APPLICABLE`: the faces, at most 12 named and all counted, the largest deviation and the part's size): a bend of the part that
small is traced flat. A Project run states the tolerance and the largest deviation per `FACET_SOLID` (run summary `element_models`,
`facets.coplanar_tolerance_mm`, `facets.max_face_deviation_mm`). Where fitted planes meet, each outline vertex is moved to the point
nearest it on the planes of every patch whose outline holds it (least squares; three planes meet in one point, two along a line, so
neighbouring outlines share their corners exactly and meet with no gap a ray could pass through; where four or more fitted planes meet at a
vertex, within their deviations). Each patch is traced as an exact plane (normal from the winding, toward the positive side) whose clear
aperture is the patch's outline, a simple counter-clockwise polygon of mesh vertices (straight corners dropped); a patch whose outline is not
one simple polygon of at most `MAX_FACET_OUTLINE_CORNERS` = 1024 corners (a face around a hole, a face touching itself at a corner) is
traced triangle by triangle. (The simplicity check compares only edges whose extents overlap: about 10 ms for a 1024-corner outline.) Patch ids are
`<component>/<surface>.t<k>`, `k` the patch's first triangle. Every plane interaction applies unchanged: Snell refraction and total internal
reflection with Fresnel (or coating) power and Jones phases, mirrors, absorbers, the Gaussian beam through a plane, the power ledger. A ray
meeting an edge or a corner exactly meets one of the faces that share it (the nearest crossing; equal distances: the lower scene index);
the edge is inclusive within the aperture tolerance on both faces, so no ray leaks through the solid there.

Facet edges. A beam whose footprint on a patch reaches beyond the patch's outline is divided by the edge: the Gaussian share beyond the
outline (with a valid Gaussian state; the geometric share of its pupil samples that meet the surface's other patches without one) falls on
the neighbouring facets (or, beyond an outer edge, past the part) and is not followed by this beam: it is booked `TERMINATED_OUT_OF_MODEL`
(detail `facet edge`, an unknown fate: it may reach any plane) and stated by `FACET_EDGE_DIVIDES_BEAM` (`WARNING`; `OUT_OF_MODEL_SCOPE`
with the end of the Gaussian state beyond `gaussian_clip_scope_fraction` = 0.1). The beam goes on with the rest.

Tessellated surfaces. Facets approximating a curved surface (`TESSELLATED_FROM_CAD`) are traced as the facets: rays meet and leave the
flat facets, not the curved surface, which lies within `max_chord_deviation_mm` of them; the facet normals are not the surface normals,
and the direction error this makes is not bounded by the chord deviation alone. Flat facets carry none of the surface's curvature, so a
beam's Gaussian state ends at the first one (`GAUSSIAN_UNDEFINED_AFTER_FACETS`, with the chord deviation) and the first-order ray-transfer
matrix is not reported behind it; rays, powers and polarization are traced on.

Domain. Refused as out of the model's scope (status `OUT_OF_MODEL_SCOPE`, termination `OUT_OF_MODEL_SCOPE`, issue
`FACET_SURFACE_NOT_TRACED` naming the surface, its approximation and the reason): an open sheet (`closed` false) that refracts between two
different media (a solid is bounded by closed facets), a uniaxial crystal on a side of the facets, and more than
`max_mesh_patches` planar patches on one surface (the Project's `compute_budget`; 1024 unless it states another: every patch is a
plane each trace step tests; the refusal names the setting and the patch count). Open sheets between the same
medium that reflect or absorb (a hollow roof mirror, a faceted beam dump) are traced. A closed surface must bound solids whose sides hold
(`optics/surfaces/facet_topology.py`, checked when lowering; refused with `FACET_SURFACE_NOT_TRACED` stating the reason otherwise): its
shells (facets connected through their edges) do not cross, overlap or touch along more than a point (facets sharing an edge are
neighbours, facets sharing one corner meet only there; facets of one planar face tile its outline and are not compared), the facets
around every corner form one fan, and each shell faces outward when an even number of other shells contain it (a solid; a solid inside a
cavity) and inward when an odd number do (a cavity), by the parity of a ray's crossings. Several separate solids, hollow solids and solids
in cavities are traced; two parts in contact are separate components. Distances below 1e-9 of the surface's extent count as contact.
Diffraction at facet edges is not modelled.

Return-light analysis. Inside a glass solid every pair of faces may return light left out of the trace by total internal reflection
(gain bound 1), and each pair has the solid's other refracting faces between it: the pair screen's work grows with the faces per pair
(`validation/return_work.py`: `PAIR_SCREEN_WORK` per pair plus `pair_cone_work`, the cones through the faces it considers, and
`momentum_work` per class of parallel faces). Within `max_return_analysis_work` (4,000,000 units for the whole run, about 15 s on a desktop
CPU) the pairs are named in `EXPOSURE_BOUND_WITHHELD` (at most `NAMED_RETURNS` = 12 named, all counted: `return_pair_count`); beyond it the
analysis stops and the bounds that read it are withheld (`RETURN_ANALYSIS_BUDGET`). Measured, one beam along a glass rod (native, one
thread): 50 faces 4.0 s (all 1225 pairs screened), 191 faces 14.9 s, 284 faces 17.9 s (the work limit reached in the pair screen).

**Validated** (`tests/test_facet_tracing.py`, `tests/test_facet_solid_project.py`): an equilateral prism (n 1.5) at minimum deviation
(`2 asin(n sin(A/2)) - A`, the ray inside parallel to the base) and at two other incidences: exit direction by vector Snell at both faces
(1e-12) and the scalar deviation `theta1 + theta2' - A` (1e-12), power the s Fresnel transmittance at both faces (1e-12); a solid corner
cube: exit direction `-input` (1e-12), the three back faces met once each by total internal reflection, the transverse offset from the axis
through the apex inverted (1e-11 mm), power `(4n/(n+1)^2)^2` (1e-12; through a Project with N-BK7 from its Sellmeier formula, 1e-9); a
silvered penta prism: exit direction the 90-degree rotation of the input about the mirrors' line for in-plane and out-of-plane tilts (1e-12);
a 12-facet block equal to the window of two analytic planes (position 1e-12 mm, direction 1e-15, power, optical path and Jones vector
1e-14, Gaussian radii and waist distances 1e-12); a beam on a prism's ridge at 1.2, 0.5 and 0 beam radii: the share beyond the edge
`erfc(sqrt(2) d / w) / 2` (relative 1e-6); a 48-sided tessellated rod: exit direction by Snell at the two facets found by an independent 2D
intersection (1e-12), no Gaussian state behind; the open-sheet and patch-count refusals; the power ledger closed in every case.
`tests/test_facet_cad_meshes.py`: a 10 mm cube turned 17 deg about (1, 2, 3) with its corners rounded to 12, 9, 8, 6, 5, 4 and 3 decimals,
stored as float32 or kept: six faces of four corners, every corner within the face's deviation of its plane and within `2 sqrt(3) q`,
neighbouring faces sharing their corners (1e-12 mm fitted, 2e-10 mm exact); a beam through two faces transmits `(4n/(n+1)^2)^2` (1e-12)
with its Gaussian state, no facet edge; rays through every corner and edge midpoint into the solid refract exactly twice; a 96-sided rod
rounded to 6 decimals keeps its 96 side faces; round caps of 64, 65, 96 and 360 corners are one face each and pass a beam whole; a
beam without a Gaussian state on a prism's ridge books the pupil share `cut / live` beyond the edge, mirrored offsets the mirrored face;
a withheld-bound notice of a glass cube names 12 of its 15 pairs and counts them. `tests/test_facet_solid_project.py`: a high reflector
(R 0.99) on a penta prism's mirrored faces gives `T0^2 R^2`, an anti-reflection coating on the other faces `0.995^2` (1e-12); the
section of a facet solid fills its medium inside the closed outlines (areas of the cut, 1e-6; a cavity keeps outlines only).
`tests/test_facet_solid_placement.py`: the turned cube rounded to 8, 6, 4 decimals and float32 as the one member of a `COMPOUND` unit (at
the origin, and turned and moved), a cell of an arrayed member and a coated penta prism (rounded, turned) trace as the part placed directly
by the composed pose: power (1e-12), direction (1e-12), position (1e-9 mm), optical path (1e-12), field vector (1e-11), Gaussian `q`
(1e-9), the faces met, the notices and the run summary's facet row equal; the coating's angle validity checked on every face crossed; a
placement composed of two equals the product of their rigid motions. `tests/test_facet_topology.py`: two separate cubes, a hollow cube
and a cube in the cavity of a hollow cube transmit `T^4`, `T^4`, `T^6` (1e-12); nested shells facing the same way, a lone inward shell,
overlapping and touching cubes, a corner pushed through a face and a corner shared by two cubes are refused with their reasons.
`tests/test_facet_cad_meshes.py`: a 7e-4 mm bend of a face written to 3 decimals is traced flat with a `FACET_FACE_FITTED` notice stating
its deviation; written to 6 decimals it is two faces and no notice.

## P10l. Second-harmonic generation (`UniaxialMaterial.nonlinear`, Project `NONLINEAR_CRYSTAL`)

A `NONLINEAR_CRYSTAL` is a rectangular slab of a declared uniaxial crystal (entrance face `z = 0`, exit face `z = length_mm`,
optic axis in `optic_axes`). The run traces the slab as any crystal (P5: refraction, ordinary and extraordinary beams,
walk-off, Fresnel or coating power at each face). Its Engine crystal material carries the interaction
(`SecondHarmonicGeneration`) and, when stated, its temperature (`CrystalTemperature`). Light of a traced Gaussian beam
entering the crystal through a face generates the second harmonic (`optics/nonlinear/shg.py`, `Tracer._second_harmonic`).

### Governing equations

Fields `E = A u exp(i(kz - wt)) + c.c.`, intensity `I = 2 n eps0 c |A|^2`, nonlinear polarization at `2w`:
`2 eps0 d_eff E_a E_b` (both photons in one wave: `OO`, `EE`) or `4 eps0 d_eff E_a E_b` (one in each: `OE`, type II).
Plane wave, undepleted pump, slowly varying envelope:

```
I_2 = 2 w^2 d_eff^2 L^2 I_a I_b f / (n_a n_b n_2 eps0 c^3) sinc^2(dk L / 2),   f = 1 (I_a = I_b = I_1), 4 (type II)
dk  = k_2 - k_a - k_b - 2 pi m / Lambda
```

* `dk` is the difference of the traced waves' wave-vector components along the face normal (the harmonic wave keeps the
  fundamental's tangential wave vector), each wave's index along its own wave normal at the crystal's temperature
  (ordinary `n_o`; extraordinary `n(theta)`, `theta` to the optic axis), minus the quasi-phase-matching grating's wave number
  (period `Lambda (1 + alpha (T - T_ref))`, odd order `m`, coefficient times `2 / (m pi)`). Phase matching is not assumed:
  angle, temperature and wavelength enter through the traced directions and the indices.
* `d_eff`: stated, or the contracted tensor `d_il` contracted with the traced waves' unit electric fields (the extraordinary
  field perpendicular to its energy flow), in the crystal frame X = `crystal_x_axis`, Z = optic axis. With a
  quasi-phase-matching grating the stated coefficient is the unpoled crystal's (for example `d33` of LiNbO3 for `EE -> E`,
  about 25 pm/V), not the reduced `2 d33 / pi`: the run applies `2 / (m pi)` itself, and a reduced value stated there is
  reduced twice.
* Crystals: uniaxial only. Biaxial crystals (KTP, LBO, BiBO, ...) and cubic chi(2) crystals (GaAs, ...) are not modelled. No
  crystal data are built in: the index laws, `d` and the thermo-optic data are stated with the Project (`nonlinear_basis`
  says where they come from).
* Gaussian beam (TEM00, waist radii `w0x`, `w0y` on its two axes, Rayleigh ranges in the crystal): every slice of the crystal
  radiates into the harmonic Gaussian mode of the same reduced q-parameter; the harmonic power is

```
P_2 = K f P_a P_b L^2 H / (pi w0x w0y),   K = 2 w^2 d_eff^2 / (n_a n_b n_2 eps0 c^3)
H   = (1 / L^2) Re int int F(u) F*(v) exp(-sum_j s_j^2 (u - v)^2 / w0_j^2) du dv
F   = exp(-i dk z) G(z) prod_j (1 + i tau_j(z))^(-1/2),   tau_j = (z - z_waist,j) / z_R,j
G   = exp(-sum_j p_j^2 z^2 / (2 w0_j^2 (1 + i tau_j)))   (type II: the two fundamental waves walk off each other by p)
```

  In an absorbing crystal (the material's attenuation `alpha`, absorption plus bulk scattering, the same at both
  wavelengths) `F` carries `exp(-a z - b (L - z))`: the two fundamental fields decay up to `z` (`a = (alpha_a + alpha_b) / 2`
  per unit path, along each wave's ray) and the harmonic generated at `z` decays from there to the exit (`b = alpha_2 / 2`).
  `P_2` is then the harmonic power at the exit face (inside the crystal); without walk-off, focusing or mismatch it is
  `P_2(alpha = 0) exp(-alpha L) ((1 - exp(-x)) / x)^2`, `x = alpha L / 2`.

  `s` is the walk-off of the harmonic's energy flow from the mean of the fundamental waves' (per unit path), on the beam's
  axes. `focusing` `near_field` freezes `tau` at the crystal's middle (the thin crystal: the beam radius is constant along
  it; then `H = (w0x w0y / (wx wy)) sinc^2(dk L / 2)` without walk-off); `boyd_kleinman` follows the beam's focusing through the
  crystal (Boyd and Kleinman, J. Appl. Phys. 39, 3597 (1968), with one beam axis per transverse direction). The integral is a
  composite Gauss-Legendre rule refined until two node counts agree to `nonlinear_quadrature_rel_tol`.
  The harmonic diffracts with the fundamental's confocal parameter (the same reduced q-parameter, per transverse axis the
  extraordinary wave's own index model of the fundamental): exact for birefringent phase matching (`n_2 = n_1`). Under
  quasi-phase matching the harmonic's own index differs by `eps = (n_2 - n_1) / n_1` (0.036 for LiNbO3 at 1064 nm): a full
  paraxial treatment in which the harmonic diffracts with its own index puts the tuning curve's peak at a `dk` shifted by
  about `eps xi` (`xi = L / b`; at `xi` 2.8 about 0.05 in `dk L / 2`, the maxima over `dk` agreeing to 7e-4) and a zero the
  focusing fills about `eps` higher (3.8 % at `eps` 0.036, waists 0.05-1 mm in a 10 mm PPLN).
* The harmonic beam: a new spectral line of the same source (index = the fundamental line's + the source's line count,
  `<source>#<index>`, half the wavelength), traced from the entrance face as the stated harmonic wave, with the
  fundamental's reduced q-parameter (waist `w0 / sqrt 2` at the same place, `M^2 = 1`) and its rays drawn in toward the chief
  ray by `1 / sqrt 2`; its rays start on the line of its power centroid (half the walk-off across the crystal from the
  driving beam's ray). Power: the converting fundamental waves give up `P_2` (type II: `P_2 / 2` each, one photon of each per
  harmonic photon); the fundamental's line books `NONLINEAR_CONVERSION`, the harmonic's line starts with it
  (`PowerBudgetLine.generated_from`). Both ledgers close. In an absorbing crystal the harmonic beam, traced from the entrance
  face and absorbed over its path like any beam, starts with the exit power divided by its transmittance through the crystal,
  so it arrives at the exit face with `P_2` (`harmonic_power_W`, `conversion_efficiency = P_2 / P_fundamental`). The ledger
  books that starting power -- the harmonic as if all generated at the entrance -- as the fundamental's conversion and the
  harmonic's absorption; the fundamental physically gives up `P_2` plus the harmonic absorbed after its generation along the
  crystal, less than that (the booked value is 1.07, 1.81 and 4.53 times it at `alpha L` = 0.1, 1 and 4, for a crystal whose
  harmonic and fundamental share one attenuation coefficient); the exit and arrival powers are exact and both ledgers
  close. That starting power is the harmonic line's input and the fundamental's `NONLINEAR_CONVERSION`.
* Coherence: the harmonic line's coherence length is half the fundamental's (`linewidth_nm` / `coherence_length_mm` of the
  source): the harmonic field `E^2` carries twice the fundamental's phase fluctuations, so `|gamma_2| = |gamma_1|^4` in the
  Gaussian-decay coherence model.
* Acceptance bandwidths (`NonlinearConversionResult`): full widths at half maximum of `sinc^2(dk L / 2)` about the traced beam,
  `5.566 / (L |d dk / dx|)` (or `2 sqrt(5.566 / (L |d^2 dk / dx^2|))` where narrower: non-critical phase matching) in the
  internal angle (the plane of the wave normal and the optic axis), the temperature (with stated thermo-optic coefficients)
  and the fundamental's vacuum wavelength; central differences of the collinear `dk`.
* Temperature: `n(lambda, T) = n(lambda, T_ref) + dn/dT(lambda) (T - T_ref)`, `dn/dT` per index, linear in wavelength between
  the stated rows; the same indices refract and walk off the beams.

### Assumptions / applicability

* Undepleted pump: the result overstates the harmonic by about `8/9 eta` (one wave, collimated Gaussian beam at phase
  matching; times `(P_a + P_b)^2 / (4 P_a P_b)` for two waves); above `nonlinear_depletion_rel_tol` the pass is
  `OUT_OF_MODEL_SCOPE` (`NONLINEAR_CONVERSION_OUT_OF_SCOPE`).
* Near field: the run computes the focused result too; where the two differ by more than `nonlinear_focusing_rel_tol` of
  the larger of the focused result and the focused result at `dk = 0` the pass is out of model scope (`boyd_kleinman` follows
  the focusing, for one fundamental wave). Near a zero of the tuning curve the near-field result vanishes while the
  focusing fills the zero: where the two agree to `nonlinear_focusing_rel_tol` of the focused result at `dk = 0` but not to
  that fraction of the focused result itself, the pass reports the focused result (`overlap_factor` and the powers; for a
  10 mm PPLN with a 0.05 mm waist at its centre, `L / b = 0.31`, at the first zero that is `6.0e-3` of the phase-matched
  value, where the near-field value is 0). The near-field result is reported only where it agrees with the focused
  result to that fraction of the value reported.
* The fundamental is one coherent, polarized, TEM00 line (`M^2 <= 1 + nonlinear_beam_quality_tol`) with a valid Gaussian
  state and a known power. A source of several lines or a partially polarized one is out of model scope: the harmonic of
  a mixture is not the sum of its parts' harmonics, and sum-frequency mixing between lines is not modelled.
* Continuous wave, one frequency: a pulsed source (`pulse`) is out of model scope (the conversion of a pulse train follows
  its peak power and pulse shape, and for short pulses its bandwidth and the group-velocity mismatch). A line of stated width
  (`linewidth_nm`, or `coherence_length_mm` read as a Gaussian line) is computed at its centre frequency while
  `sinc^2(dk L / 2)` averaged over its Gaussian spectrum stays within `nonlinear_linewidth_rel_tol` of the centre's value
  (relative to the larger of the two: near a zero of the tuning curve the line's average must agree with the centre's
  value, not merely be small against phase matching; where the near-field run reports the focused result that fills a zero,
  the centre's value is that result's share of the focused result at `dk = 0`, so a line whose average changes by a
  negligible part of it converts as one frequency); a wider line is out of model scope, the reason stating both widths. The
  line is read as one longitudinal mode with Gaussian phase noise (its harmonic's coherence length is half the
  fundamental's); a multimode or chaotic line of the same width has a harmonic of up to twice the single-frequency power and
  a coherence length of `1 / sqrt 2` of the fundamental's, which the model does not compute. For
  example a 2 mm PPLN doubler at 1064 nm (spectral acceptance 1.02 nm) holds up to about 0.2 nm.
* Cascades: light that already generated a harmonic which travels with it (a second crystal, a second pass through the same
  crystal), and a generated harmonic entering a crystal, are out of model scope there: the harmonics add coherently, and
  sum-frequency mixing with the fundamental and higher harmonics are not modelled.
* A fundamental whose power is out of model scope where it enters (`power_out_of_scope_at`: part of the beam met an element or
  an absorbing plane beside its chief ray, an attenuator or transmission map varying across it, a gradient-index rod's side, an
  earlier conversion out of scope) carries its chief ray's power, not the light's: the pass is out of model scope (the reason
  names those surfaces), with no harmonic power or efficiency. Example: a dump taking 30 % of a 1 W beam beside its chief ray
  before a 2 mm PPLN -- the harmonic of the light that enters is at most 0.49 of the chief ray's conversion (`P2 ~ P1^2`).
* Walk-off spreads the harmonic over a strip: past `nonlinear_harmonic_gaussian_rel_tol` of growth of its width the harmonic is
  traced without a Gaussian state (`HARMONIC_BEAM_NOT_GAUSSIAN`); its power is known.
* Where a pass is out of model scope the harmonic is still traced with the undepleted value, and the powers behind the
  crystal (harmonic and fundamental) are out of model scope (`ArrivalResult.power_out_of_scope_at`): a power target there is
  `OUT_OF_MODEL_SCOPE`, and so are the exposure of every surface that light reaches (`ExposureHit` `OUT_OF_MODEL_SCOPE`,
  `NONLINEAR_CONVERSION_OUT_OF_SCOPE`: no irradiance, fluence or pulse energy) and a `power_rating_utilization` target of its
  element -- for a pulsed source the harmonic follows the peak power, not the average power, so the continuous-wave value
  would understate its fluence by orders of magnitude.
* A slab only: two parallel plane faces. A crystal of another shape is traced as the linear crystal it also is; each pass into it
  reports its conversion `OUT_OF_MODEL_SCOPE` (`NONLINEAR_CONVERSION_OUT_OF_SCOPE`, the reason names the faces) and the powers
  behind its entrance face are out of model scope (`power_out_of_scope_at`): the fundamental's depletion is not computed. Light of a probe trace
  (untraced reflections followed to bound a plane) and an extended emitter's sampled rays pass the crystal unconverted; the
  harmonic they would generate is not bounded. Thermal lensing, photorefraction, an absorption that differs between the
  fundamental and the harmonic (the material states one coefficient), higher harmonics, cascaded processes, optical
  parametric processes and chi(3) effects are not modelled.

### Independent references (`tests/test_shg_reference.py`, `tests/test_second_harmonic.py`, `tests/test_nonlinear_crystal_project.py`)

* The coupled-wave equations integrated numerically (fourth-order Runge-Kutta): undepleted (`K I^2 L^2 sinc^2`, relative
  1e-9 for `dk L / 2` from 0 to 10) and depleted (`I tanh^2(Gamma L)`, 1e-8); integrated point by point over a collimated
  Gaussian beam (relative 1e-5) with the depletion overstatement `8/9 eta`.
* A split-step Fourier solution of the paraxial coupled-wave equation for focused beams with and without walk-off (relative 1e-5);
  Boyd and Kleinman's optimum `h = 1.068` at `xi = 2.84`, `sigma = 0.57`; the closed-form near-field walk-off factor.
* PPLN (congruent LiNbO3, Zelmon et al. 1997) quasi-phase matched 1064 -> 532 nm through the tracer: harmonic power, the
  sinc^2 tuning curve over the grating period, spectral and temperature acceptance from the Sellmeier formula; BBO
  (Tamosauskas et al. 2018) type I: phase-matching angle 22.83 deg (handbooks: about 22.8 deg), walk-off 3.19 deg, angular
  acceptance, `d_eff` from the 3m tensor with the traced fields, type II's `4 P_o P_e`; ledgers of both lines.
* `tests/test_second_harmonic_scope.py`: an absorbing crystal against `P_2(0) exp(-alpha L) ((1 - exp(-x)) / x)^2` (relative
  1e-6, `alpha L` = 1 and 0.1); a line's width against `sinc^2` averaged over its Gaussian spectrum from the Sellmeier formula
  (in scope at 0.1 nm, out of scope at 0.4 and 5 nm for a 2 mm PPLN); the harmonic's coherence length half the fundamental's;
  the first zero of the tuning curve against the phase-matched power; a pulsed source and cascaded crystals out of model scope
  with their targets.

## P10. Tracer semantics

* A *packet* is a bundle of rays sharing one source, one wavelength, one lineage and one medium; ray 0 is the chief ray and
  carries power, OPL, Jones vector and the Gaussian state. The other rays are the deterministic equal-power pupil samples
  (`pupil.py`); they represent the geometrical wavefront (phase-front normals), not diffraction.
* Every step takes the chief ray to its next crossing. At a physical surface every live ray advances to it; pupil rays that
  would first meet another physical surface or absorbing plane (or that miss) are retired as *diverged*; the tracer does not follow
  marginal stray light. A crossing of observation planes alone is every plane the chief ray crosses within
  `ray_min_advance_mm` of the first, whatever their orientation (planes through one point, coplanar, opposite or tilted, or
  closer together than the tracer can step), transparent planes before absorbing ones, then by id -- never by declaration
  order. Each plane records every live ray that crosses it before the surface that would stop or turn that ray, where that
  ray crosses it (its point, optical path length, length and power there); no ray moves -- the chief ray too stays where
  the leg began until the leg ends, so adding a transparent plane changes no other plane's, target's or exposure's result
  (it is one more trace step, and an escaping path's recorded length ends at the chief ray's last event) -- and no ray
  is retired there.
  An absorbing plane that counts the beam ends the path after the transparent planes of its crossing, which record only
  the rays that reach them no later than it (relative to the chief ray).
* Splitting elements (beam splitter, optional Fresnel reflection) create child packets; path ids are
  `<source>#<spectral index>[.<k>...]` (`k` counts a path's branches in the order it creates them), so path ids do not
  depend on Python hashing or on the order in which queued packets are traced. The branch part is kept at most 96
  characters: a deeper branch's part becomes `~` and the first 32 hex digits of the SHA-256 of the part it would have had,
  and its own branches append `.<k>` again (`optics/rays/path_ids.py`). An id is therefore at most 256 characters for a
  source id of at most 128, whatever the trace limits — a two-splitter cavity at R = 0.9 is followed through about 240
  reflections before its power falls below `power_epsilon_W` — and the optical simulation scene admits that length.
* Recombination is represented by observations that receive arrivals from several paths (`InterferenceResult`), never by
  merging paths.
* Termination reasons: `ESCAPED, CLIPPED, ABSORBED, DETECTED, OUTSIDE_ACCEPTANCE, BLOCKED_BY_BACK_SIDE, MAX_INTERACTIONS,
  ZERO_POWER, GRAZING_INCIDENCE, MEDIUM_MISMATCH, MATERIAL_OUT_OF_RANGE, COATING_OUT_OF_RANGE, DEGENERATE_SURFACE,
  TRACE_BUDGET, ANNULAR_BEAM_NOT_FOLLOWED, TRUNCATED_BEAM_NOT_FOLLOWED, ND_UNCOATED_REGION, FARADAY_RESPONSE_UNAVAILABLE`
  (`TRUNCATED_BEAM_NOT_FOLLOWED` is no longer produced, and `ANNULAR_BEAM_NOT_FOLLOWED` only where no pupil ray passes beside
  an obstruction: a beam whose power is not computed goes on as geometry and ends where the geometry does, with
  `final_power_W` null; `COATING_OUT_OF_RANGE`, `ND_UNCOATED_REGION`, `FARADAY_RESPONSE_UNAVAILABLE` and
  `RETARDANCE_UNAVAILABLE` end only a probe of untraced light). `OUTSIDE_ACCEPTANCE`: an absorbing plane stops a
  beam on its active area outside its acceptance cone without detecting it.
* Observation planes have no front: a plane observes (and an absorbing one absorbs) a beam from either side, and its
  acceptance cone is two-sided -- the AOI is measured from the normal line, `arccos |d . n|`, so a beam travelling
  against the normal at 0 deg is inside a 10 deg cone. To accept light from one side only, orient the plane and keep
  the other side dark. Without an active area every crossing counts; with one, a crossing counts when the chief ray is
  inside it or at least `capture_min_power_fraction` of the Gaussian power falls on it. A transparent plane receives
  the whole power of a counted beam (not its share on the area; `active_area_power_fraction` gives the share), an
  absorbing one detects its share on the area.
* Interaction limit: a path stops after `max_interactions_per_path` (64) physical interactions, the guard against
  endless cavity loops. A branch (a split, a traced reflection) starts with the count of its parent including the
  interaction that creates it, so a cavity followed through its reflected branches (a plate-splitter cavity) stops at the
  limit too. Observation planes are virtual: crossing them never counts toward it, so adding monitors does
  not move where the interaction limit stops a path (a two-mirror cavity stops after 64 reflections with 0, 1 or 2
  monitors inside). The stopped path's power is the ledger's `INTERACTION_LIMIT`; the `MAX_INTERACTIONS` issue is
  `UNKNOWN`, and so is every result that light could still change. The run's step budget (`max_trace_steps`, below) is
  different: it counts every step, observation crossings included (they cost the same work), so in a long cavity each
  monitor spends it sooner (with `max_trace_steps` 100 the cavity above stops after 64, 33 or 16 reflections with 0, 2
  or 5 monitors, `TRACE_BUDGET`, `MAX_PATHS` issue).
* Trace budget: a run traces at most `max_paths` beam paths (512: source spectral lines plus every branch a splitting
  surface creates) in at most `max_trace_steps` chief-ray steps over all paths (2048). The strongest source lines are
  launched first and the strongest queued packet is traced next, by power on a grid of the run's launched power times
  2^-40 (ties in source and queue order; NUMERICAL_CONVENTIONS section 3), so a cascade's weakest
  branches are the ones left out. Beyond it source lines and branches are not traced and the path being traced when the
  steps run out stops (`TRACE_BUDGET`); their power is the ledger's `PATH_LIMIT`, and one `MAX_PATHS` issue per run names
  the limit reached and gives the counts and that power. Source lines share the budget with branches: a line through N
  surfaces takes N + 1 steps, so a run traces 512 lines through one lens to a monitor plane (16 sources of 32 lines)
  and fewer through more surfaces. A target that no traced beam reaches while the budget left its source lines untraced is `UNKNOWN`
  (`MAX_PATHS`), not 0 W or `OBSERVATION_NOT_REACHED`. A power summed over the matching beams is bounded: the traced
  light is a lower bound of `power_W`, and on an absorbing plane the traced light plus the untraced power `U` is an upper
  bound (each untraced watt is received at most once; a transparent plane may be crossed again and again by it, so there
  nothing bounds it from above); for `coherent_power_W` (untraced light of the wavelength may interfere either way) the
  interval is `(√P − √U)²` (0 when `U ≥ P`) to `(√P + √U)²` on an absorbing plane, 0 to unbounded on a transparent one. A
  requirement is decided only where the whole interval agrees -- a minimum passes when the lower end meets it and fails
  when the upper end misses it, a maximum passes when the upper end meets it and fails when the lower end exceeds it --
  and is otherwise `UNKNOWN`; the value reported is the traced light's, the message states the interval and the cause
  code (`MAX_PATHS`) is listed. `power_rating_utilization` reports likewise the largest exposure / damage threshold of
  the ratings that apply (the exposure exact, or its lower bound; no value when none of those exposures is known or
  bounded from below) and is graded on its interval -- from an incomplete trace on `[lower, unbounded)`, `lower` the
  largest exposure / damage threshold of the ratings that bound the utilization from below (as rated, or with a
  threshold at most the rated value), which lies below the value where a rating with a threshold at least the rated
  value gives the larger ratio; light of unknown fate of the rated beams' own line may also cancel them, so there the
  utilization has no lower bound and is `UNKNOWN`. Issues name the requirements they leave `UNKNOWN` or
  `OUT_OF_MODEL_SCOPE` in `affected_targets`. A two-beam target on a plane where only the first `max_interference_pairs`
  pairs were analysed (`MAX_BEAM_PAIRS`) is `UNKNOWN` unless it names both beams and that pair was analysed. The same
  holds for the other power of unknown fate (`INTERACTION_LIMIT`: `MAX_INTERACTIONS`; `TERMINATED_OUT_OF_MODEL` /
  `SOURCE_NOT_TRACED`: `OUT_OF_MODEL_SCOPE`); with none, a plane no traced beam reaches receives 0 W of those lines
  (`NO_TRACED_ARRIVAL`) -- provided none of the light the trace does not follow can reach it (next item). A traced path
  costs about 70 kB of run files per step (its pupil rays in `ray_data.json`, the scene and the GLB), so a run stays
  within about 150 MB.
* Known zero and untraced light (`validation/untraced.py`). Some light leaves the traced beams by a model approximation
  while going on physically: an interface reflection the trace does not follow (`UNTRACED_REFLECTION`), the
  transmission of a reflector's coating (`UNTRACED_TRANSMISSION`), the Gaussian part inside the opening of a beam whose
  chief ray falls on the rim where no pupil sample passes (booked `CLIPPED` as a whole; where samples pass, the beam goes
  on as them, P1), the part of a beam beyond an element's support (it passes
  beside the element; inside `APERTURE_TRUNCATION`). On a plane a traced beam of the line reaches, the power's value
  counts the traced beams; every run with an observation plane follows every branch, and the
  untraced light of the line that reaches that plane bounds its power from above as well (a weak traced beam does not
  stand for a strong ghost or coating transmission of its line). A plane no traced beam of a line reaches is a known zero
  for that line only when none of it arrives above the line's negligible level (`negligible_power_fraction` of its input); a partial crossing at or below that level is left out. Each such branch (more than "negligible", below) is followed by a probe trace: the
  same tracer, from the branch's own state (chief ray, Jones vector, Gaussian beam, medium), through every optic it
  meets -- refracted, reflected, split, absorbed, clipped like any traced beam -- with the pupil rays of two whole rings
  of the beam it leaves (its Gaussian state checked against them where it meets a surface; where a traced beam's power
  that passes is not computed it goes on with its whole power, an upper bound), within its own trace budget
  (`max_paths`, `max_trace_steps` for all probes together). The probe's own untraced light (a ghost of
  the ghost, and so on) is followed the same way, recursively and strongest first, to every order above the negligible
  level; a branch at or below it is not followed. The branch reaches the plane when its probe (or a descendant) arrives
  there (what the plane receives from it) or crosses it beside the active area with a non-negligible Gaussian tail on
  it; probe light whose fate is not known (the probes' budget -- then a `MAX_PATHS` issue at `untraced_light` --, an
  interaction limit, a path ended out of model scope, light at the power floor) may reach every plane, so a design
  whose ghost cascade the budget cannot follow (several uncoated lenses) keeps its dark planes `UNKNOWN`. Light passing
  beside an element (beyond its support), travelling in the element's surround, does not meet that element's surfaces on its way past; inside the element's own body (a lens's glass) that part meets the element's edge or mount, an ordinary known loss booked `APERTURE_TRUNCATION` like the main beam's, with no branch. Probe light whose fate is not known is reported by why: `UNTRACED_LIGHT_OUT_OF_MODEL` (a probe path ended out of model scope), `UNTRACED_LIGHT_BUDGET`, `UNTRACED_LIGHT_INTERACTION_LIMIT`, `UNTRACED_LIGHT_POWER_FLOOR` (codes `OUT_OF_MODEL_SCOPE`, `MAX_PATHS`, `MAX_INTERACTIONS`, `ZERO_POWER`). A beam that crosses the plane beside its active area is not an arrival when less than
  `capture_min_power_fraction` of it falls on the area, but its Gaussian tail there must still be negligible -- or
  rejected by the plane's acceptance cone, as a counted beam outside it is. The tail lies off the chief ray: it travels
  along the beam's local wavefront normal, tilted from the chief by up to `rho_max |1/R|` (`rho_max`: the farthest point
  of the active area from the chief's crossing, `R`: the wavefront radius of curvature), with an angular spread of at
  most the far-field half angle `theta_0` (taken to `aperture_quad_extent_sigmas` of it). Only a tail all of whose
  directions are outside the cone is dropped; one whose range straddles the cone's edge, or whose beam has no valid
  Gaussian state, may still arrive. A beam's power follows its chief ray, so an observation plane that part of the beam
  reaches on a straight leg its chief ray does not cross the plane on -- before the surface the leg ends at: a tilted
  absorbing plane or absorber, a fold mirror, a lens -- gets no value from that beam. At the end of each leg the tracer
  records such a plane (`PARTIAL_CROSSING`; a probe's own for its light) with a bound of the power that part carries: the
  Gaussian beam's share, the beam taken as rays parallel to its chief ray with its Gaussian profile, that meets the plane
  after the leg's start surface and before its end surface and on the plane's active area, with the beam's radius at its
  widest (or narrowest, for a share that holds the chief ray) on the leg (`UNTRACED_LIGHT_BOUNDED`); a share at or below
  the line's negligible level is left out. A pupil ray of a beam with a valid Gaussian state that crosses a plane end
  surface's plane beyond its support (through a hole, past its outline) ends there for this count: that light is booked at
  the end surface and goes on as its untraced branch, whose probe trace counts it once on the planes beyond. A plane start or end surface is taken as the plane through the chief ray's
  point there with the surface's normal; a curved one as a plane shifted as far as the surface reaches for the rays
  within `K` beam radii of the chief ray (beyond the chief ray's point for an end surface, back from it for a start
  surface; `K` makes the share beyond them an eighth of the negligible level). A sphere: the plane across the ray, by
  the roots nearest the chief ray's point (`|a| - sqrt(a^2 - 2 reach p - reach^2)` the way it bends) where those lie
  inside the cap's aperture, else by its whole cap. A sag surface: its tangent plane at the chief ray's point, shifted
  by `D (n_t . n) / |d . n_t|` with `D = H r^2 / 2 + 2 |L| r` the most its profile departs from that plane within `r`
  of the point (`H` a bound of the profile's curvature there: the conic term's `c / S^3` and `c / S`, each polynomial
  term's second derivative at the window's outer end; `L` the axicon term), `r` the part of the support a ray within
  `K` radii can meet (`(reach + depth |n x d|) / |d . n|`, every hit of such a ray; or, where that is wider than the
  support, across the axis' plane, the radius within the support that the hit continuing the chief ray's cannot leave
  -- the assumption there is that a ray of the envelope meets the start or end surface at that hit, not at another
  hit of its line, a fold of the surface across the ray); else the plane across the ray with the sag range over that
  part (the conic term monotonic, each polynomial term at most `|A| rho^(2j+4)`, the axicon term between `L rho` at its
  ends) or the support's whole extent. Off-axis rays that meet a convex surface later than its tangent plane are so
  counted, and the bound near such a surface may be the whole beam; a sag surface without a bounded support leaves the
  bound to the pupil rays (below). The share of the beam that meets the surface the chief ray meets next before a
  plane it crosses (above) takes that surface the same way, except that a sag surface's tangent plane is used only
  with the first `r` (every hit): across the axis' plane the surface is the plane across the ray before every point of
  the support a ray can meet, so a ray whose first hit is the near rim of a steep mirror, not the part the chief ray
  meets, is counted; the share it takes is then bounded on the surface itself (`_sag_region_share`, above), so an
  off-axis paraboloid used as a collimator (the beam from its focus side) reads a monitor on its incoming leg within the
  share its surface may take first. A sphere there is taken by the roots nearest
  the chief ray's point only where no ray of the envelope can meet the cap first at the other root of its line: where
  the chief ray's line crossed the sphere before its hit, at `P2`, every point within the envelope's reach of `P2`
  (across the ray the reach, along it the nearest roots' spread) must lie below the cap's rim plane; else the whole
  cap counts (the near rim of a steep cap the beam passes at oblique incidence). A plane within `ray_min_advance_mm` (1e-7 mm) of a
  physical surface's chief-ray hit point records the power arriving at that surface before the interaction (each such
  arrival carries `surface_coincidence`, and the beam that leaves the surface back across the plane is a second arrival with
  `role` `RETURNING`: `docs/engine/RESULTS_AND_EXPORT.md`). A plane through a hit point whose active area lies beside it counts no arrival there and keeps the light of the leg leaving that surface: it reads `BOUNDED` from 0 to that light's bound when that light may reach the area under a supported bound, `UNKNOWN` without one. Only a plane that counted an arrival there leaves that leg out of its reading. A monitor through a fold mirror's hit point reads the power arriving at the
  mirror, and the same plane 1 nm before the hit point is outside this window and refused as partly intercepted (the mirror meets part of the beam first). A plane along the leg (exactly parallel to the chief ray)
  with an active area gets the share whose energy flow crosses it on the stretch of the leg along that area: a paraxial
  Gaussian beam's flow lines keep their place in its profile (`rho / w(s)` fixed), so that share is the share beyond the
  plane where the beam is widest on the stretch less where it is narrowest (both ways round a waist inside it; for an
  astigmatic beam crossed off its axes, the share beyond it where it is widest); without an active area, the share
  beyond it where the beam is widest on the leg. That share is the parallel-ray picture on a finite leg. The bound of the part
  of a beam that crosses a plane its chief ray does not cross on that leg is the larger of it and the paraxial energy flow across
  the plane; on a leg that leaves the scene only the energy flow exists. On a leg that leaves the scene, a plane without an active area counts the light that crosses it up to `max(1000 mm, support)` from the leg's start, `support` being the farthest along the ray of the design's physical surfaces, source points and active areas; no plane's origin enters it. The 1000 mm is a declared numerical choice, recorded with every run as `laboratory_reach_mm` in `manifest.numerical_tolerances` and not configurable. Part of a beam beyond that distance on a leaving leg is not counted on a plane with no active area; a chief ray that crosses the plane is an arrival wherever it crosses. To read it, state an active area or end the leg on a dump. A plane with an active area counts what crosses that area, wherever it lies; a chief ray does not cross such a plane beyond `max(1000 mm, support)` along it (that crossing is off the area), so that beam is not listed in `outside_active_area` and only its light on the area is counted. A paraxial Gaussian beam's energy follows lines that
  keep their place in the profile (`rho / w(s)` fixed). With `f(s)` the chief ray's signed distance from the plane, `N` the
  length of the plane normal's component across the ray and `w(s)` the beam radius across the plane, the lines that meet the
  plane at `s` are those at normalized offset `z(s) = -f(s) / (N w(s))`. For a round beam, or a plane along a beam axis, the
  power that crosses is the total variation over the stretch of the Gaussian share beyond `z(s)`, each crossing counted.
  Otherwise it is at most `4 exp(-2 h_min^2)`, `h_min` the least `|z(s)|`. With an active area the stretch is the area's
  extent along the ray, the share beyond the area's offset across the plane multiplies the result, and the area's footprint at
  the beam's widest radii (with its flow and narrowest-radius factors) also bounds it. A beam whose chief ray crosses the plane
  beside the active area is no arrival, and its tail on the area counts at least this flow. Reference, by independent
  numerical integration of the flow (`tests/test_tail_dark_planes_and_surface_coincidence.py`): 1 mW, 780 nm, 20 um waist, 5 mm
  disc 2 mm off axis at x = 90, parallel to the beam: flux 2.71e-7 W, bound 2.73e-7 W. The bound is that of the paraxial
  beam: a plane more than `K` beam radii from the chief ray over the stretch reads as dark, `K` about 3.3 at
  `negligible_power_fraction`. Behind a severe hard-edge truncation (`GAUSSIAN_SEVERE_TRUNCATION`) a traced beam has no Gaussian state, and its diffraction falls off too slowly for its rays to locate it. Any plane the line does not light with a point ahead of that leg's start, within that reach on a plane with no active area, reads `BOUNDED` from 0 to the beam's power: `total_power_W` null, `power_lower_bound_W` 0 and `power_upper_bound_W` the beam's power, `DIFFRACTED_BEAM` in the plane's `followed_light` and in the `OBSERVATION_NOT_REACHED` notice's `untraced_light`, status `PASS`, and `GAUSSIAN_SEVERE_TRUNCATION` among its target's codes. A `power_W` maximum at or above that bound is `MET` and a minimum above it `UNMET`; any other power limit, and any beam metric there, is `INDETERMINATE`. A plane the line does light keeps its reading, and the light is not taken out of the beam, on an absorbing plane too. To decide any other target there, open the aperture until the beam keeps its Gaussian state, or place the plane where the line lights it. A milder truncation keeps its Gaussian state, and aperture diffraction is not propagated. A beam without a Gaussian state for any other reason, and untraced light (a ghost a lens's rim truncates), is bounded by the whole beam within three envelopes of its live pupil rays; beyond that reach, or with no live pupil ray, nothing is recorded for that leg. An absorbing plane takes that part
  out of the beam (on its active area): the beam's power is out of model scope from there on (`power_out_of_scope_at`),
  as is the power a transparent plane records after part of the beam met an absorbing plane or a physical surface on
  the leg.
  "Negligible" is the line's negligible level (`negligible_power_fraction` x its input, a caller setting). Otherwise a
  power target on the plane summed over the line (`power_W`, `coherent_power_W` without `path_id` /
  `crossing_index`) carries the cause (`UNTRACED_LIGHT_BOUNDED` for an untraced reflection / transmission or
  light beside an element the probes followed, whatever else was not followed; `CLIPPED`; `INCOMPLETE_ENVELOPE` for a
  tail), the plane's power is `null`, never 0 W, and
  `untraced_power_W` states the power of that light where it arrives at the plane (the first-order estimate) plus the
  power whose fate is not known. That light is detected and not traced: the plane states no power. Where only light the
  probes followed reaches it under a supported bound the plane reads `BOUNDED` from 0 to `power_upper_bound_W`
  (`power_lower_bound_W` 0, that light in `followed_light`, status `PASS`); with light of unknown fate, or without a
  finite bound, it is `UNKNOWN`, with `power_upper_bound_W` beside its state where a bound is supported. A `power_W` or `coherent_power_W` target there is graded on the
  interval from 0 to the bound of the light the probes followed to the plane (for `coherent_power_W` the square of its
  summed field amplitudes): a maximum at or above the bound is `MET`, a minimum above it is `UNMET`, any other limit is
  `INDETERMINATE`; with light of unknown fate, or without a finite bound, the target is `INDETERMINATE`. A beam metric
  (radius, capture fraction) is `INDETERMINATE` there, with any selector (`path_id`, `crossing_index`) that selects no arrival. A selector that selects an arrival reads that arrival's value, with no untraced light added. So a ghost of a lens or window that leaves the scene, or that a blocking filter sends
  back toward the source, keeps the known zero of a plane it cannot reach; a pick-off detector in a window's reflection
  is `UNKNOWN` with the reflections' power, and so is a detector only a ghost of a ghost reaches (a second window's
  reflection of the first one's ghost). It never adds power to a result. The tail of a beam whose chief ray misses a
  surface its outer rays meet (`DIVERGED`, P1) is not part of it. The light left out as negligible still bounds a plane's
  power targets, the one case of a plane with no traced arrival graded on a bound (`quantity_state` `BOUNDED`): `power_W` from 0 to its
  power on an absorbing plane, on a transparent one to its power times its open paths' crossings of the plane (at least
  twice, once either way; once per straight leg after each number of reflections), more where two surfaces facing each
  other return it across the plane (no upper bound where nothing bounds their returns); `coherent_power_W` to the square of its
  field (`NO_TRACED_ARRIVAL`, `FAINT_LIGHT_BOUNDED`, no value). The plane's own `power_by_wavelength` stays a zero at
  the line's negligible level. Probe light at the interaction limit is never left out as negligible (it may be stored
  in a cavity).
  Below-floor power (`BELOW_POWER_FLOOR`) counts as zero when it is at most `negligible_power_fraction` times the input
  of the lines in question together (a 1e-12 W line's 3e-18 W remainder next to a 1 W line does not make their sum
  incomplete; the weak line alone is `UNKNOWN`, `ZERO_POWER`).
* The ideal thin lens and the aperture/mount model are idealizations declared in P1; they are not claims about a real lens.
