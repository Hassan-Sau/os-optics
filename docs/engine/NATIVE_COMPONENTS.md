# Native components

Native components are optical parts you describe by a few engineering parameters instead of picking a catalog product:
an iris, a pinhole, a slit, a lens designed for a focal length, an achromatic doublet, an axicon, a wedge prism, a window
or a cuvette, a mirror of stated reflectance, a hollow roof mirror or corner-cube retroreflector, a polarizing plate
splitter, a polarizer, a waveplate, a Faraday rotator of a stated response, a diffraction grating, a variable neutral-density filter, a lens described by a response table from another
tool (a metalens), a beam expander or a 4f relay. OpenSci turns the
parameters into a complete optical model that you add to your Project. The model is simulated, viewed and checked exactly
like a catalog product's model.

A native component is not a catalog product: it has no price and OpenSci does not source it. In an Order it is listed as
customer-supplied.

## Using them

From Python or a browser host (JSON API):

```python
from os_optics.local import json_api

json_api.native_components()                       # every class, its parameters, physics and references
built = json_api.build_native_model("lens.singlet", {
    "focal_length_mm": "100", "design_wavelength_nm": "632.8", "material": "N-BK7",
    "diameter_mm": "25.4", "form": "BEST_FORM"})
built["optical_model"]        # add to the Project's optical_models
built["member_models"]        # compound units only: their member models (add them too)
built["model_materials"]      # add to model_materials (null when the component has none, like the Project field)
built["model_coatings"]       # add to model_coatings (null when none)
built["component_template"]   # a component that uses the model: set its instance_id and nominal_pose
built["design"]               # the computed values (radii, thicknesses, focal lengths, indices) and the equations used

# into a new Project, each part as it is:
project["optical_models"] = [*built["member_models"], built["optical_model"]]
project["model_materials"] = built["model_materials"]
project["model_coatings"] = built["model_coatings"]
project["components"].append(built["component_template"])

# or let OpenSci merge it into an existing Project (ids already there with the same content are shared) and check it:
project = json_api.build_native_model("aperture.circular", {"opening_diameter_mm": "0.05"},
                                      {"into": project, "instance_id": "PH1", "position_mm": [50, 0, 0]})["project"]
```

From a shell:

```bash
os-optics native list
os-optics native describe lens.achromat
os-optics native build aperture.circular --param opening_diameter_mm=0.05
os-optics native build lens.singlet --param focal_length_mm=100 --param design_wavelength_nm=632.8 \
    --param material=N-BK7 --param diameter_mm=25.4 --into project.json --instance-id L1 --position 0,0,0 \
    --quaternion 0.5,0.5,0.5,0.5
```

`--into` adds the models, materials, coatings and the component to the Project file and checks the result. Numbers are
given as text (`100`, `0.05`); lists and objects as JSON (`--param 'valid_band_nm=[400,700]'`).

`capabilities()["native_component_classes"]` lists the class ids a runtime offers.

Every class description also states:

* `availability`: `["LOCAL_SIMULATION", "CUSTOMER_SUPPLIED"]` -- it simulates on this device, and a component using it is
  your own optic (a `custom_part`). A native class is not an orderable product;
* `catalog_equivalents`: where the public catalog has comparable products -- the `category`, the `forms`, the summary
  values to compare (`match`, e.g. `focal_length_mm`, `outer_diameter_mm`) and the `query` to run (`os-optics catalog
  list --category LENS --form PLANO_CONVEX_LENS --wavelength-nm 780`, one form per call). An empty list means the catalog has
  no product of that kind (stops, cuvettes, compound units, corner cubes, tabulated lenses). Classes added with this
  release name the forms the catalog publishes: `lens.axicon` -> `AXICON`, `prism.wedge` -> `WEDGE_PRISM`,
  `filter.variable_nd` -> `NEUTRAL_DENSITY_FILTER` (variable wheels among them), `mirror.hollow_roof` -> `RIGHT_ANGLE_PRISM`
  (a solid prism used through its hypotenuse returns a beam the same way, by total internal reflection). It helps you find a catalog product to compare with your
  parameters or to order instead; it does not claim that one is equivalent.

## Frames and placement

Every native element uses the catalog's frame conventions:

* transmitting elements (lenses, plates, stops, polarizers, waveplates): light travels along the element's local +Z; the
  origin is the first surface (or the element's plane);
* mirrors and plate splitters: local +Z is the normal of the coated face, pointing toward the incoming light; the origin
  is the centre of that face;
* elements placed by their ports (a wedge prism): local +Z is the nominal input direction (port `IN`), the origin is the
  body's centroid height on its axis, and port `OUT` gives the deviated direction at the design wavelength;
* compound units: local +Z is the unit's axis; the members sit on it. A retroreflector's +Z points from its apex toward
  the light (light enters travelling along -Z).

Place a component with its `nominal_pose`. Rotating it about its local Z turns a polarizer's transmission axis, a
waveplate's fast axis (both along local +X) and a slit's width (local X).

## Materials

A lens, window or cuvette names its material in one of these ways:

* a public catalog material id, for example `"N-BK7"`: its published dispersion is copied into the Project;
* `{"constant_index": "1.333", "valid_band_nm": [400, 900]}`: a constant refractive index over a band you state;
* `{"sellmeier": {"b": [b1, b2, b3], "c_um2": [c1, c2, c3]}, "valid_band_nm": [min, max]}`: a Sellmeier formula
  `n^2 = 1 + sum b_i l^2 / (l^2 - c_i)` (`l` in micrometres);
* a complete material document (`material_id`, `display_name`, `dispersion`, `valid_band_nm`).

Declared materials may add `material_id`, `display_name`, `absorption_coefficient_per_mm` (bulk absorption: power falls
as `exp(-alpha * path)`) and `basis`. OpenSci never supplies glass data of its own. Copying a catalog material needs a
catalog release on this computer (`os-optics catalog sync`).

Every material's refractive index must be real and at least 1 throughout its band: a Sellmeier formula with a pole inside
the band, `n^2 < 1` anywhere in it, or a constant index below 1 is refused (`NATIVE_PARAMETER_INVALID`).

A catalog material is copied the same way by the native library, the prescription import and `catalog_model`: its
published dispersion and band, with `basis` naming the catalog release. When you add a component to a Project that
already has a material of the same id (`add_to_project`, `--into`), the two are one material if their physics is the
same (dispersion, valid band, bulk absorption; the name and basis are only provenance), even when copied from different
catalog releases or by another route; the Project keeps its own entry. Materials whose physics differs are refused,
naming what differs; give one of them another `material_id`.

The model's wavelength band is the band all its materials (and coatings) cover, or a narrower `valid_band_nm` you give.
A run outside it is reported as outside the model.

A crystal class (`splitter.beam_displacer`, `polarizer.glan`, `polarizer.glan_thompson`, `splitter.wollaston`, `splitter.rochon`) takes a uniaxial crystal: `{"ordinary": INDEX, "extraordinary":
INDEX, "valid_band_nm": [min, max]}` with each `INDEX` `{"constant_index": n}` or `{"sellmeier": {"b": [..3], "c_um2": [..3]}}`,
or a complete material document with `birefringence`. A constant term `A` of a formula `n^2 = A + ...` is a Sellmeier term
with `c = 0` (`b = A - 1`); calcite, for example (G. Ghosh, Opt. Commun. 163, 95 (1999)):

```json
{"material_id": "CALCITE", "valid_band_nm": [400, 2000],
 "ordinary": {"sellmeier": {"b": ["0.73358749", "0.96464345", "1.82831454"], "c_um2": ["0", "0.0194325203", "120"]}},
 "extraordinary": {"sellmeier": {"b": ["0.35859695", "0.8242783", "0.14429128"], "c_um2": ["0", "0.0106689543", "120"]}}}
```

The model states the crystal's optic axis in its own frame (`optic_axes`); the run traces the ordinary and the
extraordinary beam of every crystal face (see `docs/engine/PHYSICS_SCOPE.md`, uniaxial crystals).

## Coatings

Lens, axicon, wedge, window and cuvette faces are uncoated unless you give `ar_coating`: a catalog coating id,
`{"reflectance": R, "valid_aoi_deg": [a, b]}` (a lossless coating, `T = 1 - R`), or a table over wavelength and angle of
incidence:

```json
{"angular_spectrum": {"wavelength_nm": ["600", "900"], "aoi_deg": ["0", "60"],
                      "reflectance_s": [["0.01", "0.1"], ["0.01", "0.1"]], "reflectance_p": [["0.01", "0.1"], ["0.01", "0.1"]],
                      "transmittance_s": [["0.99", "0.9"], ["0.99", "0.9"]], "transmittance_p": [["0.99", "0.9"], ["0.99", "0.9"]]}}
```

One row per wavelength, one value per angle, s and p power fractions read in the medium the light arrives from and
interpolated bilinearly; a face is read at its own angle of incidence (the exit face of a tilted window at the refracted
angle inside the glass). `valid_aoi_deg` defaults to the table's angles; light outside the table is outside the model.

Mirrors and splitters take a declared `reflectance` (or `reflectance_s` and `reflectance_p`), over a `valid_band_nm` you
state. Without a stated transmittance, a metallic mirror transmits nothing, and a dielectric mirror or a splitter is
lossless (`T = 1 - R`). Declared values are constant over their band. They hold over `valid_aoi_deg`, which defaults to
0-90 deg (the same values at every angle). Give the part's real angle range so that runs outside it are flagged.

## The classes

### Stops

All stops are an opaque plate at the element's plane with one opening. A ray crossing the plane inside the opening passes
unchanged. A ray hitting the plate is stopped. Light beyond the plate's outline passes by. Diffraction at the opening is not
modelled.

| Class | What it is | Key parameters |
|---|---|---|
| `aperture.circular` | iris, aperture or field stop, pinhole | `opening_diameter_mm`, `opening_center_mm`, `plate_diameter_mm` (default: 25.4 mm or twice the reach of the opening; a given plate must reach the far edge of a decentred opening, at `|centre| + radius` from the plate centre) |
| `aperture.slit` | rectangular opening | `slit_width_mm` (local X), `slit_height_mm` (local Y) |
| `aperture.annular` | open ring: an opening with an opaque central disc | `opening_diameter_mm`, `obstruction_diameter_mm` |
| `aperture.central_obstruction` | opaque disc in a clear carrier (beam block, DC stop, secondary-mirror obstruction) | `obstruction_diameter_mm`, `carrier_diameter_mm` |

### Lenses

**`lens.ideal_thin`**: a perfect lens of focal length `f` at one plane. Every ray is bent as `u' = u - y / f`. It has no
thickness, no aberrations and no dispersion. It adds no optical path, so phase and interference results through it are not
defined.

**`lens.singlet`**: a lens of one material whose paraxial focal length at `design_wavelength_nm` is `focal_length_mm`.

* Shape: `form` is `SYMMETRIC`, `PLANO_EXIT` (flat back, curved face toward the light), `PLANO_ENTRY` (flat front) or
  `BEST_FORM` (least spherical aberration for a collimated input). You can also give the Coddington `shape_factor`
  `X = (c1 + c2) / (c1 - c2)` directly.
* The curvatures solve the thick-lens equation `1/f = (n-1)(c1 - c2) + (n-1)^2 t c1 c2 / n` exactly at that shape
  and thickness.
* `BEST_FORM` uses `X = 2 (n^2 - 1) / (n + 2)`.
* `rear_conic: STIGMATIC_COLLIMATED` (with `PLANO_ENTRY`) makes the back surface the hyperboloid `k = -n^2`. A
  collimated on-axis beam then comes to a perfect geometric focus.
* `cylinder_axis: X | Y` makes both curved surfaces cylinders, straight along that axis.

**`lens.achromat`**: a cemented crown / flint doublet whose focal length at the centre wavelength is `focal_length_mm`
and which focuses a collimated beam at the same point for the short and long design wavelengths (equal back focal
lengths); the defaults are the F and C lines, with the d line as the centre.

* It starts from the thin-lens solution `phi1 = phi V1/(V1 - V2)`, `phi2 = -phi V2/(V1 - V2)`, with Abbe numbers taken
  at your three wavelengths.
* After thickening, the two element powers are re-solved on the real thick lens, so the foci coincide exactly for the
  lens you get (to the 1 nm the radii are written to).
* Coincident foci are the axial achromat condition. For a thick lens it is not the same as equal effective focal lengths:
  the principal planes move with wavelength, and an f = 100 mm, 25.4 mm doublet designed for equal EFLs would leave its F
  and C foci 10 to 130 um apart (more than the f/4 depth of focus). A doublet with coincident foci keeps a small EFL
  difference instead (a chromatic difference of image size). The design report gives both: `bfl_short_mm`,
  `bfl_long_mm`, `axial_focus_shift_long_minus_short_mm` (zero by design) and `efl_difference_long_minus_short_mm`.
* The crown is shaped by `crown_shape_factor` (default equi-convex). Not every shape has a solution: a fast doublet
  (for example f = 40 mm at 25.4 mm with N-BK7 / N-SF5) has no thick crown of the required edge thickness with
  coincident foci, and is refused with `NATIVE_DESIGN_INFEASIBLE`; try another shape factor, a longer focal length or a
  smaller diameter. The cement layer is not modelled.

**Thickness rule** (both designed lenses, for an element whose centre thickness you do not give): the thinnest element,
in 1 um steps, whose centre is at least `min_center_thickness_mm` (default max(1 mm, 0.08 D)) and whose edge is at least
`min_edge_thickness_mm` (default max(0.5 mm, 0.05 D)). A centre thickness you give is kept: its edge must be positive,
and at least `min_edge_thickness_mm` only when you give that too.

Radii are written to 1 nm. The model's paraxial data (EFL, back and front focal length, principal planes) is computed
from the written prescription. A focal length too short for the diameter is refused: a surface would not cover the outline.

**`lens.axicon`**: a cone lens of `cone_angle_deg` (the base angle `alpha` between the cone and the plane face) in one
material. `PLANE_FIRST` (default) has the plane face toward the light and the cone at the exit, its apex on the axis;
`CONE_FIRST` points the apex at the light. A ray parallel to the axis leaves at `beta = asin(n sin alpha) - alpha`
(plane first) toward the axis; a beam of radius `h` makes a line focus (the Bessel zone) up to `h / tan(beta)` behind the
cone. Give `design_wavelength_nm` for a design report with `n`, `beta` and the zone length of the clear aperture. The
centre thickness (at the apex) defaults to an edge of at least max(0.5 mm, 0.05 D); an angle at which the light is totally
reflected is refused. An axicon has no focal length: rays, power and spot metrics are traced behind it, while a Gaussian
beam there is outside the Gaussian model (the run says so).

**`lens.tabulated_thin`**: a thin lens whose response comes from another tool or a measurement (a metalens, a
diffractive lens). `rows` give the focal length and the delivered power fraction per wavelength, as a list of
`{"wavelength_nm", "focal_length_mm", "transmittance"}` or as three columns; `focal_interpolation`
(`LINEAR_IN_OPTICAL_POWER`, default, or `POINTS_ONLY`) and `transmittance_interpolation` say how values between rows are
read. The table alone builds the lens. When you have them, state the domain it was validated for --
`max_field_angle_deg`, `max_beam_radius_mm` and `validated_polarizations` (`LINEAR_X`, `LINEAR_Y`, `ANY`) -- the
`supported_metrics` it was validated for, each with its error bound and basis, and the `evidence` (method, tools and
versions, data hashes, fit, validation summary, date). Each traced line is focused with its own focal length and keeps
`T` of its power; the rest is the element's unmodelled response. A run outside a stated limit, or a target asking for a
metric outside a stated list, is outside the model's scope; a limit or list that is not stated is not checked. A table
without `evidence` is an assumed response: the run summary reports `data_kind` `ASSUMED`.

### Plates

**`plate.window`**: a plane-parallel plate of one material, round (`diameter_mm`) or rectangular (`width_mm`,
`height_mm`). Use a declared index and absorption for a sample slab.

**`plate.layered`**: layers of different materials in contact, given as `layers: [{"material": ..., "thickness_mm": ...},
...]`. Examples: a cuvette (wall / liquid / wall), a cemented plate stack. Every interface refracts between its two media.

### Mirrors and splitters

**`mirror.flat`**: one reflecting plane with a declared reflectance.

**`mirror.curved`**: a concave (`f > 0`) or convex (`f < 0`) mirror with vertex radius `R = 2 f`. `conic` is optional;
`-1` makes a paraboloid, which focuses an on-axis collimated beam perfectly.

**`splitter.plate`**: one coated plane that reflects and transmits the declared power fractions. With s / p values it
is a polarizing splitter. Its substrate is not modelled.

**`mirror.hollow_roof`**: two flat mirrors at 90 deg meeting along a ridge on local X, each at 45 deg to the axis, with an
aperture of `aperture_width_mm` (along the ridge) by `aperture_height_mm`. A beam travelling in the plane across the ridge
comes back antiparallel, and its position across the ridge is reversed (`y -> -y`): reflection matrix `diag(1, -1, -1)`
in the unit frame.

**`retroreflector.hollow_corner`**: three mutually perpendicular square mirrors of edge `mirror_edge_mm` (a cube corner)
with their axis along the cube diagonal. Every ray that meets all three mirrors returns antiparallel to itself
(reflection matrix `-I`), through the point reflection of its entry point about the apex.

Both are compound units of flat mirror members with the declared `reflectance` (as `mirror.flat`). Neighbouring faces
leave a seam of `seam_mm` (default 0.01 mm) along their common edge, so the mirror bodies never touch; light that falls
on a seam passes between the mirrors. Member poses are written to 1 nm.

### Prisms

**`prism.wedge`**: a round wedge prism of `wedge_angle_deg` in one material, the exit face tilted about local Y with the
thick end toward local -X. Light along the axis is deviated toward the thick end by `delta = asin(n sin alpha) - alpha`
(`(n - 1) alpha` for a thin prism); each wavelength leaves at its own angle. The model is placed by its ports: `IN` along
local +Z at the entrance face, `OUT` the deviated direction at `design_wavelength_nm` from the exit face.

### Polarization

**`polarizer.linear`**: transmittance `T` along its axis (local +X) and `T / extinction_ratio` across it.

**`waveplate.retarder`**: retardance in waves at `design_wavelength_nm` (0.5 for a half-wave plate, 0.25 for a
quarter-wave plate); fast axis along local +X.

* `ZERO_ORDER` (the default) scales the retardance as `delta0 * l0 / l`.
* `ACHROMATIC` keeps it constant over the band.

Both are thin elements: plate thickness and refraction are ignored.

**`splitter.beam_displacer`**: a rectangular plate (`width_mm` x `height_mm`, default 10 x 10) of a uniaxial crystal, `length_mm`
long (or the length that gives `displacement_mm` at `design_wavelength_nm`), its optic axis at `axis_angle_deg` (default 45) from
local +Z toward local -X. The ordinary beam (polarized along local Y) goes straight through; the extraordinary beam (polarized
along local X) walks off by `rho`, `tan(theta + rho) = (n_o / n_e)^2 tan(theta)`, and leaves parallel to it, displaced by
`L tan(rho)` along local +X (a negative crystal such as calcite; a positive crystal displaces it along -X). The design report
states both indices, the extraordinary phase index, the walk-off and the displacement.

**`polarizer.glan`**: two right-angle prisms of a negative uniaxial crystal with an air gap (`air_gap_mm`, default 0.02), a
square aperture `aperture_mm`; a compound unit (prism `A` holds the entrance face at `z = 0`, prism `B` the exit face). The gap
plane holds local Y; the axial beam meets it at `gap_angle_deg` (default: the middle of the window between the ordinary and the
extraordinary critical angles at `design_wavelength_nm`). The ordinary wave is totally reflected and leaves prism A through its
side face at local +X; the extraordinary wave passes along +Z. `GLAN_TAYLOR` (default): optic axes along local X, the
transmitted light polarized along local X (p at the gap); `GLAN_FOUCAULT`: axes along local Y, the transmitted light polarized
along local Y (s at the gap, an exact split). Faces with no traced surface absorb what reaches them.

**`polarizer.glan_thompson`**: the Glan geometry cemented: the gap is a layer of `cement` (an isotropic material whose index is
below the crystal's ordinary index), `cement_thickness_mm` thick (default 0.01), joined as a `COMPOUND` joint. The ordinary wave is
totally reflected at the crystal-cement face when `gap_angle_deg` exceeds `asin(n_c / n_o)`; the extraordinary wave crosses the
layer below `asin(n_c / n_e)` (at any angle when `n_c >= n_e`: then state `gap_angle_deg`). Optic axes along local Y: the
transmitted light is polarized along local Y, an exact split.

**`splitter.wollaston`** and **`splitter.rochon`**: two right-angle prisms of one crystal joined along a plane through local Y
at `wedge_angle_deg` to the entrance face, square aperture `aperture_mm`, each prism `edge_thickness_mm` thick at its thin end
(default a tenth of the aperture); an optical contact, or a `cement` layer. Wollaston: prism A's optic axis along local X, prism
B's along local Y; the two polarizations leave on either side of local +Z, about `2 asin((n_o - n_e) tan(theta))` apart.
Rochon: prism A's axis along local Z, prism B's along local Y; light polarized along local X leaves undeviated, along local Y
deviated. The design report gives both exit angles at `design_wavelength_nm` from Snell's law.

**`lens.grin_rod`**: a radial gradient-index rod, `diameter_mm` across, `length_mm` long or the length of `pitch` at
`design_wavelength_nm` (`L = 2 pi P / g`, rounded up to 1 um), plane end faces. Its `material` is a material document with
`gradient` (`{"kind": "RADIAL_QUADRATIC", "gradient_constant_per_mm": g}` or `gradient_rows`); its `dispersion` is the index on
the axis. The design report states `n0`, `g`, the pitch, the period `2 pi / g`, the paraxial matrix, the effective focal length
`1 / (n0 g sin gL)` and the back focal distance from the exit face.

**`grating.plane`**: a plane ruled grating on a rectangular plate: `mode` (`REFLECTION`: local +Z the ruled face's outward normal;
`TRANSMISSION`: light along +Z), `line_density_per_mm` or `period_nm`, `grating_vector_angle_deg` (0: the grating vector along local +X, the
grooves along local +Y), `orders` or `all_propagating_orders`, `width_mm`, `height_mm`, clear aperture, `thickness_mm` (the plate drawn behind the
ruled plane; it does not act on the light) and `response`, the order powers with their provenance.

* An order leaves by `n_exit t_exit = n_in t_in + m (lambda / period) K`, whatever the response says: `docs/engine/PHYSICS_SCOPE.md`, section P10d.
* Without a `response`, or where it does not hold, each order is traced with its power not known and bounded, and its position and
  direction at a plane stay valid. `results grating-orders RUN` lists each order and what is known of it.
* In the plate's own media other than air, write the `DIFFRACTION_GRATING` model directly (`medium_positive_z`, `medium_negative_z`).
* A line density or a product name says nothing about efficiency: state what you measured, computed or assume, as `data_kind`.

**`rotator.faraday`**: a nonreciprocal rotator described by what you state: `rows` of `wavelength_nm`, `rotation_deg` and
`transmittance`, `interpolation` and, when you have them, `valid_aoi_deg`, `operating_conditions` and `evidence`
(without `valid_aoi_deg` no angle of incidence is checked; without `evidence` the run summary reports `data_kind`
`ASSUMED`). Local +Z is the field axis:
`rotation_deg` turns the field right-handed about it whichever way the light travels, so a round trip adds the rotation.
`length_mm` is the rod drawn around its plane; it does not act on the light.

* Light of a wavelength outside the rows stops at the element, outside the model.
* Phase, wavefront and the linear optical network through it have no value: it states no optical path.
* Between `polarizer.linear` elements it builds an optical isolator. Launch one source from each end and read each on its own:
  `power_budget.lines[].observed_W`, or the arrivals of that source (`docs/engine/PHYSICS_SCOPE.md`, section P10c).

### Filters

**`filter.variable_nd`**: a variable neutral-density filter: power transmittance `10^-OD` at each point where light
crosses it.

* `law: LINEAR_ROTARY` -- a filter wheel (`diameter_mm`): OD linear in the azimuth of the crossing point about the axis,
  from `od_start` at `arc_start_deg` (default 0, local +X) to `od_end` at `arc_end_deg`, measured toward local +Y. Its
  position is the wheel's rotation `nd_rotation_deg`.
* `law: LINEAR_GRADIENT` -- OD linear along local X from `od_start` at `x_start_mm` to `od_end` at `x_end_mm` (round or
  rectangular). Its position is the shift `nd_offset_mm` along local X.
* `coated_inner_radius_mm` leaves an uncoated hub (a wheel coated on an annulus); `uncoated_transmittance` is the
  transmittance where the plate is uncoated (outside the arc or interval, in the hub). Without it, light there is outside
  the model.
* `reflectance` (default 0) is reflected as a traced beam; the rest of the light that is not transmitted is absorbed.
* `position` writes the starting rotation (deg) or shift (mm) into the component template's `selected_parameters`;
  change it there to move the filter.

A beam whose density varies by more than 0.2 OD across its 1/e^2 footprint (a large beam on a steep part of the law, or
one across the hub's edge) is outside the model. The density is the same at every wavelength of `valid_band_nm`.

### Modulators

**`deflector.acousto_optic`**: an acousto-optic deflector or modulator driven by RF tones: `acoustic_velocity_m_per_s`, `rf_tones_mhz` (a list, at most 64), `diffraction_order` (`+1`, toward the acoustic axis, optical
frequency up by `f_RF`; `-1`, away, down), `acoustic_axis_angle_deg`, `width_mm`, `height_mm`, `efficiency` and `design_wavelength_nm`. It compiles onto one `DIFFRACTION_GRATING`, transmitting, of period `v / df`, `df` the greatest common divisor of the
tones; each tone is the grating's integer order `f_RF / df` and the zero order stays: `sin(theta) = sin(theta_in) + m lambda f_RF / v`. `design.tones` maps each tone to its order and its `optical_frequency_shift_hz` (`m f_RF`).

* `efficiency` is what you state: `{"rows": [{"wavelength_nm", "rf_mhz", "efficiency"}, ...], "valid_aoi_deg", "data_kind", "basis", "evidence"?}`, every tone at every wavelength, `rf_mhz` `0` for the zero order, for this set of tones. Without it each
  order's direction is traced and its power is not known.
* `design.quantities` says how each quantity is known: direction and frequency shift `MODELED`; order power `EXTERNALLY_CHARACTERIZED` or `UNAVAILABLE`; tone phases, timing, RF crosstalk and intermodulation `UNAVAILABLE`.
* Tones add in irradiance at a detector: they differ in optical frequency. `results pupil-field` with a `HANDOFF_BEAMS` illumination takes them on to a focal plane (`path_ids` leaves the zero order out).
* Tones on a very fine common step make a very coarse grating: a period above 10 mm is refused with the step that would do.

### Compound units

A compound unit is placed as one component. Every surface of its members is simulated. Its lenses are any of the lens
classes above, given as `{"class_id": ..., "parameters": {...}}`.

Spacing is paraxial, at `design_wavelength_nm`: the second lens's front focal point sits on the first lens's rear focal
point, so the unit is afocal.

| Class | What it is | Notes |
|---|---|---|
| `compound.keplerian_telescope` | two positive lenses (beam expander) | beam magnification `-f2/f1`; `focal_stop_diameter_mm` puts a pinhole at the internal focus (a spatial filter) |
| `compound.galilean_telescope` | a positive and a negative lens | beam magnification `-f2/f1`, upright, no internal focus |
| `compound.relay_4f` | two positive lenses imaging the unit's origin (object plane) to its `OUT` port (image plane) | magnification `-f2/f1`; optional Fourier-plane stop: `fourier_stop_diameter_mm` (low-pass), `fourier_obstruction_diameter_mm` (DC block), both (band-pass) |

Member positions and ports are written to 1 nm (1e-6 mm). The design report's system matrices are those of the unit as
placed, so the 4f relay's object-to-image `B` shows the sub-micrometre residue of that rounding rather than an exact 0.

## Field models

Two devices act on a field or an irradiance map and have no optical model of their own. `native_components()` lists them under `field_models`, `os-optics native list` beside the classes, and `os-optics native describe ID` prints one. They depend on no
catalog or supplier: you state the device. Place a mirror, a window or an observation plane in the Project where the device sits if the layout needs it.

| Id | What it is | Run it with | Takes |
|---|---|---|---|
| `slm.pupil_mask` | A spatial light modulator or phase plate: a sampled or parameterized mask in a pupil, and the scalar focal field behind an ideal lens | `pupil_field`, `os-optics results pupil-field` | wavelength, focal length, pixel count, pitch and fill, opening, illumination or a run's beams, mask, device efficiency with its source |
| `detector.camera_response` | A camera or photodetector: photons, electrons, ADU, saturation, noise and SNR per pixel | `detector_response`, `os-optics results detector` | QE rows with their source, exposure, pixel pitch and fill, full well, read noise, dark current, gain, bit depth; each optional |

Both state how each quantity is known (`PHYSICS_SCOPE.md`, P10e). Their own limits are in `native_field_models()`.

## What a native model does not include

These hold for every native class:

* Tolerances, surface errors and scatter are not modelled.
* The light that a transmitting surface reflects back (a ghost) is not followed.
* Stops and Fourier filters act geometrically: diffraction is not modelled; an axicon's Bessel beam is traced as rays.
* Thin elements (ideal and tabulated thin lenses, neutral-density filters, polarizers, waveplates) have no substrate.
  Ideal and tabulated lenses and neutral-density filters state no phase, so a linear optical network through them is
  refused.

Each class lists its own approximations in `native_components()`.

## When a request is refused

The reply lists every problem with a code and the parameter it concerns:

| Code | Meaning |
|---|---|
| `NATIVE_CLASS_UNKNOWN` | no such class |
| `NATIVE_PARAMETER_UNKNOWN` | the class takes no such parameter |
| `NATIVE_PARAMETER_MISSING` | a required parameter is absent |
| `NATIVE_PARAMETER_INVALID` | wrong type, out of range or inconsistent |
| `NATIVE_DESIGN_INFEASIBLE` | valid parameters, but no physically valid element meets them (for example a focal length too short for the diameter) |
| `NATIVE_CATALOG_DATA_UNAVAILABLE` | a named catalog material or coating cannot be copied |

Parameters of a compound unit's lens are reported as `lens1.<name>` / `lens2.<name>`.

## References

* E. Hecht, *Optics*, 5th ed.
* M. Born and E. Wolf, *Principles of Optics*, 7th ed.
* R. Kingslake and R. B. Johnson, *Lens Design Fundamentals*, 2nd ed.
* W. J. Smith, *Modern Optical Engineering*, 4th ed.
* J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed.
* J. H. McLeod, "The axicon: a new type of optical element", J. Opt. Soc. Am. 44, 592 (1954).
* J. Durnin, J. J. Miceli and J. H. Eberly, "Diffraction-free beams", Phys. Rev. Lett. 58, 1499 (1987).
* H. D. Eckhardt, "Simple model of corner reflector phenomena", Appl. Opt. 10, 1559 (1971).
