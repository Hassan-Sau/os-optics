# Linear optical network export

`os-optics` exports the optics of a Project as a **linear optical network**: the input modes (each source), the
output modes (each beam leaving the layout through an output observation plane), the complex transfer matrix between them,
every beam path with its optical path length, and the Jones matrix of every element the light meets. The document format is
`opensci-linear-optical-network/v1`. Use it to run quantum-optics or classical simulations of an OpenSci layout in a
simulator of your choice (Perceval, Piquasso, your own code); OpenSci adds no simulator dependency. The quantum state of
every source (photon number, wavepacket, indistinguishability) and every detector property are your inputs to that
simulator: they are never inferred from the classical beam. `os-optics interop show quantum-photonics` lists the
tools, inputs and limits of this workflow.

The exported network is a reproducible description of the physical layout as the Engine models it: the same Project
gives the same document, with the same `contract_hash`, in Python and in the browser runtime (see "Precision and the
contract hash" for what that rests on). What you compute with it in your simulator is your computation.

The network holds amplitudes of Gaussian beams. An `EXTENDED_EMITTER` source has no phase relation between its rays and is not part of it; a Project whose only sources are extended emitters is refused `OUT_OF_MODEL_SCOPE`.

## How to export

```bash
os-optics project network project.json --out network.json
os-optics project network project.json --wavelength-nm 780 --observation D1 --observation D2 --out network.json
```

```python
from os_optics.local import json_api

reply = json_api.linear_optical_network(project_document, {"wavelength_nm": 780.0})
if reply["ok"]:
    network = reply["network"]          # the opensci-linear-optical-network/v1 document
    reply["contract_hash"]               # equals network["contract_hash"]
else:
    reply["error"]["code"]               # NETWORK_NOT_EXPORTABLE, or LOCAL_RUN_ERROR for a Project that cannot run
    reply["error"]["problems"]           # [{"code", "message", "location", "observation_id"?, "source_id"?}]
```

A problem that concerns one output plane or one source names it in `observation_id` / `source_id`: export again with
that plane or source left out of `observation_ids` / `source_ids`. A finding refuses the export only when it concerns
light the network describes: a surface on the path of light that reaches an exported output plane, such a plane, or an
exported source. An out-of-model finding about light that reaches no exported output (a branch toward a plane the
export leaves out) is listed in `validity.warnings` with its code and location, and the matrix is that of the exported
outputs.

`os_optics.local.export_linear_optical_network(project, options=...)` returns the same document as a
`LinearOpticalNetworkV1` model; an Engine request (`ValidationRequest` JSON) is accepted too. The browser runtime runs
the same function. `json_api.json_schema("linear-optical-network")` and `os-optics schema linear-optical-network`
return the JSON Schema of the document.

Options (all optional):

| Option | Meaning |
|---|---|
| `wavelength_nm` | The spectral line to export. Required when the sources emit more than one wavelength. |
| `observation_ids` | The observation planes that are output ports. Default: every observation plane. |
| `source_ids` | The sources that get input modes. Default: every source. |
| `mode_overlap_tolerance` | Two beams on one plane are the same mode when their mode overlap is at least `1 - tolerance`, and separate modes when it is at most `tolerance`. Default `1e-6`. |

Place an observation plane on every output port, after the last optical element of that port, and a source on every input
port that should have a column: an input port without a source (a vacuum port) has no column in `T`.

## What the document holds

| Field | Content |
|---|---|
| `wavelength_nm` | The exported line (vacuum wavelength). |
| `input_modes` | `in:<source_id>#<line>:<H\|V>`: the source's emission point and direction, its lab `H` / `V` unit vectors, line power, declared polarization in this basis, coherence group and coherence length. |
| `output_modes` | `out:<observation_id>:<beam>:<H\|V>`: each distinct beam crossing an output plane: crossing point, direction, `H` / `V`, refractive index there, its Gaussian mode (beam radii, waist radii, waist distances, axes, M²) and `reference_opl_mm`, the optical path length of the beam's first path (the phase reference of its rows). Ids are joined with `:` and an observation or source id may itself contain `:`: read `observation_id` / `source_id` from the mode, never by splitting an id. |
| `transfer_matrix` | `entries[row][column]` = `T` as `[re, im]` (rows: output modes, columns: input modes), `column_transmission` (Σᵢ\|Tᵢⱼ\|²), `singular_values`, `lossless_tolerance` and `lossless` (\|TᴴT − I\| within that tolerance). When no light reaches any output, `T` has no rows, every column transmission is 0 and the note `NO_OUTPUT_REACHED` says so. |
| `paths` | Every beam path from a source to an output beam: `opl_mm`, `opl_offset_mm` (its optical path length minus its output beam's `reference_opl_mm`), `length_mm`, `group_opl_mm` and `group_delay_ps` (the group optical path Σ n_g × length and its delay, see "Group delay"), the interactions it meets (`lineage`) and its 2 × 2 Jones amplitude matrix without propagation phase. `T` is the sum over paths of `amplitudes · exp(i k₀ reference_opl_mm) · exp(i k₀ opl_offset_mm)`. |
| `elements` | The 2 × 2 Jones matrix of each interaction in the lab bases of its input and output directions (mirror, splitter reflection / transmission, refraction, waveplate, polarizer), with its angle of incidence. |
| `hints` | `path_pairs`: for every two paths that reach the same output beam, the optical path difference and phase delay (`opd_mm`, `delay_ps`), the group optical path difference and group delay (`group_opd_mm`, `group_delay_ps`) and the coherence length; `notes`: what the network does not include. |
| `classical_check` | Per source and output plane, for the source's declared polarization and for lab `H` and `V` input: the output-plane power `Σ\|T J\|²` against the Engine's own coherent superposition of the same beams in a trace with that polarization; and the power ledger (outputs + losses by kind = 1). See "What the classical check validates". |
| `provenance` | Engine version and build id, numerical configuration version, catalog release and its SHA-256, Project hash, Engine input hash, the options. |
| `contract_hash` | SHA-256 of the canonical JSON of the document without `contract_hash`. |

A source that states a continuous spectrum (`spectral_density`) is traced as quadrature lines; the network is that of the
line at `wavelength_nm`, and `line_power_W` is that line's share of the source's power. Its `coherence_length_mm` (input
modes and path pairs) is the spectrum's: the smallest path difference at which the degree of coherence of the declared
spectral density falls to 1/e, the value the run reports in `source_spectra`. The width of one quadrature line is the bin
width, so a line's own coherence length depends on the number of lines and is not exported. The note `CONTINUOUS_SPECTRUM`
names the source, its line count and bin width; when the degree of coherence stays above 1/e over the scanned path
differences, `coherence_length_mm` is null and the note `COHERENCE_BEYOND_SCAN` says so.

## Conventions

The document repeats these in `conventions`.

* **Time dependence**: `E ∝ exp[i(k·r − ωt)]`. Propagation over an optical path length `L` multiplies the amplitude by
  `exp(+i k₀ L)`, `k₀ = 2π / λ` (vacuum wavelength). This is the sign of a phase shifter `exp(iφ)` in Perceval.
* **Amplitudes are flux-normalized**: `|Tᵢⱼ|²` is the probability that a photon entering in mode `j` leaves in mode `i`.
* **Transfer definition**: `b_out = T a_in` for field amplitudes; for photons `a†ⱼ → Σᵢ Tᵢⱼ b†ᵢ`. `T` is not unitary when
  light is lost (absorbed, reflected out of the layout, missing every output); `column_transmission` gives the surviving
  fraction of each input.
* **Polarization basis**: at every port, `V` = design `+Z` (up) projected perpendicular to the beam and `H = V × d`, so
  `H × V = d`. For a beam within 45° of vertical (`(d·Z)² > 1/2`), design `+Y` replaces `+Z` (note `VERTICAL_BEAM_BASIS`).
  No rule avoids a switch somewhere on the sphere of directions; at 45° both references are well conditioned, and a mode
  within 1e-9 of the switch carries the warning `POLARIZATION_BASIS_SWITCH` (another runtime's last-bit difference could
  pick the other reference and turn that mode's `H` / `V` about the beam).
* **Reference planes**: input phase 0 is the source's emission point; an output's phase is taken at the beam's chief-ray
  crossing of its plane. A phase on a whole row or column is a choice of reference plane: photon statistics, `|Tᵢⱼ|`,
  permanents and loop phases `arg(Tᵢⱼ Tₖₗ Tᵢₗ* Tₖⱼ*)` do not depend on it.
* **Reflection models**: a mirror reflects as an ideal metal, `(r_s, r_p) = (−√R_s, +√R_p)` in the `(s, d_out × s)` triad
  (the vector law `E_out = −(I − 2nnᵀ)E_in`): `diag(+1, −1)` on `(H, V)` at a fold in the horizontal plane. A beam splitter
  has `t = √T`, `r_s = i√R_s`, `r_p = −i√R_p` on both sides: a non-polarizing splitter reflects as a mirror times `−i√R`; a
  polarizing one scales the s and p parts of that reflection by `√R_s`, `√R_p`. Interfaces use the Fresnel amplitudes.
  A coating that states its phases (a Project coating's `phase`) uses them instead: `r = √R e^{iφ_r}`, `t = √T e^{iφ_t}`
  per polarization in the same triad, so the transfer matrix and the element matrices carry the stated phases (the note
  `REFLECTION_PHASE_CONVENTION` names those surfaces); catalog coatings publish powers, and their phases follow these
  models. At normal incidence there is no physical s axis: the export
  refuses a coating whose model is polarization-dependent there (`R_s ≠ R_p` or `T_s ≠ T_p` at 0°, such as a constant PBS
  coating) met within 1° of it (`POLARIZING_NORMAL_INCIDENCE`); coating data with `R_s = R_p` at 0° are kept.
* **Modes**: beams on one output plane whose Gaussian modes coincide are one mode; beams that do not overlap are separate
  modes; beams crossing a plane in opposite directions are separate modes.

The network is for one wavelength: `T` is exact there, and the path list lets you re-phase each path for a nearby
wavelength from its optical path length. The Gouy phase, group-delay dispersion (pulse broadening), scattering,
diffraction, nonlinear and active elements are not modelled. Ideal and tabulated thin lenses, neutral-density attenuators,
Faraday rotators and diffraction gratings state no phase, so a layout whose light passes them is refused (`PHASE_UNDEFINED_IDEAL_LENS`,
`PHASE_UNDEFINED_ATTENUATOR`, `PHASE_UNDEFINED_FARADAY_ROTATOR`, `PHASE_UNDEFINED_GRATING`).

**Group delay.** Each path also carries its group optical path `group_opl_mm` = Σ n_g × length over its chief-ray
segments from the emission point to the output plane, with `n_g = n − λ dn/dλ` of each segment's medium at the exported
wavelength from the same material model the trace uses (closed form for a Sellmeier glass, a ±0.01 nm difference of the
table for a tabulated index, `n_g = n` for air and constant indices: the note `GROUP_INDEX_CONSTANT_MEDIA` names the
media the exported paths cross whose model has a constant index), and `group_delay_ps = group_opl_mm / c`. It is
the arrival time of a wavepacket's envelope centred at that wavelength; the phase delay `opl_mm / c` is the carrier's.
For the temporal overlap of photons or pulses use the group delay: behind a 3 mm N-BK7 window at 780 nm the envelope is
delayed by 5.278 ps while the phase delay is 5.115 ps, and compensating by the phase delay would leave 0.163 ps.
`hints.path_pairs[].group_delay_ps` gives the difference for every two paths that reach one output beam. Both are
quantized to 1e-9 (mm, ps) and enter the contract hash.

**Clear apertures.** A beam an aperture cuts keeps its Gaussian mode in the Engine, its amplitude scaled by `√P` (`P` the
power passed); the part of the cut field that is still in the mode is `P`, not `√P`. A cut `1 − P` up to the Engine's
warning fraction (1e-3) is exported as loss, so an entry of `T` can be off by up to `(1 − P)/2 ≤ 5e-4` relative; a larger
cut of a beam that reaches an output is refused (`MODE_TRUNCATED`).

## Precision and the contract hash

Before any phase is formed, path amplitudes are rounded to 1e-12, each output beam's reference optical path length to
1e-9 mm and each path's optical path length relative to that reference to 1e-12 mm; every phase is reduced modulo one
wave in exact decimal arithmetic; transfer-matrix entries are rounded to 1e-9. So:

* the relative phase of two paths reaching the same output beam (what decides interference, `|Tᵢⱼ|`, `TᴴT`, permanents and
  closed-loop phases) carries at most `k₀ · 5e-13 mm` of export rounding (5e-9 rad at 633 nm), on top of the Engine's own
  optical path lengths (float64, about 1e-13 relative);
* the phase common to an output beam's two rows carries at most `k₀ · 5e-10 mm` (5e-6 rad at 633 nm): a choice of
  reference plane that no gauge-invariant quantity sees;
* `transfer_matrix.lossless_tolerance` is 1e-6, or the bound these roundings put on the entries of `TᴴT` when that is
  larger (very large networks); a lossless layout is `lossless: true`, and a singular value above `1 + lossless_tolerance`
  (gain) is refused (`NETWORK_NOT_PASSIVE`), as is a source whose paths lose no light while `|T J|²` delivers less or more
  than all of it.

Passivity is also checked element by element, so it does not depend on arm lengths. A beam splitter, and a refracting
interface whose coating states phases and whose reflection is traced, applies the same stated amplitudes to light from
either side: per polarization its two-port matrix is `[[r, t], [t, r]]`, passive only when
`R + T + 2 √(R T) |cos(φ_r − φ_t)| ≤ 1` at the beam's angle of incidence (a lossless splitter needs `φ_r − φ_t = ±π/2`).
An interface whose reflection keeps the bare Fresnel phase (its sign changes with the side) needs
`R + T + 2 √(R T) |sin(φ_r − φ_t)| ≤ 1`. An element outside these bounds is refused (`NETWORK_NOT_PASSIVE` at its
surface).

Every other number carries a stated resolution, listed in `conventions.quantization`: positions, beam and waist radii and
optical path differences 1e-9 mm, delays 1e-9 ps; unit vectors 1e-12; element angles of incidence (deg) and Gaussian waist
distances (mm) 1e-6; refractive indices, M² and OPD / coherence-length ratios 1e-9; power fractions, singular values and
check values 1e-9; element matrices 1e-9; `wavelength_nm`, `line_power_W` and `coherence_length_mm` 12 significant digits.

The hash is the same in the Python Engine and in the browser runtime when their floating-point results agree before this
rounding. The tracer and the export use `cos`, `sin`, `exp`, `log` and `sqrt` of the platform's math library, which can
differ in the last bit between runtimes; rounding to the stated resolutions makes a different document unlikely (a value
would have to lie within a last-bit difference of a rounding boundary) but not impossible. The test suite exports the
reference layouts in both runtimes and compares the documents byte for byte.

## What the classical check validates

For every source and every output plane the check compares `Σ|T J|²` over the plane's output modes with the Engine's own
coherent superposition of the same traced beams (a closed-form whole-plane integral of the Gaussian beams with their
tilts, offsets and curvatures), for the source's declared polarization and for lab `H` and `V` input, each from its own
trace. A continuous spectrum's line is traced alone in these traces (one frequency, as the network). It validates the
export: the grouping of beams into modes, the projections on the lab bases, the assembly of the phases and the
quantization. The allowance is `1e-6 ×` the power, plus the quantization bound, plus the cross terms the
network leaves out between beams it calls separate modes (their overlap is at most `mode_overlap_tolerance`). Both sides
come from the same trace, so the check does not validate the Engine's physics: an error in a Jones model, a coating phase
or an optical path length would be in both. The reflection, refraction and polarization models are tested against
independent textbook references instead (`tests/test_network_export.py`, `tests/test_network_accuracy_and_scope.py`,
`tests/test_splitter_reflection_convention.py`).

## When a layout is not exported

The export returns a complete network or a list of problems, never a partial network.

| Code | What it means | What to do |
|---|---|---|
| `MIXED_WAVELENGTHS` | The sources emit several wavelengths. | Choose one with `wavelength_nm`. |
| `WAVELENGTH_NOT_EMITTED` | No source emits the chosen wavelength. | Use a wavelength the sources emit. |
| `OUTPUT_NOT_TERMINAL` | A beam crosses an output plane and then meets further optics. | Move the plane after the last element, or leave it out of `observation_ids`. |
| `OUTSIDE_ACCEPTANCE` | A beam crosses an output plane outside the plane's acceptance cone, so the plane does not count it. | Align the plane with the beam or widen its acceptance. |
| `PARTIAL_MODE_OVERLAP` | Two beams on one output plane partly overlap (a tilt, an offset, a different size or curvature). | Align them, or separate them. |
| `CLASSICAL_LIMIT_MISMATCH` | `Σ\|T J\|²` disagrees with the Engine's coherent superposition on a plane. | Align the beams; keep the default `mode_overlap_tolerance`. |
| `CAVITY_NOT_SUPPORTED` | Light circulates in a resonator. | Export the layout without the cavity. |
| `TRACE_INCOMPLETE` / `NETWORK_NOT_CLOSED` | The trace could not follow every beam to its end. | Reduce what multiplies beam paths (surfaces reflecting back and forth). |
| `PHASE_UNDEFINED_IDEAL_LENS` | A path crosses an ideal or tabulated thin lens, whose phase is not modelled. | Use a lens with a real prescription. |
| `PHASE_UNDEFINED_GRATING` | A path crosses a diffraction grating, whose model states its orders' directions and at most their powers, not their phase. | Export the layout without it, or read its directions and powers from a run. |
| `PHASE_UNDEFINED_FARADAY_ROTATOR` | A path crosses a Faraday rotator, whose model states a rotation and a transmittance, not the optical path through its body. | Export the layout without it, or read its powers and polarization from a run. |
| `PHASE_UNDEFINED_ATTENUATOR` | A path crosses or is reflected by a neutral-density attenuator, a thin plane whose model states powers only (no optical path through its substrate, no reflection phase). | Model the filter as a plate of its substrate with a coating, or export the layout without it. |
| `MODE_UNDEFINED` | A beam is not a valid Gaussian mode (aberrations, high NA). | Reduce the beam or the aberration. |
| `MODE_TRUNCATED` | An aperture cuts more than 1e-3 of the power of a beam that reaches an output, or a detector's active area cuts an output beam. | Enlarge the aperture or the detector area, or reduce the beam. |
| `POLARIZING_NORMAL_INCIDENCE` | A coating whose model is polarization-dependent at normal incidence (`R_s ≠ R_p` or `T_s ≠ T_p` at 0°, such as a constant PBS coating) is met within 1° of normal incidence, where which field component it treats as s is set by a misalignment or the Engine's internal axis. | Tilt it to its design angle of incidence, or use a non-polarizing element. |
| `NETWORK_NOT_PASSIVE` | An element's stated amplitudes would create light for light arriving from both sides (for example a splitter's `reflection_phase_rad` other than ±π/2 with `R + T = 1`, or a lossless coating whose stated reflection and transmission phases do not differ by ±π/2), a singular value of `T` exceeds `1 + lossless_tolerance`, or a source whose paths lose no light keeps less or more than all of it in `T`. | Fix the declared phase or coating. |
| `OUT_OF_MODEL_SCOPE` | The exported light uses an element outside its published model (wavelength, angle of incidence), with the declared polarization or with lab `H` or `V` input. Only the exported light is checked: the selected sources (`source_ids`), each at the exported line (`wavelength_nm`); a source or line the export leaves out never refuses it, and a violation whose light reaches only planes the export leaves out is listed in `validity.warnings`. A uniaxial crystal on the path of exported light is refused too (the network has no phase model of its split). | Use the element inside its characterized range, or export a selection that does. |
| `OPTIONS_INVALID`, `SOURCES_MISSING`, `OUTPUTS_MISSING` | The options or the Project do not name what to export. | Fix the option or add the source / observation plane. |
| `ENGINE_INPUT_INVALID` | The Engine cannot build the scene of the input (an invalid layout). | Fix the problem the message names (`check_project` names it too). |
| `NUMERICAL_FAILURE` | An arithmetic, value or linear-algebra error while computing the network (non-finite or extreme numbers). | Check the layout's numbers (positions, radii, powers) for extreme magnitudes. |

## Loading the network into a simulator

The snippets below are examples; OpenSci does not depend on these packages.

Two-photon coincidences with numpy (Hong–Ou–Mandel): one photon in each of two input modes, the probability of one photon
in each of two output modes is `|perm(T_sub)|²`.

```python
import json
import numpy as np

net = json.load(open("network.json", encoding="utf-8"))
tm = net["transfer_matrix"]
T = np.array([[complex(re, im) for re, im in row] for row in tm["entries"]])
col = {m: j for j, m in enumerate(tm["columns"])}
row = {m: i for i, m in enumerate(tm["rows"])}

i1, i2 = row["out:D1:0:H"], row["out:D2:0:H"]
j1, j2 = col["in:A#0:H"], col["in:B#0:H"]
sub = T[np.ix_([i1, i2], [j1, j2])]
coincidence = abs(sub[0, 0] * sub[1, 1] + sub[0, 1] * sub[1, 0]) ** 2   # 0 for a balanced splitter
```

Orientation: Perceval 1.3.1 `Unitary(Matrix(U))` and Piquasso 8.0.1 `Interferometer(U)` both use `U[out, in]` (a single
photon in input mode 0 leaves with the probabilities `|U[:, 0]|²`), the same orientation as OpenSci's `T[row = output][column
= input]`: pass `T` as it is. With Perceval, a lossless square network is a unitary component:

```python
import perceval as pcvl

assert tm["lossless"] and T.shape[0] == T.shape[1]
simulator = pcvl.Simulator(pcvl.BackendFactory.get_backend("SLOS"))
simulator.set_circuit(pcvl.Unitary(pcvl.Matrix(T)))
occupation = ["0"] * T.shape[1]
occupation[j1] = occupation[j2] = "1"                   # one photon in each of the two input modes
inputs = pcvl.SVDistribution({pcvl.StateVector(pcvl.BasicState("|" + ",".join(occupation) + ">")): 1.0})
print(simulator.probs_svd(inputs)["results"])
```

A lossy network (`lossless` false) is a sub-block of a larger unitary: add loss modes, for example by the unitary dilation
`U = [[T, (I − T Tᴴ)^½], [(I − Tᴴ T)^½, −Tᴴ]]`, before giving it to a simulator that needs a unitary, and sum the
detection events over the output modes of each detector. A non-square `T` (sources only on some input ports) is
completed to a unitary the same way; any completion of the vacuum columns gives the same statistics.

To include the photons' timing, read `paths[].group_delay_ps` (or `hints.path_pairs[].group_delay_ps` for two paths to one
output beam): two photons are indistinguishable at an output only when their envelope delays differ by much less than
their coherence time. Partial distinguishability, the wavepacket shape, g², detector efficiency and dark counts are inputs
you give the simulator (in Perceval, noise-tagged input states and its source and noise models).
