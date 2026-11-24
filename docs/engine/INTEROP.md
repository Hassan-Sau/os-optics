# External scientific tools with OpenSci

OpenSci designs and simulates the free-space optical system; many questions around it are answered best by other
scientific tools: an optimizer searching lens positions, a wave-optics code computing a PSF behind a stop, a quantum
simulator evaluating multi-photon interference, a full-wave solver for a metasurface or a grating, a CAD kernel building
exact solids. The **interop capability catalog** states, for each such workflow, which tools fit, what OpenSci provides
and what the tool computes, which OpenSci data and commands to use, what only you can provide, the assumptions and limits,
and how far OpenSci has checked the combination.

OpenSci does not ship, host or run these tools. You (or your own Agent) install them where you choose -- next to the
Engine, in a separate environment, under WSL or as commercial software under your own license -- and exchange data through
files or Python objects.

## Reading the catalog

```bash
os-optics interop list                        # every entry: id, status, title, tools
os-optics interop list --status VERIFIED_EXAMPLE
os-optics interop show quantum-photonics      # one entry in full
os-optics interop show cad-brep-geometry --json
```

```python
from os_optics import interop

catalog = interop.interop_catalog()                # schema, statuses, directions, entries
entry = interop.interop_entry("scalar-diffraction-imaging")
```

From a JSON host (e.g. a browser running Pyodide): `json_api.interop_catalog(entry_id=None)`; `json_api.capabilities()
["interop_catalog"]` names the schema version and the entry ids.

Each entry (`opensci.interop_catalog` 1.0) has:

| Field | Content |
|---|---|
| `scenario` | the question the workflow answers |
| `tools` | name, official link, role (recommended, alternative, cross-check, companion, historical), maintainer (upstream project, third-party wrapper, commercial vendor), distribution, maintenance (active, archived, legacy environment), the license as the upstream distribution reports it with its source and date, what the tool computes, notes |
| `opensci_provides` / `tool_computes` | the split of the work |
| `entry_points` | the exact OpenSci Python attributes, `json_api` functions and CLI commands the workflow uses |
| `data` | the OpenSci exports and documents it reads |
| `user_inputs` | what only you can give (quantum source and detector models, device structures, error models, licenses) |
| `assumptions`, `limits` | the scientific conditions, and what OpenSci refuses with which code |
| `conventions` | where units, frames, phase and normalization are stated |
| `runtime` | where and how the tool runs |
| `status`, `verified` | how far the combination is checked (below) |

## Status

| Status | Meaning |
|---|---|
| `GUIDANCE_ONLY` | The entry states which OpenSci data the tool needs, how to obtain it and what the tool adds. OpenSci has not executed this combination. |
| `DATA_CONTRACT_AVAILABLE` | OpenSci publishes a versioned export or data contract, with stated units, frames and conventions, that carries the data this workflow needs. |
| `VERIFIED_EXAMPLE` | A named example with pinned tool versions was executed and its result checked against an independent reference. |
| `PLANNED` | The OpenSci data this workflow needs is not published yet. |
| `DEPRECATED` | The tool is no longer maintained upstream; the entry is kept for reproducing earlier work and names its replacement. |

A `verified` record exists only for a check that was executed: tool and version, Python version, date, the case's
path in the public kit repository (os-optics-kit) and the commit it ran at, the Engine version, the independent reference and
the outcome. That a tool can do something, that it could be combined with OpenSci, and that OpenSci has verified the
combination are three different statements; the catalog keeps them apart.

## Directions

| Entry | Workflow | Status |
|---|---|---|
| `optimization-tolerance` | SciPy / Optuna / SALib (BoTorch): optimization, tolerancing, sensitivity over OpenSci runs | verified example |
| `scalar-diffraction-imaging` | LightPipes / prysm / POPPY (Diffractio): diffraction, PSF, spatial filtering from the handoff export | verified example |
| `vector-doe-inverse-design` | Diffractio, TorchOptics, Chromatix: DOE / SLM, vector focusing, inverse design | guidance |
| `cavity-interferometer` | Finesse 3: cavities and interferometers | guidance |
| `quantum-photonics` | Perceval (Piquasso cross-check; QuTiP, NetSquid companions) on the linear optical network | verified example |
| `quantum-photonics-strawberry-fields` | Strawberry Fields, archived upstream: reproducing earlier work | deprecated |
| `micro-nano-pic` | Meep, Tidy3D, grcwa / torcwa, gdsfactory + SAX, PhotonForge, Lumerical: local devices and PICs | guidance |
| `metasurface-virtual-element` | grcwa / meent / DFlat / metabox: a metalens response as a `TABULATED_THIN_LENS` model | verified example |
| `coatings-materials-polarization` | tmm, PyMoosh, refractiveindex.info, py_pol | guidance |
| `cad-brep-geometry` | NVIDIA USD BRep / SMLib (official) and smlibpy (third-party wrapper): exact solids of public geometry | guidance |
| `adaptive-optics`, `multiphysics-stray-light`, `nonlinear-ultrafast-communications`, `scattering-media`, `laboratory-instruments` | specialist topics | guidance |

## The data OpenSci provides

* **Projects and runs** (`ProjectDocumentV2`, `os_optics.local.run_optical_project`, `os-optics project run`): an
  external loop edits positions and declared model parameters, runs, and reads the results; every run is identified by
  `project_hash`, `engine_input_hash`, `result_hash` and `engine_build_id`. `retained_dofs` state the design intent for
  assembly, not optimizer variables.
* **Scientific results** (`os_optics.results`, [RESULTS_AND_EXPORT.md](RESULTS_AND_EXPORT.md)): beams at every
  monitor, paths, power budget, rays, geometric wavefronts, irradiance maps, and the metric descriptors with the result
  field that carries each target metric (`divergence_half_angle_x_rad` is `arrivals.far_field_half_angle_x_rad`).
* **Transverse-plane handoff** (`RunResults.handoff`, `os-optics results handoff RUN --monitor ID [--element ID]`):
  the beams at a plane in its (u, v, n) frame -- q per axis, waist, curvature, M², Jones vector and basis, power, medium --
  with the TEM00 field-construction convention, each beam's validity, and a named element's clear apertures in the same
  frame.
* **Linear optical network** (`opensci-linear-optical-network/v1`, [LINEAR_OPTICAL_NETWORK.md](LINEAR_OPTICAL_NETWORK.md)):
  modes, the transfer matrix `T[out, in]`, per-path optical path, group delay and Jones amplitudes.
* **Paraxial analysis** (`os-optics project paraxial`, `os_optics.local.paraxial_analysis`): thick-lens and mirror ABCD
  (folded axes unfolded, compound units composed), focal distances, principal planes, afocal and collimation spacing -- a
  closed-form starting point for an optimizer.
* **Cavity analysis** (`os-optics project cavity`, `os_optics.local.cavity_analysis`): stability, the fundamental Gaussian
  eigenmode, Gouy phase, free spectral range, transverse-mode spacing, finesse, linewidth and mode matching of a linear cavity, and
  the steady-state build-up of light injected through its first mirror (per Hermite-Gauss mode of a coaxial input beam), as an
  analysis document -- the cavity parameters a full interferometer model starts from.
* **Public geometry** (`os-optics catalog show`, `project model-from-catalog`, `json_api.optical_section`):
  prescriptions, apertures, frames and poses for CAD.

## Scientific conditions

* Intensity or power alone does not determine a complex field. OpenSci exports beam parameters; a sampled field built
  from them is your construction under the stated convention, valid for a TEM00 beam.
* Sparse pupil rays are samples of the geometric wavefront, not a sampled pupil field: fit modes of a stated order.
* The Engine's Gaussian and geometric results contain no diffraction at hard apertures. After `GAUSSIAN_TRUNCATION` the
  transmitted power is exact while spot size and irradiance are those of the untruncated Gaussian;
  `GAUSSIAN_SEVERE_TRUNCATION` withholds the Gaussian state.
* M² > 1 alone does not determine the modal content or the coherence of a beam.
* Power transmittance and amplitude transmission are different quantities: `t = sqrt(T) exp(i phi)` needs the phase.
* Quantum source and detector properties are your inputs; they are never inferred from a classical beam.
* A full-wave solver is for a local device; a mode-port S-matrix and a sampled free-space field are different interfaces.
* One external output (a spot, a focal length, an efficiency for one input) does not define an element for other inputs:
  a model built from it holds only inside the domain it was validated on.
* External results stay in their own files next to the run they used, naming its hashes. They never change the run's
  hashes or its Local Validation, and they are not a commercial acceptance.

## Examples

An entry's `verified` records name the executed cases in the public kit repository
[os-optics-kit](https://github.com/JunLin615/os-optics-kit) (`cases/interop-*`; `python -m pip install os-optics-kit`,
`os-optics-kit skills list`): each runs in its own environment with its own requirements file, with pinned tool
versions and an independent reference. A record's `companion_commit` is the commit of that repository's history at which
the case was executed (the case then lived in the interop examples directory). The Engine has no dependency on any of these
tools.
