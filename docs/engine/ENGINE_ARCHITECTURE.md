# os-optics — Architecture

Status: describes the Engine at the contract version `SCHEMA_VERSION` of `src/os_optics/_version.py`; a change that moves a
boundary stated here updates it in the same change. This document is normative for module
boundaries, public contracts and dependency choices. Physics scope is in
`PHYSICS_SCOPE.md`; numeric conventions in `NUMERICAL_CONVENTIONS.md`; Pyodide
constraints in `PYODIDE_COMPATIBILITY.md`.

## 1. What the engine is

`os-optics` is a **deterministic, offline engineering computation library**. One evaluation is a pure function of its
request and of the `EvaluationState` the caller holds (section 7); without a state it is one call:

```
ValidationRequest (Pydantic, JSON)  ->  validate()  ->  ValidationResult (Pydantic, JSON)
```

It is not a database, web API, cloud service, supplier resolver, catalog or
file store. It performs no I/O of any kind while computing (no network, no
filesystem reads of user data, no environment probing other than the
Python/NumPy versions recorded in the manifest, no clocks, no global
mutable caches, no multiprocessing).

Invariant: same canonical input + same engine version + same
`NumericalConfig` = same result within the declared numerical tolerance.

## 2. Layering

```
 JSON / Python objects
        |
        v
 contracts/   Pydantic v2, frozen, extra="forbid", JSON friendly   <-- public boundary
        |  (validated, then lowered once)
        v
 scene/       lowering: contracts -> numpy-backed internal structures (dataclasses)
        |
        v
 math/ optics/ layout/ control/   pure numerical code, NumPy arrays / dataclasses, no Pydantic in loops
        |
        v
 validation/  targets, aggregation, issue synthesis   -> Pydantic results
        |
        v
 manifest/    canonical serialization, hashing
```

Rules:

* Pydantic objects are created only at the boundary (input validation and
  result assembly), never inside ray/beam/optimizer inner loops.
* `math/` depends on NumPy only. `optics/`, `layout/`, `control/` depend on
  `math/`, `config` and `errors/`, never on `contracts/` request/result types
  except through `scene/` lowering.
* `contracts/` never import `optics/` etc. (one-way dependency), so contracts
  can be frozen independently of algorithms.
* No module reads a module-level mutable. All tolerances travel in an explicit
  `NumericalConfig` object (frozen, hashed into the manifest).

## 3. Package layout

```
src/os_optics/
  __init__.py          public API: validate, validate_json, __version__, contracts re-exports
  _version.py          ENGINE_VERSION, SCHEMA_VERSION, NUMERICAL_CONFIG_VERSION, MODEL_VERSIONS
  config.py            NumericalConfig (all epsilons/tolerances, each documented)
  errors/              Status, ErrorCode, ValidationIssue, EngineInputError
  math/                vectors, rotation (quaternion), pose, rng (PCG64 -> uniform/normal)
  contracts/           frozen public models (see 4)
  scene/               contracts -> internal lowering
  optics/
    materials/         constant / Sellmeier / tabulated n(lambda), coatings, range checks
    surfaces/          plane & sphere intersection, aperture, Snell/reflect/TIR, thin lens
    polarization/      Jones calculus, transverse-basis transport, Fresnel, elements
    gaussian/          q-parameter / ABCD (x,y independent), truncation, applicability
    rays/              vectorized ray bundles, sequential-agnostic (non-sequential) tracer, pupil sampling
    interference/      two-beam analytic model
  layout/              footprint, keepout, beam envelope, moving envelope (Shapely, 2D/2.5D only)
  control/             actuators, sensors, Jacobian, Levenberg-Marquardt, recoverability, Monte Carlo
  validation/          target metric registry, out-of-scope registry, pipeline, ray_paths (chief / pupil ray export)
  evaluation/          the capability table, the evaluation plan and EvaluationState (composable evaluation, section 7)
  manifest/            canonical JSON, hashing, ValidationManifest assembly
  catalog/             the local public optical catalog store (verified releases, state.json) and its synchronization with
                       the catalog service (manifest + conditional GET, sha256-verified downloads, heartbeat, import);
                       the only place besides client/ with network access (remote.py), never inside Pyodide
  local/               local design workflow: catalog resolution (explicit > installed in memory > local store, pinned
                       release first), ProjectDocumentV2 -> Engine lowering, compile.py (the one optical model -> Engine
                       element compiler: catalog products through a catalog resolver, Project-declared models through a
                       Project resolver, compound units flattened into one component), declared.py (catalog -> declared
                       model copies), model validity guards, LocalValidationRecordV1, run_optical_project (the stages of
                       run.py composed by evaluate.py: EvaluationSession, selective evaluation), run
                       directory (artifacts.py),
                       json_api.py (JSON-in/JSON-out entry points for non-Python hosts, e.g. a browser's Pyodide worker),
                       prepared.py (the JSON API's prepared evaluation handles: EvaluationSession objects kept by handle,
                       bounded, least recently used evicted),
                       submission.py (the transport-free commercial submission body + idempotency keys; re-exported by client/)
  results/             the scientific result layer of a run (load_run -> RunResults): monitors, arrivals, paths, power
                       budget, rays, geometric wavefront, irradiance map, metric descriptors; CSV / JSONL / JSON / NPZ
                       export built in memory (docs/engine/RESULTS_AND_EXPORT.md)
  optical_view/        Optical Simulation View: scene builder over one run's result + ray paths, GLB writer (no I/O)
  network/             linear optical network export (opensci-linear-optical-network/v1, contracts/network.py): per source two
                       lab-polarized probe traces of the same tracer, output-mode clustering by Gaussian mode overlap, transfer
                       matrix from quantized path terms, element Jones matrices (TraceObserver.branch_spawned), classical-limit
                       check against the coherent superposition (docs/engine/LINEAR_OPTICAL_NETWORK.md); local/network.py lowers a
                       Project for it
  assembly/            local assembly layout / gross-collision check of a verified Assembly Preview Package (stage six):
                       GLB -> closed welded meshes, winding-number ray-column overlap, interface engagement zones,
                       AssemblyLocalValidationV1 (pure Python floats, no NumPy: bit-identical native / Pyodide)
  _vendor/             the vendored contract bundle shared with the OpenSci service (design, catalog, optical
                       view, assembly preview package + local validation result, catalog release manifest); no catalog data
  cli.py               `os-optics catalog|project|assembly|schema|validate|compare`
  cli_results.py       `os-optics results ...` / `os-optics export ...` (stdio and the export file only)
  cli_network.py       `os-optics project network PROJECT` (the network JSON on stdout or --out)
  cli_evaluate.py      `os-optics project evaluate PROJECT [--outputs ...]` / `--describe [ID]` (json_api.evaluate_project)
```

## 4. Public contracts

### 4.1 Frozen (physics/math layer)

`Vector3`, `Quaternion`, `Pose3D`, `Ray`, `RaySegment`, `OpticalSurface`
(plane, sphere; apertures; interactions), `OpticalPort`, `OpticalPath`,
`BeamState`, `GaussianBeamState`, `JonesVector`, `PolarizationBasis`,
`MaterialOpticalModel` (+ `CoatingModel`), `Target`, `Observation`,
`ValidationResult`, `ValidationIssue`, `ValidationManifest`, `NumericalConfig`.

"Frozen" means: field names and semantics may only change via a
`SCHEMA_VERSION` bump plus migration. This is enforced: `tests/fixtures/contract_freeze.json` stores the JSON-Schema hash of
each frozen model and `tests/test_fixtures_cli_freeze.py::test_frozen_contracts_are_unchanged_since_the_recorded_baseline`
fails on any change without a version bump (`python tests/generate_fixtures.py --freeze` rewrites the baseline). The version
moves once per release: the first change after a published release bumps it, and further changes before the next release
re-record the baseline within that unreleased version (`--freeze --unreleased`); a published version's contracts never
change in place.

### 4.2 Engine-side input snapshots

`EngineeringComponentSnapshot`, `EngineeringEnvelopeSnapshot`,
`ActuatorSpec`, `ControlSpec`, `SourceSpec` (wrapper fields other than the
physical beam parameters), `ProjectSnapshot` (container fields such as ids,
names, revision metadata), `ValidationRequest.project` envelope. An envelope's footprints are stated in its
plan-parallel frame (`EngineeringEnvelopeSnapshot.frame_pose`); an element's `mount_margin_mm` is the width of the
opaque mount ring around its clear aperture (rays beyond it pass by).

The engine never imports an ORM or `DesignPart`/`SupplierPart`/`AssemblyImplementation`; data reaches it as these
snapshots (`Database model -> adapter -> Engineering*Snapshot -> engine`).

## 5. Dependency decisions

| Package | Decision | Reason |
|---|---|---|
| numpy | required, `>=2.3,<3` | vector math; shipped by Pyodide 314 (2.4.6) |
| scipy | required, `>=1.16,<2` | interpolation (`PchipInterpolator`, `RegularGridInterpolator`); shipped by Pyodide (1.18.0) |
| pydantic | required, `>=2.11,<3` | public contracts; Pyodide ships 2.12.5 + pydantic-core wasm |
| shapely | required, `>=2.1,<3` | *only* `layout/` footprint / keepout polygon operations; Pyodide ships 2.1.2 |
| networkx | not adopted | Pyodide's build depends on matplotlib; graph needs here are trees |
| trimesh | not adopted | not in the Pyodide lock; the Engine's surfaces (analytic, and planar facets stated as vertices) are intersected with NumPy, so no Rtree/Embree question arises |

Shapely does 2D/2.5D footprint booleans, distances and buffering. It does
not replace 3D physics: beam/ray propagation, apertures and surface
intersection are NumPy.

## 6. Runtime targets

Native CPython 3.11 to 3.14 (`requires-python = ">=3.11"`, `MIN_PYTHON` in
`src/os_optics/_version.py`), Pyodide 314.0.x in a Web Worker, CI,
controlled server verification. Results do not depend on the interpreter
version: every run artifact, golden case, determinism hash and drawing is
identical on 3.11, 3.12, 3.13 and 3.14 (only the descriptive `runtime` record
differs); CI runs the suite on 3.11 with the exact dependency floors and on
3.14. The package is pure Python (`py3-none-any` wheel); all binary
dependencies come from the target's own distribution (PyPI wheels / Pyodide
lock). No compiler, CMake, CUDA or system shared library.

## 7. Statefulness

None in the Engine. `validate()` is a pure function of `(request)`. Caches (e.g., material
index lookups inside one trace) live for one call. There are no module-level
caches or mutable globals in the Engine. `engine_build_id` is recomputed per call from the
installed package's `.py` source bytes (read-only) and is used only for
the manifest; it cannot influence physics. Reuse across calls is an explicit object the
caller holds: an `EvaluationState` (below) keeps each capability's product and the build
id; a local `EvaluationSession` keeps that state, the catalogs it loaded and the compiled
elements of the latest Project it lowered. The local workflow's catalog store memoizes the parsed releases it reads
(`local/catalog.py`, at most four) and holds the releases a host installs in memory.

Composable evaluation. An evaluation is a plan over named capabilities, one constant
table (`os_optics/evaluation/capabilities.py`), each with one implementation:

| Capability | Level | Requires | Phases | Implementation |
|---|---|---|---|---|
| `solid_check` | local | -- | lower | `local/lowering.py` `solid_check` |
| `scene` | engine | -- | (in trace) | `validation/pipeline.py` `_scene_node` |
| `beam_trace` | engine | scene | trace | `_beam_trace_node` |
| `secondary_rays` | engine | beam_trace | emitters | `_secondary_rays_node` |
| `plane_readings` | engine | beam_trace, secondary_rays; in a Project evaluation also model_scope | observation_results | `_plane_readings_node` |
| `power_ledger` | engine | plane_readings | -- | `_power_ledger_node` |
| `untraced_light` | engine | beam_trace, secondary_rays | untraced_light | `_untraced_light_node` |
| `plane_bounds` | engine | plane_readings, power_ledger, untraced_light | bounds, return_analysis | `_plane_bounds_node` |
| `surface_exposures` | engine | power_ledger; untraced_light for a design that states a power rating | exposures, return_analysis | `_surface_exposures_node` |
| `beam_pairs` | engine | plane_readings | interference | `_beam_pairs_node` |
| `layout` / `control` | engine (an Engine request only) | -- (flags `run_layout` / `run_control`) | -- | `_layout_node` / `_control_node` |
| `requirements` | engine | per target, `MetricDescriptor.requires`; in a Project evaluation also model_scope | targets | `_requirements` |
| `ray_paths` | local | beam_trace | ray_paths | `local/run.py` `traced_evaluation` (`validation/ray_paths.py`) |
| `model_scope` | local | ray_paths, plane_readings | model_scope | `local/run.py` `scope_findings` |
| `record` | local | solid_check, model_scope, the request's complete scope, every declared target | record | `local/run.py` `run_record` |
| `view` | local | ray_paths, plane_readings | scene | `local/run.py` `run_view` |

The run directory is no capability of an evaluation (an evaluation writes nothing): the table's
`EXPORTS` row `run_files` (`local/artifacts.py`, phase write) says how it is written --
`Evaluation.write_run(dir)`, `run_optical_project(project, output_dir=...)` or
`os-optics project run --out-dir`. `layout` and `control` run from an Engine request
(`os_optics.evaluate(request, capabilities=["layout"])`); a Project evaluation neither lists nor
accepts them.

`plan_evaluation` adds the requirements of what is requested (and of each selected
target's metric) and refuses a selection it cannot run: an excluded capability that something
requested requires (`EXCLUDED_CAPABILITY_REQUIRED`, naming what requires it and the preset the
caller wrote), and the record with a selection of targets that leaves a declared target out
(`TARGET_SELECTION_INCOMPLETE`). `validate(request)` is `evaluate(request)`: the complete plan of
the same nodes; `os_optics.evaluate(request, capabilities=..., state=...)` runs a narrowed one,
whose result names its scope (`ValidationResult.evaluation`: requested, executed, not computed
with what each produces, the declared targets not selected) and states what did not run as not
computed (a plane without the plane bounds: `quantity_state` `NOT_COMPUTED`, `untraced_power_W`
null). Every target it is asked to assess has a result. A target on a stage the request itself
turns off reads as in `validate`; a target whose verdict a capability outside the evaluation would
decide is `UNKNOWN` with issue code `CAPABILITY_NOT_COMPUTED`, naming the capability to add. The local
run is the same composition one level up: `run_optical_project(project)` is
`EvaluationSession().evaluate(project, outputs=("complete",)).run()` -- the stages of
`local/run.py` (`prepare_project`, `traced_evaluation`, `scope_findings`, `run_record`,
`run_view`, `bound_run`) composed by `local/evaluate.py`; a `readings` evaluation runs
the stages its plan names and enters no other phase. An `EvaluationState` binds its
products to the physical identity (build, the design's physical inputs, numerical
configuration, schema version): another identity drops them, a changed target threshold
reuses them, and an evaluation over a state equals a fresh one of the same scope.
`os-optics project evaluate --describe` and `json_api.evaluation_capabilities` publish
the table.

Ray paths. `validate_with_ray_paths(request)` runs `validate` once with a
read-only `TraceObserver` (`optics/rays/tracer.py`) attached to the tracer. The
tracer calls it when a packet starts (source or branch point), after it is
advanced to a surface, after each interaction and when its path ends; the
observer copies positions, directions and alive masks and never writes back.
With no observer the tracer is unchanged. The resulting `RayPathTrace` is bound
to the run by the manifest's `input_hash` / `trace_hash`.

Result layer. A local run writes its ray paths as `ray_data.json`
(`ray_paths_document`, run manifest 1.1, checked by `verify_run_artifacts`).
`os_optics.results` reads a run directory (or its files in memory, or a bare
Engine result, or an evaluation held in memory: `Evaluation.results()`) and derives its
tables from the recorded results, so a table and its exports are a pure function of the
run's bytes. The wavefront and irradiance map are analysis over stored data (geometric
OPD of the stored pupil rays; the paraxial Gaussian superposition of the stored
arrivals), not new physics in the tracer. The imaging, ghost and field analyses are
the exception: they trace the run's own request again with their own pupil or field
sampling (`results/imaging_trace.py`, `results/ghost_trace.py`, `results/field.py`). A
query of a part an evaluation did not compute is refused with `NOT_COMPUTED`. `json_api.run_results` exposes the same queries to
the browser runtime. See `docs/engine/RESULTS_AND_EXPORT.md`.

## 8. Extension points

* New element type: add a discriminated-union member in `contracts/optics.py`
  and an interaction handler in `optics/surfaces/interaction.py`.
* New Project-declared model kind: add it to the Project contract's `ProjectSimulationModelV1` (vendored
  `_vendor/shared_contracts/optical_design/optical_model.py`) and compile it in `local/compile.py` (`compile_model`) into existing Engine surfaces;
  lowering, validity, the Optical Simulation View and both runtimes then use it with no other physics path. Its roles
  (`local/lowering.py` `DECLARED_MODEL_ROLES`) and its scene body (`optical_view/builder.py` `declared_body`) follow.
* New metric: register in `validation/metrics.py`; unsupported physics goes
  into the explicit out-of-scope registry (returns `OUT_OF_MODEL_SCOPE`).
