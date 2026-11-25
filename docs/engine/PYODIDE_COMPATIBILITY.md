# os-optics — Pyodide Compatibility

Target: **Python 3.14.x** and **Pyodide 314.0.x** (pinned to 314.0.7 in CI;
Python 3.14.2, `platform=emscripten_5_0_3`, `abi=2026_0`, arch `wasm32`).

## 1. Rule

A core dependency is acceptable only if (a) the target Pyodide distribution
ships it, or (b) it is a `py3-none-any` wheel that installs *and runs* in the
real Pyodide (tested). `pip install` succeeding on a desktop proves nothing.

## 2. Findings from the Pyodide 314.0.7 lock file (`pyodide-lock.json`)

| Package | In Pyodide 314.0.7 | Version | Verdict |
|---|---|---|---|
| numpy | yes (wasm wheel) | 2.4.6 | core |
| scipy | yes (wasm wheel) | 1.18.0 | core |
| pydantic | yes (pure) | 2.12.5 | core |
| pydantic-core | yes (wasm wheel) | 2.41.5 | core (dependency of pydantic) |
| shapely | yes (wasm wheel) | 2.1.2 | core (layout only) |
| typing-extensions, annotated-types | yes | 4.15.0 / 0.7.0 | transitive |
| micropip | yes | 0.11.1 | used to install the engine wheel |
| networkx | yes, but declares `matplotlib`, `decorator`, `setuptools` deps | 3.6.1 | not adopted |
| trimesh | **no** | – | not adopted |
| rtree | **no** | – | not adopted (trimesh optional dep; unnecessary) |

Trimesh policy: not used. The Engine needs only analytic plane/sphere geometry,
implemented in NumPy. If trimesh is ever adopted it must pass the four gates
(native pin, Pyodide install, in-Pyodide ray/mesh smoke
test, no implicit `rtree`/Embree).

## 3. Coding constraints that keep the engine Pyodide-safe

* No threads/multiprocessing, no `subprocess`, no sockets, no
  `concurrent.futures` pools.
* No filesystem access while computing (the engine build id reads its own
  package `.py` files, read-only, and degrades to a version-derived id if the
  package is zipped).
* No `numpy.random.Generator` distributions (stream-stability), no
  `scipy.optimize` C-extension features beyond what Pyodide's SciPy ships
  (none required: LM is implemented in NumPy).
* Only `numpy`, `scipy.interpolate`, `pydantic`, `shapely` imports at top-level
  of the packages that need them (`layout/` is the only importer of shapely;
  `import os_optics` must succeed even if shapely were missing —
  layout stages then return `UNKNOWN` with `DEPENDENCY_UNAVAILABLE` rather than
  crash; this is tested by monkeypatching the import).
* Pyodide float behaviour is IEEE-754 binary64 (WASM), identical algorithms
  as native; differences arise from libm, NumPy's SIMD kernels, BLAS/LAPACK and
  the fused multiply-add a native NumPy uses in `einsum`, `matmul` and complex
  products. The Engine keeps them out of every reported quantity
  (NUMERICAL_CONVENTIONS §4), so a run's files are identical in both runtimes
  apart from the runtime record.

## 4. Automated tests (not documentation)

`tests/pyodide/run_parity.mjs` (Node, pinned `pyodide@314.0.7` from `tests/pyodide/package.json`) and
`tests/test_packaging_and_pyodide.py`:

1. build the pure-Python wheel (`python -m build --wheel`), assert it is `py3-none-any` with no binary members;
2. load Pyodide from the npm package; load `numpy scipy pydantic shapely micropip` from the **versioned** distribution URL
   `https://cdn.jsdelivr.net/pyodide/v314.0.7/full/` (never `latest`; wheels are cached under `node_modules`);
3. write the engine wheel into the Emscripten FS and `micropip.install("emfs:/tmp/<wheel>")` — this exercises the wheel's own
   `Requires-Dist` ranges against the Pyodide distribution;
4. `import os_optics`, read versions, run every `tests/fixtures/requests/*.json` through `validate_json`, time cold and warm runs;
5. pytest compares each result with the native result via `compare_results`: canonical input hash identical, statuses / target
   statuses / issue codes identical, floats within `cross_runtime_rel_tol` / `cross_runtime_abs_tol`;
   the local assembly layout / collision check (`os_optics.local.json_api.validate_assembly_package`) runs on the
   reference Assembly Preview Package (`tests/fixtures/assembly_preview/`), given as bytes, as a JS `Uint8Array` and as a
   file map: `checker.runtime_kind` is `BROWSER_PYTHON` and the result equals the native one exactly (every float bit for
   bit) apart from `runtime_kind` and the `result_hash` that covers it;
6. `test_clean_environment_install_from_the_wheel_needs_no_compiler` installs the wheel into a fresh venv with
   `pip install --only-binary=:all:` and runs the 405 nm / 500 nm case through the CLI.

Prerequisites (Node >= 20, network on a cold cache) missing means the test **fails** with an explicit message; only
`OPENSCI_SKIP_PYODIDE=1` skips them, and CI must not set it.
