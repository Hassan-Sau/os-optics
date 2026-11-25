# os-optics — development notes

Repository-only notes for working on the package source. The package's reference documentation is
`docs/engine/README.md` (also the PyPI long description) and the package's own docstrings / `--help`; the sdist ships
`src/os_optics`, `docs/engine/README.md`, `LICENSE` and `pyproject.toml`.

## Engineering documents (repository only)

* `docs/engine/ENGINE_ARCHITECTURE.md` — layers, contracts (frozen, engine-side input snapshots), dependency decisions, state
* `docs/engine/PHYSICS_SCOPE.md` — equations, assumptions, applicability, failure modes, independent references per module; P9 lists the metrics a run does not evaluate (generated)
* `docs/engine/NUMERICAL_CONVENTIONS.md` — units, frames, every tolerance and its reason, determinism, hashing
* `docs/engine/PYODIDE_COMPATIBILITY.md` — Pyodide policy and the automated tests
* `docs/engine/KNOWN_LIMITATIONS.md` — the model limits and their validity ranges

Every low-level example request is an executable fixture (`tests/fixtures/requests/*.json`, built by
`tests/fixture_scenes.py`).

## Working on the source

```bash
uv sync --extra test                     # any supported Python (3.11+); 3.14 is the Pyodide line
uv pip install build hatchling           # the install tests build the wheel with `python -m build --no-isolation`
uv run --no-sync pytest                  # native tests (golden physics, robustness, layout, control, determinism, hygiene)
python tests/generate_fixtures.py        # regenerate fixtures / snapshots
python benchmarks/bench_validation.py    # standard local-validation benchmark
python benchmarks/bench_scale.py core-625 --ray-data summary --out-dir run   # large-run time, memory and file sizes
python benchmarks/bench_evaluation.py --repeats 5 --analyses   # selective vs complete evaluation, candidate loop, CLI, memory
cd tests/pyodide && npm ci && cd ../..
pytest -m pyodide                        # real Pyodide install + native/Pyodide parity (needs node + network on a cold cache)
pytest -m install                        # build the wheel, install into a clean venv, run the local design workflow
```

## Value basis

The contract is `src/os_optics/contracts/basis.py` (types, carrier rules, `merge_bases`, the target rule `grade_point`),
the enums `ValueKind`, `OmittedPart` (with `ACCOUNTING_PARTS`) and `AssumedInput` in `src/os_optics/errors/status.py`,
the reader `src/os_optics/results/basis.py` (`basis_of`, the derivation table `DERIVATIONS`, the carriage registry
`CARRIAGE`) and the tests `tests/test_value_basis.py`. The reader-facing statement is in `RESULTS_AND_EXPORT.md`.

* **Attaching an entry.** A producer that reports a number that is not simply the model's value builds a `ValueBasis` and
  sets it on the record: `value_basis={"<field>": entry}` (`"*"` for every number of the record, a dotted name such as
  `beam.power_W` for a nested number); an analysis `Quantity` and a `results/quantity.py` dict take one entry
  (`value_basis=entry`). The carrier refuses a key that names no field or a null field, any entry on a `KNOWN_ZERO` /
  `NOT_APPLICABLE` quantity and a non-`MODEL` entry on a `KNOWN` one. An entry is written only when the kind is not
  `MODEL` or a caller setting other than the Engine default produced the number; a fact a flat field already holds is not
  stored again (`basis_of` derives it).
* **A target.** The metric's measurement (`MetricValue.basis`) carries the entry of the quantity it read; `_evaluate`
  copies it to `TargetResult.value_basis["value"]` and grades with `grade_point` (rule T1). No other code grades on a basis.
* **Derived quantities.** `merge_bases(*entries, monotone=...)` of exactly the inputs the producer reads: the union of the
  lists, the inputs' bound kind where the derived value is monotone in each and their kinds agree, `ESTIMATE` otherwise.
  Nothing propagates by itself.
* **Notices.** A finding whose whole content an entry or a flat field carries is issued with
  `os_optics.errors.notice(code, message, location=...)`: status `NOT_APPLICABLE`, located at the quantity, so it reaches
  no aggregate status or exit code; the entry names the same `code`.
* **Carriage.** Every projection that copies a quantity into a row (results queries, tables, run summary, JSON API) copies
  its entry with `row_bases` / `dump_with_bases` under `value_basis`, keyed by the row's own name of the number, or carries
  null for the number. A new projection is registered in `CARRIAGE` and gets its check in the carriage test.
* **A new enum member** is one physically different fact: add it with a one-sentence `#:` comment above it in
  `errors/status.py`, a row in the tables of `RESULTS_AND_EXPORT.md`, and, for an `OmittedPart`, decide whether it belongs
  in `ACCOUNTING_PARTS`. The contract digests move; re-record the contract freeze in the same change: while `SCHEMA_VERSION` (`_version.py`) is
  past the last published release, `python tests/generate_fixtures.py --freeze --unreleased`; otherwise bump
  `SCHEMA_VERSION` and run `python tests/generate_fixtures.py --freeze`.

## Findings and tests

How a finding of a wrong value is resolved, and what a review checks, in the Engine:

* **A finding that a reported value is wrong** is resolved in this order: (1) compute the value correctly; (2) report it
  qualified, with a `value_basis` entry that states its kind and what makes it a bound or an estimate (section "Value
  basis"); (3) only where no value can stand, withhold it with the stated physical reason, and enter the case in the
  [containment register](audit/CONTAINMENT_REGISTER.md) as an open row with the work that would close it.
* **The containment register** has one row for every tracer method that marks a beam's power as not the light's
  (`_Packet.power_scope`) and for every test of the three containment modules: the finding, the wrong value, what is
  reported today, the independent reference, the open work and the state. The change that fixes a row edits the row and
  rewrites its test in the same commit; a test whose row stays withheld carries `Register row T-xx, withheld: <reason>`
  in its docstring. `tests/test_containment_register.py` keeps the register, the tracer and the tests in step.
* **A test of a computed quantity** pins the number against an independent reference (a closed form, a second method,
  energy conservation) within a stated error, and pins its `value_basis` entry: the kind and the facts it lists.
* **A test of a refusal that stays** states the physical or integrity reason in its docstring and asserts the code, the
  location and the structured context, not the sentence.
* **Facts are fields, not sentences.** Every fact a client needs from an issue is a `numerical_context` key named with
  its unit; `os_optics.errors.issue_facts.ISSUE_FACTS` lists, per code, the keys every issue of it carries, and
  `tests/test_issue_facts.py` compares each with an independently computed value. A code a client words or routes on is
  listed there in the change that makes its producers carry the keys. A test asserts the code, the location and those
  keys; it asserts a message only to pin the one sentence template of a code. Existing substring assertions are
  converted when their test file is next changed, not in bulk.
* **A rewritten pin.** When a change of specification replaces a refusal, a withheld value or a status by a computed or
  qualified value, the tests that pinned the old behaviour are rewritten in the same commit as the behaviour: the scene
  and the independent reference stay, the new assertion pins the value and its entry, and the PR lists each one (test,
  old assertion, new assertion, the physical reason). That is a change of specification, not a weakened check; a changed
  numeric tolerance still needs its own stated reason.
* **What a test reads.** A new or re-recorded assertion reads the field that answers its question: `outcome.execution`
  (did the calculation complete), `assessment` (is the target met), `quantity_state` and the `value_basis` entry (is the
  number usable). The aggregate `ValidationResult.status` is a diagnostic summary: an existing assertion on it is
  re-recorded only where a change of specification moves it, and no new test decides anything from it.
* **A review** lists, beside every value or `PASS` that should not stand, the reference designs and examples that lost
  a value.

## Refusals

A refusal is one of three kinds, and `os_optics/errors/registry.py` records which each one is:

| Kind | When | What the caller gets |
|---|---|---|
| Whole operation | malformed input or input that violates a contract; an integrity or security check; an edit operation (atomic by design); catalog access and import parsing | a structured error with a code and a location, and no result |
| Per item | a valid request that contains a construct, a source, an element or an analysis quantity outside the model | the result, with an issue at that item, a state on the quantities that depend on it, and every other quantity computed |
| Resource limit | a budget or the run monitor ended part of the work | the result so far, `outcome.execution` `PARTIAL` or `STOPPED`, and the budget record |

The registry names every code vocabulary (`VOCABULARIES`) and every exception class (`EXCEPTIONS`) with its family and
kind, the code strings two vocabularies share on purpose (`ALIASES`), and the per-item refusals that still end their whole
operation (`PENDING`, with the per-item form each is to take). `tests/test_refusal_registry.py` fails when a vocabulary,
an exception class, a shared code string or an `EngineInputError` raise is not classified. A new refusal of a valid
request is written in its per-item form; when that is not yet possible it is added to `PENDING`.

## Supported Python

`requires-python` is `>=3.11` and `MIN_PYTHON` in `src/os_optics/_version.py` states the same minimum (a test keeps
them in step, `tests/test_python_support.py`). The floor is empirical: the whole non-Pyodide suite passes on 3.11, 3.12,
3.13 and 3.14, and every CLI output (catalog, all reference designs run with `--out-dir`, every low-level request,
assembly validation and drawing of every fixture, schemas) is byte-identical across them apart from the descriptive
`runtime` record. numpy 2.5 / scipy 1.18 need Python 3.12+, so `uv.lock` forks: 3.11 resolves numpy 2.4 / scipy 1.17.
The dependency floors (`numpy 2.3`, `scipy 1.16`, `pydantic 2.11.5`, `shapely 2.1`) pass the suite on 3.11
(`pydantic < 2.11.5` changes the generated JSON Schemas, which the contract-freeze test detects). CI runs the suite on
3.11 with exactly those floors and on 3.14 with the lock. When raising the minimum, change `requires-python`,
`MIN_PYTHON`, the classifiers and the CI matrix together; before lowering it, repeat the comparison above.

`os-optics doctor` (`src/os_optics/cli_doctor.py`) is the customer-side check of an installation.

## Release check

`scripts/public_install_smoke.py` exercises an installed `os-optics` without the repository: run it with the
interpreter of a fresh virtual environment into which the built wheel / sdist, or later the public PyPI release, was
installed, from a directory outside the checkout (`python -I scripts/public_install_smoke.py --catalog <release file>`
to import a catalog release offline, or without `--catalog` to synchronize it from the catalog service; pass
`--protected` when the `[protected]` extra is installed). It prints one JSON evidence summary and exits non-zero on any
failure.

`scripts/build_dist.py` builds the wheel and the sdist of a commit reproducibly (pinned build tools,
`SOURCE_DATE_EPOCH` from the commit: the same bytes on the same operating system and interpreter build) and writes their
identity (`dist/identity.json`: commit, tree, version, Engine build id, build platform, interpreter and zlib, SHA-256 and
size of each file); CI and the release workflow use it.

`scripts/release_parity.py` checks three recorded scenarios (retained-DOF travel, limits, drift and their targets;
the 0.852612768013 mW plate and a thin complex-transmission sample; the 32-line source, a continuous
spectrum and the static approximation) on an installed `os-optics`, against closed-form references, and
that Projects using none of the newer fields keep the hashes the published 0.12.0 gives them. Run it the same way
(`python -I scripts/release_parity.py --expect-version <V> --out parity.json`) against the candidate wheel and sdist
before publication and against the PyPI release after it; `tests/test_release_parity_script.py` runs it on the sources.

## Catalog and versions

The wheel contains no public optical catalog: catalog releases are data the engine synchronizes into a local store
(`os_optics.catalog`). A new catalog release, a supplier change or a price change is never an engine release. A
change of the distributed engine artifact (code, contracts, CLI, packaged assets) always gets a new version in
`src/os_optics/_version.py`; a published version is never reused. The test suite seeds a temporary catalog store
from explicit files (`tests/conftest.py`, `tests/catalog_store_helpers.py`): the small synthetic fixture catalog by
default, and one pinned release of the public optical catalog only where a test needs the real catalog
(`real_catalog_store` or `REAL_CATALOG`). That release is not part of the repository: the suite downloads it once from the
public catalog service into `.cache/public-catalog/` and checks its SHA-256 (`python tests/catalog_store_helpers.py`
fetches it ahead; `OS_OPTICS_TEST_PUBLIC_CATALOG` names a copy you already have). In CI only the `public-catalog` job
fetches it (from the actions cache, else downloaded with `--attempts`, each attempt time-limited); the test jobs take it
from that run's artifact.
