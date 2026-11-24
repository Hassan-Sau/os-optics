# Contributing to OS Optics

Thank you for helping improve OS Optics. This repository holds the `os-optics` engine: its
source in `src/os_optics`, its tests, its documentation and the checks that build and
publish the package, including the read-only 3D viewer (`os-optics view`). Bug reports, change
proposals and pull requests are welcome. Tools, cases, agent skills and extra viewing commands
live in [os-optics-kit](https://github.com/JunLin615/os-optics-kit).

## Language

Write issue titles, issue bodies, pull request titles and pull request descriptions in
Chinese or English.

## Public reports

Issues and pull requests are public: don't include tokens, passwords or personal data, and use
a small example anyone can run (a released `os-optics` or this repository's source, and a
project or script with data you may share). End with checkable acceptance.

Report a security vulnerability privately through the repository's **Security** tab
(**Report a vulnerability**), not in a public issue.

## Report a bug

[Open a bug report](https://github.com/JunLin615/os-optics/issues/new/choose) and include:

- **Observed vs expected:** what you ran, what happened, and what you expected.
- **Steps to reproduce:** the exact commands or a small example project, with input values
  and units, and the catalog version if relevant.
- **Installed engine version:** from `os-optics doctor`, or the commit of this repository you
  built.
- **Where the work lives:** the documentation page or engine module (for example a path
  under `os_optics/`) involved, if you know it.
- **Acceptance:** a check that tells when the bug is fixed (see below).

## Propose a change

[Open a feature or engine change request](https://github.com/JunLin615/os-optics/issues/new/choose).
A proposal is written around the work that remains to be done:

- **Remaining work:** what has to change, stated as concrete deliverables.
- **Background** (optional): context, motivation and links.
- **Where the work lives:** the files or modules involved, such as `docs/en/results.md` or a
  module path under `os_optics/`.
- **Acceptance:** how a reviewer checks that the work is done.

## Writing the Acceptance section

Every issue ends with an Acceptance section made of checkable items. Each item states one
of:

- a given input and the expected output, with units;
- a command and the result it should produce (exit code, status, values);
- a test that fails before the change and passes after it.

For example:

```markdown
## Acceptance

- [ ] `os-optics project run project.json --out-dir run` exits 0 and `os-optics project verify run` reports the run as verified.
- [ ] `python scripts/check_docs.py` exits 0.
```

## Pull requests

- Reference the issue the pull request closes in its description, for example
  `Closes #123`.
- Keep one pull request focused on one issue.
- List the files you changed and say how you checked them.
- Customer-facing guides exist in English (`docs/en/`) and Simplified Chinese
  (`docs/zh-CN/`); update both language versions with the same headings, commands and
  links.

## Run the checks locally

The engine tests need [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer; `uv.lock`
pins the dependencies they use:

```bash
uv sync --extra test                                  # .venv with the package (editable) and the test tools
uv pip install --require-hashes -r scripts/build-requirements.txt   # build + hatchling: the wheel tests build the package
uv run --no-sync pytest -m "not pyodide and not install"   # the native test suite
uv run --no-sync pytest -m install                    # builds the wheel and runs the quickstart in a clean venv
(cd tests/pyodide && npm ci) && uv run --no-sync pytest -m pyodide   # Pyodide parity; needs Node.js 20 or newer
```

Tests that use the real public optical catalog download one pinned release (about 13 MB)
into `.cache/public-catalog/` the first time they run; `python tests/catalog_store_helpers.py`
fetches it ahead, and `OS_OPTICS_TEST_PUBLIC_CATALOG` can name a copy you already have.
When the release cannot be obtained, each test module that needs it reports a failing test that names the reason
(also under `-m install` or `-m pyodide`); no test is skipped silently.

[Development notes](docs/engine/DEVELOPMENT.md) describe the fixtures, benchmarks and
conventions. The documentation checks:

```bash
# Offline checks: links and anchors, language pairs, README index, version pins and baseline
python scripts/check_docs.py

# Installs the release recorded in engine-baseline.json into a fresh virtual environment
# and runs the documented quickstart commands (needs network access)
python scripts/check_quickstart.py --work quickstart
```

Both print one compact JSON line and exit 0 when clean.

## License

The engine is MIT-licensed from release 0.14.0; releases 0.13.0 and earlier keep the licence
they were published under. By contributing, you agree that your contribution to this
repository is licensed under the [MIT License](LICENSE).
