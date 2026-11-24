# Contributing to OS Optics

Thank you for helping improve OS Optics. This repository holds the product documentation
and the feedback entry for the `os-optics` engine. Bug reports, change proposals and
documentation pull requests are welcome.

## Language

Write issue titles, issue bodies, pull request titles and pull request descriptions in
Chinese or English.

## Report a bug

[Open a bug report](https://github.com/JunLin615/os-optics/issues/new/choose) and include:

- **Observed vs expected:** what you ran, what happened, and what you expected.
- **Steps to reproduce:** the exact commands or a small example project, with input values
  and units, and the catalog version if relevant.
- **Installed engine version:** from `os-optics doctor`.
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

The repository provides two checks, which also run in CI:

```bash
# Offline checks: links and anchors, language pairs, README index, version pins and baseline
python scripts/check_docs.py

# Installs the release recorded in engine-baseline.json into a fresh virtual environment
# and runs the documented quickstart commands (needs network access)
python scripts/check_quickstart.py --work quickstart
```

Both print one compact JSON line and exit 0 when clean.

## License

By contributing, you agree that your contribution to this repository is licensed under the
[MIT License](LICENSE).
