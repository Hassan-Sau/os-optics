"""The release guards of .github/workflows/release.yml: the tag names this version, and the tagged commit is on main.

    python scripts/release_guard.py --tag v0.14.0 [--commit SHA --main-ref origin/main]

* ``--tag``: the release tag must be ``v<version>`` with ``<version>`` exactly ``__version__`` of
  ``src/os_optics/_version.py`` (``v0.14.0`` for 0.14.0; ``0.14.0``, ``v0.14``, ``v0.14.0rc1`` for 0.14.0 are refused);
* ``--commit`` with ``--main-ref``: the commit must be an ancestor of (or equal to) ``--main-ref`` in this repository
  (``git merge-base --is-ancestor``); the workflow fetches the whole history first.

Prints one JSON line naming each check and its result; exit 0 when every requested check passes, 1 otherwise, 2 on a
usage error. Nothing here builds, uploads or changes the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "src" / "os_optics" / "_version.py"
VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.M)


def package_version(version_file: Path = VERSION_FILE) -> str:
    match = VERSION.search(version_file.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"no __version__ in {version_file}")
    return match.group(1)


def tag_problem(tag: str, version: str) -> str | None:
    """None when ``tag`` is exactly ``v<version>``, else why not."""
    if tag == f"v{version}":
        return None
    return f"release tag {tag!r} does not name os-optics {version} (expected 'v{version}')"


def ancestry_problem(commit: str, main_ref: str, cwd: Path = ROOT) -> str | None:
    """None when ``commit`` is ``main_ref`` or one of its ancestors, else why not."""
    proc = subprocess.run(["git", "merge-base", "--is-ancestor", commit, main_ref], cwd=cwd, capture_output=True, text=True)
    if proc.returncode == 0:
        return None
    if proc.returncode == 1:
        return f"commit {commit} is not on {main_ref}; merge the pull request and tag the merge commit"
    return f"cannot compare {commit} with {main_ref}: {proc.stderr.strip()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help="the release tag, e.g. v0.14.0")
    parser.add_argument("--commit", help="the tagged commit (checked with --main-ref)")
    parser.add_argument("--main-ref", help="the ref the commit must be on, e.g. origin/main")
    parser.add_argument("--version-file", type=Path, default=VERSION_FILE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if bool(args.commit) != bool(args.main_ref):
        parser.print_usage(sys.stderr)
        print("release_guard: --commit and --main-ref go together", file=sys.stderr)
        return 2
    version = package_version(args.version_file)
    checks = {"tag": tag_problem(args.tag, version)}
    if args.commit:
        checks["on_main"] = ancestry_problem(args.commit, args.main_ref)
    failed = {name: problem for name, problem in checks.items() if problem}
    for problem in failed.values():
        print(f"::error::{problem}", file=sys.stderr)
    print(json.dumps({"status": "FAIL" if failed else "OK", "tag": args.tag, "version": version,
                      "checks": {name: problem or "OK" for name, problem in checks.items()}}, separators=(",", ":")))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
