"""Build the os-optics wheel and sdist of the checked-out commit reproducibly, and record their identity.

    python scripts/build_dist.py [--outdir dist] [--allow-dirty]

The one build entry point of CI (job ``candidate``) and of the release workflow. It

* checks that the build tools are exactly the versions ``scripts/build-requirements.txt`` pins (install them with
  ``python -m pip install --require-hashes -r scripts/build-requirements.txt``);
* refuses a working tree with uncommitted changes to tracked files (``--allow-dirty`` builds anyway and says so);
* sets ``SOURCE_DATE_EPOCH`` to the commit time, so the archive timestamps are the commit's and a rebuild of the same
  commit with the same tools on the same operating system and interpreter build gives the same bytes (the archives also
  record the build OS in their member headers and are compressed by the interpreter's zlib, so another OS or zlib can
  give other bytes for the same files; the Engine build id does not change);
* runs ``python -m build --no-isolation``: the sdist, then the wheel built from that sdist (so the sdist is complete);
* writes ``identity.json`` beside the files: distribution, version, commit, tree, commit time, Engine build id (computed
  from the wheel's sources exactly as ``os_optics.manifest.build.engine_build_id`` computes it at run time), the build
  tool versions, the platform, the interpreter and its zlib, and, per file, its name, SHA-256 and size (in GitHub Actions also the repository, workflow, run, event,
  ref and SHA that built them).

Prints the identity as one JSON line; exit 0 on success, 1 on a failed check or build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
import zlib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "scripts" / "build-requirements.txt"
VERSION_FILE = ROOT / "src" / "os_optics" / "_version.py"
DISTRIBUTION = "os-optics"
PACKAGE = "os_optics"
#: The tools whose versions decide the bytes of the distributions.
PINNED_TOOLS = ("build", "hatchling")
BUILD_ID_HEX_CHARS = 16  # os_optics.manifest.build.BUILD_ID_HEX_CHARS


class BuildError(Exception):
    pass


def pinned_versions(requirements: Path = REQUIREMENTS) -> dict[str, str]:
    pins = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def check_tools(pins: dict[str, str]) -> dict[str, str]:
    found = {}
    for tool in PINNED_TOOLS:
        try:
            found[tool] = metadata.version(tool)
        except metadata.PackageNotFoundError:
            found[tool] = "missing"
    wrong = {tool: (found[tool], pins.get(tool)) for tool in PINNED_TOOLS if found[tool] != pins.get(tool)}
    if wrong:
        raise BuildError(f"build tools differ from {REQUIREMENTS.relative_to(ROOT).as_posix()}: "
                         + ", ".join(f"{t} {have} (pinned {want})" for t, (have, want) in wrong.items())
                         + "; run: python -m pip install --require-hashes -r scripts/build-requirements.txt")
    return found


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def source_version(text: str) -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', text, re.M)
    if not match:
        raise BuildError("no __version__ in src/os_optics/_version.py")
    return match.group(1)


def engine_build_id(wheel: Path) -> str:
    """``<version>+<sha256[:16]>`` of the wheel's ``os_optics`` .py sources: the run-time ``engine_build_id``
    (``os_optics.manifest.build._hash_sources``: paths relative to the package, sorted, CRLF normalized)."""
    prefix = PACKAGE + "/"
    with zipfile.ZipFile(wheel) as archive:
        entries = sorted((name[len(prefix):], archive.read(name).replace(b"\r\n", b"\n"))
                         for name in archive.namelist()
                         if name.startswith(prefix) and name.endswith(".py") and "/__pycache__/" not in name)
        version = source_version(archive.read(f"{PACKAGE}/_version.py").decode("utf-8"))
    digest = hashlib.sha256()
    for rel, data in entries:
        digest.update(rel.encode("utf-8") + b"\0" + data + b"\0")
    return f"{version}+{digest.hexdigest()[:BUILD_ID_HEX_CHARS]}"


def file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def build(outdir: Path, *, allow_dirty: bool = False) -> dict:
    tools = check_tools(pinned_versions())
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not allow_dirty:
        raise BuildError("the working tree has uncommitted changes to tracked files; commit them or pass --allow-dirty")
    commit, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    epoch = git("log", "-1", "--format=%ct", "HEAD")
    version = source_version(VERSION_FILE.read_text(encoding="utf-8"))
    env = {**os.environ, "SOURCE_DATE_EPOCH": epoch, "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryDirectory(prefix="os-optics-dist-") as scratch:
        proc = subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--outdir", scratch, str(ROOT)],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BuildError("python -m build failed:\n" + proc.stdout[-4000:] + proc.stderr[-4000:])
        built = sorted(Path(scratch).iterdir())
        wheels = [p for p in built if p.name.endswith(".whl")]
        sdists = [p for p in built if p.name.endswith(".tar.gz")]
        expected = {f"os_optics-{version}-py3-none-any.whl", f"os_optics-{version}.tar.gz"}
        if {p.name for p in built} != expected:
            raise BuildError(f"built {[p.name for p in built]}, expected {sorted(expected)}")
        outdir.mkdir(parents=True, exist_ok=True)
        for path in built:
            shutil.copyfile(path, outdir / path.name)
    wheel, sdist = outdir / wheels[0].name, outdir / sdists[0].name
    identity = {
        "schema": "os-optics.candidate_identity",
        "schema_version": 1,
        "distribution": DISTRIBUTION,
        "version": version,
        "commit": commit,
        "tree": tree,
        "source_date_epoch": int(epoch),
        "dirty": bool(dirty),
        "engine_build_id": engine_build_id(wheel),
        "build_tools": {"python": ".".join(map(str, sys.version_info[:3])), **tools},
        "build_platform": {"system": platform.system(), "machine": platform.machine(), "sys_platform": sys.platform,
                           "python_implementation": platform.python_implementation(),
                           "python_build": " ".join(platform.python_build()), "zlib": zlib.ZLIB_VERSION,
                           "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
                           **({"zlibng": zlib.ZLIBNG_VERSION} if hasattr(zlib, "ZLIBNG_VERSION") else {})},
        "files": [file_record(wheel), file_record(sdist)],
    }
    if os.environ.get("GITHUB_RUN_ID"):  # where it was built (descriptive; the files do not depend on it)
        identity["ci"] = {key.lower(): os.environ.get(key) for key in (
            "GITHUB_REPOSITORY", "GITHUB_WORKFLOW", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_EVENT_NAME", "GITHUB_REF",
            "GITHUB_SHA")}
    (outdir / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", type=Path, default=ROOT / "dist", help="output directory (default: dist)")
    parser.add_argument("--allow-dirty", action="store_true", help="build a working tree with uncommitted changes")
    args = parser.parse_args(argv)
    try:
        identity = build(args.outdir.resolve(), allow_dirty=args.allow_dirty)
    except (BuildError, subprocess.CalledProcessError) as exc:
        print(f"build_dist: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(identity, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
