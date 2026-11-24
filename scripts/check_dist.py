"""Check built os-optics distributions before they are installed or published.

    python scripts/check_dist.py DIST_DIR [--version X.Y.Z]

For the one wheel and the one sdist in DIST_DIR (default version: src/os_optics/_version.py):

* metadata: name, version, ``License-Expression: MIT``, ``License-File: LICENSE``, the Homepage / Source / Issues /
  Documentation project URLs, ``Requires-Python``; the same in the wheel's METADATA and the sdist's PKG-INFO;
* the licence: the MIT ``LICENSE`` (``Copyright (c) 2026 JunLin615``) is in both, the three.js notice is in the wheel, and
  no member is an earlier licence or notice file (``LICENSE.txt``, ``NOTICE.txt``) or declares the earlier proprietary
  licence;
* contents: the wheel holds only ``os_optics/`` and its ``.dist-info``, with the console script ``os-optics``; the sdist
  holds only ``src/os_optics/``, ``docs/engine/README.md``, ``LICENSE``, ``pyproject.toml``, ``PKG-INFO`` and hatchling's ``.gitignore``; neither
  carries tests, compiled caches, a catalog release, the earlier ``opensci_engine`` package or a reference to a
  repository of the owner other than ``os-optics`` and ``os-optics-kit``;
* ``identity.json``, when present (``scripts/build_dist.py``): every file's SHA-256 and size, and the Engine build id
  recomputed from the wheel.

Prints one compact JSON document; exit 0 when clean, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dist  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URLS = {
    "Homepage": "https://opensci-optics.com/",
    "Source": "https://github.com/JunLin615/os-optics",
    "Issues": "https://github.com/JunLin615/os-optics/issues",
    "Documentation": "https://github.com/JunLin615/os-optics/blob/main/docs/engine/README.md",
}
MIT_MARKERS = ("MIT License", "Copyright (c) 2026 JunLin615", "Permission is hereby granted, free of charge")
THREE_NOTICE = "os_optics/viewer/assets/THREE_LICENSE.txt"
FORBIDDEN_NAMES = {"LICENSE.txt", "NOTICE.txt", "NOTICE", "NOTICE.md"}
#: Text no shipped file may contain: the earlier proprietary licence declaration.
FORBIDDEN_TEXT = (
    re.compile(r"LicenseRef-OpenSci-Engine"),
    re.compile(r"is proprietary software"),
    re.compile(r"is not open-source software"),
)
#: The owner's repositories a shipped file may name; a reference to any other repository of the owner is refused.
PUBLIC_REPOSITORIES = {"os-optics", "os-optics-kit"}
REPOSITORY_REFERENCE = re.compile(r"JunLin615/([A-Za-z0-9_.-]+)")
TEXT_SUFFIXES = {".py", ".json", ".toml", ".txt", ".md", ".html", ".css", ".js", ".typed", ""}
SDIST_TOP = ("src/os_optics/", "docs/engine/README.md", "LICENSE", "pyproject.toml", "PKG-INFO", ".gitignore")


def metadata_problems(text: str, where: str, version: str) -> list[str]:
    message = Parser().parsestr(text)
    problems = []
    expect = {"Name": "os-optics", "Version": version, "License-Expression": "MIT", "Requires-Python": ">=3.11"}
    for key, value in expect.items():
        if message.get(key) != value:
            problems.append(f"{where}: {key} is {message.get(key)!r}, expected {value!r}")
    if message.get_all("License-File") != ["LICENSE"]:
        problems.append(f"{where}: License-File is {message.get_all('License-File')!r}, expected ['LICENSE']")
    if message.get("License") is not None:
        problems.append(f"{where}: carries a License field beside License-Expression")
    urls = dict(entry.split(", ", 1) for entry in message.get_all("Project-URL") or [])
    if urls != URLS:
        problems.append(f"{where}: Project-URL is {urls!r}, expected {URLS!r}")
    if "Classifier: License ::" in text:
        problems.append(f"{where}: a licence classifier beside License-Expression")
    return problems


def member_problems(name: str, data: bytes) -> list[str]:
    path = PurePosixPath(name)
    problems = []
    if path.name in FORBIDDEN_NAMES:
        problems.append(f"{name}: earlier licence or notice file")
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        problems.append(f"{name}: compiled cache")
    if "opensci_engine" in path.parts or "tests" in path.parts or ".cache" in path.parts:
        problems.append(f"{name}: not part of the distribution")
    if "public_optical_catalog" in path.name:
        problems.append(f"{name}: catalog release")
    if path.suffix in TEXT_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
        problems += [f"{name}: contains {token.pattern!r}" for token in FORBIDDEN_TEXT if token.search(text)]
        problems += [f"{name}: names the repository JunLin615/{repo}" for repo in
                     sorted({m.group(1).rstrip(".") for m in REPOSITORY_REFERENCE.finditer(text)} - PUBLIC_REPOSITORIES)]
    return problems


def license_problems(text: str, where: str) -> list[str]:
    return [f"{where}: LICENSE lacks {marker!r}" for marker in MIT_MARKERS if marker not in text]


def check_wheel(path: Path, version: str) -> list[str]:
    problems = []
    info = f"os_optics-{version}.dist-info/"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            if not (name.startswith("os_optics/") or name.startswith(info)):
                problems.append(f"{path.name}: {name}: outside the package and its metadata")
            problems += [f"{path.name}: {p}" for p in member_problems(name, archive.read(name))]
        if info + "METADATA" not in names:
            return problems + [f"{path.name}: no METADATA"]
        problems += metadata_problems(archive.read(info + "METADATA").decode("utf-8"), f"{path.name} METADATA", version)
        if info + "licenses/LICENSE" not in names:
            problems.append(f"{path.name}: no {info}licenses/LICENSE")
        else:
            problems += license_problems(archive.read(info + "licenses/LICENSE").decode("utf-8"), path.name)
        if THREE_NOTICE not in names:
            problems.append(f"{path.name}: the three.js notice {THREE_NOTICE} is missing")
        entry_points = archive.read(info + "entry_points.txt").decode("utf-8") if info + "entry_points.txt" in names else ""
        if not re.search(r"^os-optics = os_optics\.cli:main$", entry_points, re.M):
            problems.append(f"{path.name}: console script os-optics = os_optics.cli:main missing")
        for required in ("os_optics/__init__.py", "os_optics/py.typed", "os_optics/deployment.toml",
                         "os_optics/_vendor/MANIFEST.json", "os_optics/viewer/assets/index.html"):
            if required not in names:
                problems.append(f"{path.name}: {required} missing")
    return problems


def check_sdist(path: Path, version: str) -> list[str]:
    problems = []
    top = f"os_optics-{version}/"
    with tarfile.open(path) as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        names = []
        for member in members:
            if not member.name.startswith(top):
                problems.append(f"{path.name}: {member.name}: outside {top}")
                continue
            rel = member.name[len(top):]
            names.append(rel)
            if not any(rel == entry or (entry.endswith("/") and rel.startswith(entry)) for entry in SDIST_TOP):
                problems.append(f"{path.name}: {rel}: not part of the source distribution")
            problems += [f"{path.name}: {p}" for p in member_problems(rel, archive.extractfile(member).read())]
        for required in ("PKG-INFO", "pyproject.toml", "LICENSE", "docs/engine/README.md", "src/os_optics/__init__.py",
                         "src/os_optics/viewer/assets/THREE_LICENSE.txt"):
            if required not in names:
                problems.append(f"{path.name}: {required} missing")
        if "PKG-INFO" in names:
            problems += metadata_problems(archive.extractfile(top + "PKG-INFO").read().decode("utf-8"),
                                          f"{path.name} PKG-INFO", version)
        if "LICENSE" in names:
            problems += license_problems(archive.extractfile(top + "LICENSE").read().decode("utf-8"), path.name)
    return problems


def check_identity(directory: Path, wheel: Path, version: str) -> list[str]:
    path = directory / "identity.json"
    if not path.is_file():
        return []
    identity = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if identity.get("version") != version:
        problems.append(f"identity.json: version {identity.get('version')!r}, expected {version!r}")
    for entry in identity.get("files", []):
        file = directory / entry["name"]
        data = file.read_bytes() if file.is_file() else b""
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["bytes"]:
            problems.append(f"identity.json: {entry['name']} does not match its recorded SHA-256 / size")
    if identity.get("engine_build_id") != build_dist.engine_build_id(wheel):
        problems.append("identity.json: engine_build_id differs from the one computed from the wheel")
    return problems


def check(directory: Path, version: str) -> dict:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    problems = []
    if len(wheels) != 1 or len(sdists) != 1:
        problems.append(f"expected one wheel and one sdist in {directory}, found {[p.name for p in wheels + sdists]}")
    else:
        if wheels[0].name != f"os_optics-{version}-py3-none-any.whl":
            problems.append(f"{wheels[0].name}: expected os_optics-{version}-py3-none-any.whl")
        if sdists[0].name != f"os_optics-{version}.tar.gz":
            problems.append(f"{sdists[0].name}: expected os_optics-{version}.tar.gz")
        problems += check_wheel(wheels[0], version) + check_sdist(sdists[0], version)
        problems += check_identity(directory, wheels[0], version)
    return {"status": "FAIL" if problems else "OK", "version": version,
            "files": [p.name for p in wheels + sdists], "problems": problems}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", help="expected version (default: src/os_optics/_version.py)")
    args = parser.parse_args(argv)
    version = args.version or build_dist.source_version(build_dist.VERSION_FILE.read_text(encoding="utf-8"))
    report = check(args.directory, version)
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
