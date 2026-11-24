"""Tests of the release tooling: the tag / version and main-ancestry guards of release.yml, and the identity helpers of
the build entry point. Run with ``python -m pytest scripts/tests`` (no Engine installation needed)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = _load("release_guard")
build_dist = _load("build_dist")
VERSION = guard.package_version()


def _guard(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / "release_guard.py"), *args], cwd=cwd, capture_output=True,
                          text=True, timeout=120)


def test_the_version_is_the_package_version():
    namespace: dict = {}
    exec((ROOT / "src" / "os_optics" / "_version.py").read_text(encoding="utf-8"), namespace)  # noqa: S102 (own source)
    assert VERSION == namespace["__version__"] == build_dist.source_version(build_dist.VERSION_FILE.read_text("utf-8"))


def test_the_matching_tag_passes():
    proc = _guard("--tag", f"v{VERSION}")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"status": "OK", "tag": f"v{VERSION}", "version": VERSION, "checks": {"tag": "OK"}}


@pytest.mark.parametrize("tag", [VERSION, f"v{VERSION}rc1", f"v{VERSION}.1", "v0.0.0", f"V{VERSION}",
                                 "v" + ".".join(VERSION.split(".")[:2]), f"release-{VERSION}", ""])
def test_a_mismatched_tag_fails_and_names_the_version(tag):
    proc = _guard("--tag", tag)
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["status"] == "FAIL" and report["version"] == VERSION
    assert f"expected 'v{VERSION}'" in report["checks"]["tag"]
    assert proc.stderr.startswith("::error::release tag")


def test_commit_and_main_ref_go_together():
    assert _guard("--tag", f"v{VERSION}", "--commit", "HEAD").returncode == 2


@pytest.fixture
def repository(tmp_path):
    """A repository whose main has two commits, and a side branch off the first."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    for name in ("one", "two"):
        (tmp_path / name).write_text(name, encoding="utf-8")
        git("add", name)
        git("commit", "-q", "-m", name)
    first = git("rev-parse", "HEAD~1")
    git("checkout", "-q", "-b", "side", first)
    (tmp_path / "side").write_text("side", encoding="utf-8")
    git("add", "side")
    git("commit", "-q", "-m", "side")
    return tmp_path, {"main": git("rev-parse", "main"), "first": first, "side": git("rev-parse", "side")}


def test_a_commit_on_main_passes_and_one_off_main_fails(repository):
    path, commits = repository
    assert guard.ancestry_problem(commits["main"], "main", cwd=path) is None
    assert guard.ancestry_problem(commits["first"], "main", cwd=path) is None
    assert "is not on main" in guard.ancestry_problem(commits["side"], "main", cwd=path)
    assert "cannot compare" in guard.ancestry_problem("0" * 40, "main", cwd=path)


def test_the_engine_build_id_is_the_runtime_algorithm(tmp_path):
    """``build_dist.engine_build_id`` hashes a wheel's package sources the way ``os_optics.manifest.build`` does."""
    sources = {"os_optics/__init__.py": b"x = 1\r\n", "os_optics/_version.py": b'__version__ = "9.8.7"\n',
               "os_optics/sub/mod.py": b"y = 2\n", "os_optics/viewer/assets/index.html": b"<html>",
               "os_optics-9.8.7.dist-info/METADATA": b"Name: os-optics\n"}
    wheel = tmp_path / "os_optics-9.8.7-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in sources.items():
            archive.writestr(name, data)
    import hashlib

    digest = hashlib.sha256()
    for rel, data in sorted([("__init__.py", b"x = 1\n"), ("_version.py", b'__version__ = "9.8.7"\n'),
                             ("sub/mod.py", b"y = 2\n")]):
        digest.update(rel.encode() + b"\0" + data + b"\0")
    assert build_dist.engine_build_id(wheel) == "9.8.7+" + digest.hexdigest()[:16]


def test_the_build_tool_pins_are_exact():
    pins = build_dist.pinned_versions()
    assert set(build_dist.PINNED_TOOLS) <= set(pins)
    assert all(pins[tool].count(".") >= 1 for tool in build_dist.PINNED_TOOLS)


def _synthetic_dist(directory: Path, *, license_expression: str = "MIT", extra: dict | None = None) -> None:
    """A minimal wheel and sdist of this version with the given metadata and extra members."""
    import io
    import tarfile

    check_dist = _load("check_dist")
    urls = "".join(f"Project-URL: {key}, {value}\n" for key, value in check_dist.URLS.items())
    metadata = (f"Metadata-Version: 2.4\nName: os-optics\nVersion: {VERSION}\n{urls}License-Expression: {license_expression}\n"
                "License-File: LICENSE\nRequires-Python: >=3.11\n\n")
    licence = (ROOT / "LICENSE").read_bytes()
    info = f"os_optics-{VERSION}.dist-info"
    members = {"os_optics/__init__.py": b"", "os_optics/py.typed": b"", "os_optics/deployment.toml": b"",
               "os_optics/_vendor/MANIFEST.json": b"{}", "os_optics/viewer/assets/index.html": b"",
               "os_optics/_version.py": f'__version__ = "{VERSION}"\n'.encode(),
               check_dist.THREE_NOTICE: b"MIT", f"{info}/METADATA": metadata.encode(), f"{info}/licenses/LICENSE": licence,
               f"{info}/entry_points.txt": b"[console_scripts]\nos-optics = os_optics.cli:main\n", **(extra or {})}
    with zipfile.ZipFile(directory / f"os_optics-{VERSION}-py3-none-any.whl", "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    top = f"os_optics-{VERSION}/"
    files = {"PKG-INFO": metadata.encode(), "pyproject.toml": b"", "LICENSE": licence, "docs/engine/README.md": b"",
             "src/os_optics/__init__.py": b"", "src/os_optics/viewer/assets/THREE_LICENSE.txt": b"MIT"}
    with tarfile.open(directory / f"os_optics-{VERSION}.tar.gz", "w:gz") as archive:
        for name, data in files.items():
            entry = tarfile.TarInfo(top + name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))


def test_check_dist_accepts_a_clean_pair_and_names_each_defect(tmp_path):
    check_dist = _load("check_dist")
    clean = tmp_path / "clean"
    clean.mkdir()
    _synthetic_dist(clean)
    assert check_dist.check(clean, VERSION)["problems"] == []
    bad = tmp_path / "bad"
    bad.mkdir()
    _synthetic_dist(bad, license_expression="LicenseRef-OpenSci-Engine",
                    extra={"os_optics/NOTICE.txt": b"x", "tests/test_x.py": b"", "os_optics/x.py": b"see JunLin615/another-repository/issues/1"})
    problems = "\n".join(check_dist.check(bad, VERSION)["problems"])
    assert "License-Expression is 'LicenseRef-OpenSci-Engine'" in problems
    assert "NOTICE.txt: earlier licence or notice file" in problems
    assert "tests/test_x.py: outside the package" in problems
    assert "os_optics/x.py: names the repository JunLin615/another-repository" in problems
