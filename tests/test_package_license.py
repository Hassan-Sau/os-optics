"""The MIT License ships in every distribution.

Builds the wheel and the source distribution from the current sources and checks that both carry the repository's
``LICENSE`` (MIT, ``Copyright (c) 2026 JunLin615``) and no earlier licence or notice file, and that the core metadata
names the licence with the PEP 639 expression ``MIT`` (no legacy licence classifier, no free-text ``License:`` field)
and the public source and issue URLs.
"""

from __future__ import annotations

import email.parser
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.install

ROOT = Path(__file__).resolve().parents[1]
EXPRESSION = "MIT"
EARLIER_FILES = ("LICENSE.txt", "NOTICE.txt")
URLS = {"Source": "https://github.com/JunLin615/os-optics", "Issues": "https://github.com/JunLin615/os-optics/issues"}


@pytest.fixture(scope="module")
def dists(tmp_path_factory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("license_dist")
    proc = subprocess.run([sys.executable, "-m", "build", "--wheel", "--sdist", "--no-isolation", "--outdir", str(out), str(ROOT)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    (wheel,) = out.glob("os_optics-*.whl")
    (sdist,) = out.glob("os_optics-*.tar.gz")
    return wheel, sdist


def _metadata(text: str):
    return email.parser.Parser().parsestr(text)


def _check_metadata(meta) -> None:
    assert meta["License-Expression"] == EXPRESSION
    assert meta.get_all("License-File") == ["LICENSE"]
    assert meta["License"] is None
    assert not [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    urls = dict(entry.split(", ", 1) for entry in meta.get_all("Project-URL") or [])
    assert {key: urls.get(key) for key in URLS} == URLS


def test_the_repository_licence_is_mit_and_names_the_holder():
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert licence.startswith("MIT License\n") and "Copyright (c) 2026 JunLin615" in licence
    assert "Permission is hereby granted, free of charge" in licence
    assert not [name for name in EARLIER_FILES if (ROOT / name).exists()]


def test_wheel_ships_the_licence_and_the_expression(dists):
    wheel, _ = dists
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        dist_info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
        assert archive.read(f"{dist_info}/licenses/LICENSE") == (ROOT / "LICENSE").read_bytes()
        assert not [n for n in names if n.rsplit("/", 1)[-1] in EARLIER_FILES]
        assert "os_optics/viewer/assets/THREE_LICENSE.txt" in names  # the bundled three.js keeps its notice
        meta = _metadata(archive.read(f"{dist_info}/METADATA").decode("utf-8"))
    _check_metadata(meta)


def test_sdist_ships_the_licence_and_the_expression(dists):
    _, sdist = dists
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        top = names[0].split("/")[0]
        assert f"{top}/LICENSE" in names
        assert archive.extractfile(f"{top}/LICENSE").read() == (ROOT / "LICENSE").read_bytes()
        assert not [n for n in names if n.rsplit("/", 1)[-1] in EARLIER_FILES]
        meta = _metadata(archive.extractfile(f"{top}/PKG-INFO").read().decode("utf-8"))
    _check_metadata(meta)
