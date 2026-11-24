"""Deployment defaults (os_optics/deployment.toml): the service origin is packaged data, not engine build identity."""

from __future__ import annotations

import shutil
import tomllib
from importlib import resources
from pathlib import Path

import os_optics.manifest.build as build
from os_optics import catalog as catalog_api
from os_optics.deployment import DEPLOYMENT_FILE, default_service_origin
from os_optics.manifest import engine_build_id

PACKAGE_ROOT = Path(str(resources.files("os_optics")))


def _configured_origin() -> str:
    return tomllib.loads((PACKAGE_ROOT / DEPLOYMENT_FILE).read_text(encoding="utf-8"))["service"]["origin"]


def test_the_default_service_origin_comes_from_the_packaged_deployment_file():
    origin = _configured_origin()
    assert origin.startswith("https://") and origin.count("/") == 2  # an origin, no path
    assert default_service_origin() == origin
    assert catalog_api.manifest_url({}) == origin + "/api/v1/catalog"
    # the environment still overrides the packaged default
    assert catalog_api.manifest_url({"OPENSCI_API_URL": "https://api.example.test"}) == "https://api.example.test/api/v1/catalog"


def test_no_engine_source_names_the_configured_host():
    host = _configured_origin().split("://", 1)[1]
    offenders = [str(p.relative_to(PACKAGE_ROOT)) for p in PACKAGE_ROOT.rglob("*.py") if host in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_engine_build_id_does_not_depend_on_the_deployment_file(tmp_path, monkeypatch):
    copy = tmp_path / "os_optics"
    shutil.copytree(PACKAGE_ROOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = build._hash_sources(copy)  # (the walk and digest themselves: engine_build_id answers once per process)
    deployment = copy / DEPLOYMENT_FILE
    deployment.write_text(deployment.read_text(encoding="utf-8").replace(_configured_origin(), "https://moved.example.test"),
                          encoding="utf-8")
    assert "https://moved.example.test" in deployment.read_text(encoding="utf-8")
    assert build._hash_sources(copy) == before
    assert build._hash_sources(PACKAGE_ROOT) == before  # the copy hashes exactly like the installed package
    # a changed source file does move the hash of the tree it is in; the running process keeps the id it first computed
    (copy / "units.py").write_text((copy / "units.py").read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    identified = engine_build_id()
    assert build._hash_sources(copy) != before and engine_build_id() == identified


def test_the_build_id_is_computed_once_per_process(monkeypatch):
    """The walk over the package sources runs on the first call only; later calls return the same string without it."""
    build.engine_build_id.cache_clear()
    calls = []
    original = build._hash_sources
    monkeypatch.setattr(build, "_hash_sources", lambda root: calls.append(root) or original(root))
    first = build.engine_build_id()
    assert build.engine_build_id() == first == engine_build_id() and len(calls) == 1
