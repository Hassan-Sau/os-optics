"""Every identity the engine writes or sends names the distribution that produced it (``os-optics``).

A script matches a run directory to an installation with ``importlib.metadata.distribution(engine_package)`` and
``engine_version``, without a per-release case: ``local_validation.json`` and ``manifest.json`` name the same
distribution, and it is the installed one. One constant (``DISTRIBUTION_NAME``) is the source; the contracts'
``ENGINE_PACKAGE_NAME`` is the same value.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import tomllib
from pathlib import Path

import pytest

from os_optics import DISTRIBUTION_NAME, ENGINE_VERSION, __version__
from os_optics._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1
from os_optics._vendor.shared_contracts.optical_design import EARLIER_ENGINE_PACKAGE_NAME, ENGINE_PACKAGE_NAME
from os_optics.assembly_drawing.pdf import PRODUCER
from os_optics.catalog.remote import USER_AGENT as CATALOG_USER_AGENT
from os_optics.cli_doctor import DISTRIBUTION as DOCTOR_DISTRIBUTION
from os_optics.client.transport import USER_AGENT as CLIENT_USER_AGENT
from os_optics.local import example_project, run_optical_project, verify_run_artifacts
from os_optics.local.artifacts import MANIFEST_FILE
from os_optics.optical_view import BUILDER_PACKAGE, canonical_json_bytes, optical_view_files, parse_glb
from os_optics.optical_view.glb import RAY_SOURCE_NOTE

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "os_optics"
#: Where the earlier name may still be a value: the alias command name among the program names the command answers to
#: (``COMMAND_NAMES``, defined once in ``_command.py``, which the command imports first; ``cli.py`` uses it from there).
EARLIER_NAME_ALLOWED = {"_command.py"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run503")
    run_optical_project(example_project(), output_dir=out)
    return out


def test_one_constant_names_the_distribution():
    pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert DISTRIBUTION_NAME == ENGINE_PACKAGE_NAME == pyproject["project"]["name"] == "os-optics"
    assert EARLIER_ENGINE_PACKAGE_NAME == "opensci-engine" != DISTRIBUTION_NAME
    assert BUILDER_PACKAGE == DOCTOR_DISTRIBUTION == DISTRIBUTION_NAME


def test_the_name_is_the_installed_distribution():
    installed = importlib.metadata.distribution(DISTRIBUTION_NAME)
    assert installed.metadata["Name"] == DISTRIBUTION_NAME and installed.version == __version__


def test_a_fresh_run_names_the_installed_distribution_in_record_and_manifest(run_dir):
    record = json.loads((run_dir / "local_validation.json").read_bytes())
    manifest = json.loads((run_dir / MANIFEST_FILE).read_bytes())
    runtime, engine_run = record["runtime"], manifest["engine_run"]
    assert runtime["engine_package"] == engine_run["engine_package"] == DISTRIBUTION_NAME
    assert runtime["engine_version"] == engine_run["engine_version"] == ENGINE_VERSION
    # the match a provenance script makes, with no per-release case
    installed = importlib.metadata.distribution(engine_run["engine_package"])
    assert (installed.metadata["Name"], installed.version) == (runtime["engine_package"], runtime["engine_version"])
    scene = json.loads((run_dir / "optical_scene.json").read_bytes())
    assert scene["engine_run"]["engine_package"] == scene["builder"]["builder_package"] == DISTRIBUTION_NAME
    gltf, _ = parse_glb((run_dir / "optical_scene.glb").read_bytes())
    assert gltf["asset"]["generator"] == f"{DISTRIBUTION_NAME} {ENGINE_VERSION}"


def test_what_the_engine_sends_and_stamps_names_the_distribution():
    assert CATALOG_USER_AGENT == CLIENT_USER_AGENT == f"{DISTRIBUTION_NAME}/{__version__}"
    assert PRODUCER == f"OpenSci {DISTRIBUTION_NAME} drawing"
    assert RAY_SOURCE_NOTE.startswith(f"{DISTRIBUTION_NAME} trace ")


def _copy(run_dir: Path, tmp_path: Path) -> Path:
    directory = tmp_path / "copy"
    directory.mkdir()
    for path in run_dir.iterdir():
        (directory / path.name).write_bytes(path.read_bytes())
    return directory


def test_the_manifest_must_name_the_records_distribution(run_dir, tmp_path):
    directory = _copy(run_dir, tmp_path)
    manifest = json.loads((directory / MANIFEST_FILE).read_bytes())
    assert manifest["schema_version"] == "1.2"
    for stated in ("opensci-engine", None):
        if stated is None:
            manifest["engine_run"].pop("engine_package")
        else:
            manifest["engine_run"]["engine_package"] = stated
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
        assert verify_run_artifacts(directory) == ("manifest.json: its engine package differs from the local validation record's",)


def _earlier_engine_run(run_dir: Path, tmp_path: Path) -> Path:
    """The run as os-optics 0.10.0 / 0.11.0 wrote it: record and scene name ``opensci-engine``, run manifest 1.1 without
    ``engine_run.engine_package``; every digest the manifest and the GLB carry is recomputed, as that engine wrote them."""

    directory = _copy(run_dir, tmp_path)
    record = json.loads((directory / "local_validation.json").read_bytes())
    record["runtime"]["engine_package"] = EARLIER_ENGINE_PACKAGE_NAME
    (directory / "local_validation.json").write_bytes(canonical_json_bytes(record))
    scene_doc = json.loads((directory / "optical_scene.json").read_bytes())
    scene_doc["engine_run"]["engine_package"] = scene_doc["builder"]["builder_package"] = EARLIER_ENGINE_PACKAGE_NAME
    scene = OpticalSimulationSceneV1.model_validate(scene_doc)
    for name, data in optical_view_files(scene).items():
        (directory / name).write_bytes(data)
    manifest = json.loads((directory / MANIFEST_FILE).read_bytes())
    manifest["schema_version"] = "1.1"
    manifest["engine_run"].pop("engine_package")
    manifest["optical_view"]["scene_content_hash_osc1"] = scene.content_hash()
    for entry in manifest["files"]:
        data = (directory / entry["name"]).read_bytes()
        entry["bytes"], entry["sha256"] = len(data), hashlib.sha256(data).hexdigest()
    (directory / MANIFEST_FILE).write_bytes(canonical_json_bytes(manifest))
    return directory


def test_a_run_directory_of_an_earlier_engine_still_verifies(run_dir, tmp_path):
    """os-optics 0.10.0 / 0.11.0 wrote ``opensci-engine`` in the record and the scene, and run manifest 1.1."""

    directory = _earlier_engine_run(run_dir, tmp_path)
    assert json.loads((directory / "local_validation.json").read_bytes())["runtime"]["engine_package"] == "opensci-engine"
    assert verify_run_artifacts(directory) == ()
    # the same files under a 1.2 manifest must state the record's package
    manifest = json.loads((directory / MANIFEST_FILE).read_bytes())
    manifest["schema_version"] = "1.2"
    manifest["engine_run"]["engine_package"] = DISTRIBUTION_NAME
    (directory / MANIFEST_FILE).write_bytes(canonical_json_bytes(manifest))
    assert verify_run_artifacts(directory) == ("manifest.json: its engine package differs from the local validation record's",)
    manifest["engine_run"]["engine_package"] = EARLIER_ENGINE_PACKAGE_NAME
    (directory / MANIFEST_FILE).write_bytes(canonical_json_bytes(manifest))
    assert verify_run_artifacts(directory) == ()


def test_a_scene_and_record_of_different_distributions_do_not_belong_together(run_dir, tmp_path):
    directory = _earlier_engine_run(run_dir, tmp_path)
    (directory / "local_validation.json").write_bytes((run_dir / "local_validation.json").read_bytes())  # names os-optics
    manifest = json.loads((directory / MANIFEST_FILE).read_bytes())
    for entry in manifest["files"]:
        data = (directory / entry["name"]).read_bytes()
        entry["bytes"], entry["sha256"] = len(data), hashlib.sha256(data).hexdigest()
    (directory / MANIFEST_FILE).write_bytes(canonical_json_bytes(manifest))
    assert verify_run_artifacts(directory) == (
        "the optical scene and the local validation record were made by different Engine distributions or versions",)


def _string_values(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def test_no_engine_source_writes_the_earlier_name():
    """A new literal ``opensci-engine`` (or ``opensci-engine/...``) in the engine's own source would be an identity under
    the earlier name; the one exception is the alias command name."""

    found = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT)
        if relative.parts[0] == "_vendor" or relative.as_posix() in EARLIER_NAME_ALLOWED:
            continue
        for value in _string_values(ast.parse(path.read_text(encoding="utf-8"))):
            if value == EARLIER_ENGINE_PACKAGE_NAME or value.startswith((EARLIER_ENGINE_PACKAGE_NAME + "/", EARLIER_ENGINE_PACKAGE_NAME + " ")):
                found.append(f"{relative.as_posix()}: {value[:80]!r}")
    assert found == []
