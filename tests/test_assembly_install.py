"""``pip install os-optics`` alone is sufficient for the local assembly layout / collision check (plan v0.12 stage six).

Builds the wheel from the current sources, installs ONLY that wheel (plus its declared PyPI dependencies) into a fresh
venv outside the repository, copies the reference Assembly Preview Package archive next to it, and then, from a
directory outside the repository with a scrubbed environment (no ``PYTHONPATH``, no user site), where
the unvendored ``shared_contracts`` package is not importable:

* Python: ``load_assembly_preview`` -> ``validate_assembly_layout`` -> ``result_file_bytes`` (and the browser (Pyodide) runtime entry point
  ``json_api.validate_assembly_package`` on the archive bytes and on the file map);
* the console script: ``os-optics assembly verify`` and ``os-optics assembly validate --out``.

The installed result must equal the in-repo result byte for byte (the wheel is built from these sources, so it
carries the same ``engine_build_id``) and validate with the vendored ``AssemblyLocalValidationV1`` contract.
Mark: ``install``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from os_optics._vendor.shared_contracts.assembly_preview import AssemblyLocalValidationV1
from os_optics.assembly import result_file_bytes, validate_assembly_layout
from os_optics.manifest import engine_build_id

pytestmark = pytest.mark.install

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
FIXTURE = ROOT / "tests/fixtures/assembly_preview/reference.assembly-preview.zip"
PACKAGE_NAME = "reference.assembly-preview.zip"
MONOREPO_MODULES = ("shared_contracts",)
REQUIRED_WHEEL_FILES = (
    "os_optics/assembly/__init__.py",
    "os_optics/assembly/checks.py",
    "os_optics/assembly/result.py",
    "os_optics/_vendor/shared_contracts/assembly_preview/__init__.py",
    "os_optics/_vendor/shared_contracts/assembly_preview/package.py",
    "os_optics/_vendor/shared_contracts/assembly_preview/validation.py",
)
EXPECTED_CHECKS = {
    "PACKAGE_INTEGRITY": "PASS",
    "SCENE_TRANSFORMS": "PASS",
    "PART_GEOMETRY": "PASS",
    "SUPPORT_STRUCTURE": "PASS",
    "SUPPORT_ATTACHMENT": "PASS",
    "BOARD_CONTAINMENT": "PASS",
    "BOARD_FIXATION": "PASS",
    "EXPECTED_ENGAGEMENT": "PASS",
    "PART_COLLISION": "PASS",
    "BEAM_AXIS_CLEARANCE": "PASS",
}


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("assembly_dist")
    proc = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out), str(ROOT)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = list(out.glob("os_optics-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="module")
def venv_python(wheel, tmp_path_factory) -> Path:
    venv = tmp_path_factory.mktemp("assembly_venv") / "venv"
    assert REPO_ROOT not in venv.resolve().parents, venv
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    proc = subprocess.run([str(py), "-m", "pip", "install", "--only-binary=:all:", "--no-cache-dir", str(wheel)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, "pip install of the wheel failed (PyPI reachable?):\n" + proc.stdout[-2000:] + proc.stderr[-2000:]
    return py


@pytest.fixture(scope="module")
def outside_dir(tmp_path_factory) -> Path:
    work = tmp_path_factory.mktemp("assembly_outside_repo")
    assert REPO_ROOT not in work.resolve().parents, work
    shutil.copyfile(FIXTURE, work / PACKAGE_NAME)
    return work


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP"}}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(args: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=_clean_env(), capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def _exe(venv_python: Path) -> Path:
    exe = venv_python.parent / ("os-optics.exe" if os.name == "nt" else "os-optics")
    assert exe.is_file(), exe
    return exe


@pytest.fixture(scope="module")
def in_repo_result_bytes() -> bytes:
    return result_file_bytes(validate_assembly_layout(FIXTURE))


# ---- the installed Python API ------------------------------------------------------------------------------------------------
INSTALLED_SCRIPT = textwrap.dedent(
    f"""
    import importlib, json, sys, sysconfig, zipfile
    from pathlib import Path

    imports = {{}}
    for module in {MONOREPO_MODULES!r}:
        try:
            importlib.import_module(module)
            imports[module] = "IMPORTED"
        except ImportError as exc:
            imports[module] = type(exc).__name__

    import os_optics.assembly
    from os_optics.assembly import load_assembly_preview, result_file_bytes, validate_assembly_layout
    from os_optics.local import json_api
    from os_optics.manifest import engine_build_id
    from os_optics._vendor.shared_contracts.assembly_preview import AssemblyLocalValidationV1

    package = load_assembly_preview({PACKAGE_NAME!r})
    result = validate_assembly_layout(package)
    assert result.status.value == "PASS", result.status
    assert result.summary.interference.value == "NO_INTERFERENCE", result.summary.interference
    data = result_file_bytes(result)
    Path("py_result.json").write_bytes(data)
    reparsed = AssemblyLocalValidationV1.model_validate_json(data)  # the vendored contract (result_hash re-checked)

    archive = Path({PACKAGE_NAME!r}).read_bytes()
    with zipfile.ZipFile({PACKAGE_NAME!r}) as z:
        file_map = {{name: z.read(name) for name in z.namelist()}}
    envelopes = {{}}
    for label, source in (("bytes", archive), ("file_map", file_map)):
        reply = json_api.validate_assembly_package(source)
        envelopes[label] = {{"ok": reply["ok"], "status": reply["status"], "same_file": reply["result_file"] == data,
                            "same_result": reply["result"] == result.payload()}}

    print(json.dumps({{
        "imports": imports,
        "assembly_file": os_optics.assembly.__file__,
        "purelib": sysconfig.get_paths()["purelib"],
        "sys_path": sys.path,
        "engine_build_id": engine_build_id(),
        "package_content_hash": package.content_hash,
        "status": result.status.value,
        "claim": result.summary.interference.value,
        "checks": {{c.check.value: c.status.value for c in result.checks}},
        "result_hash": result.result_hash,
        "reparsed_hash": reparsed.result_hash,
        "envelopes": envelopes,
    }}, sort_keys=True))
    """
)


@pytest.fixture(scope="module")
def installed(venv_python, outside_dir) -> dict:
    script = outside_dir / "assembly_check.py"
    script.write_text(INSTALLED_SCRIPT, encoding="utf-8")
    proc = _run([str(venv_python), str(script)], outside_dir)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(proc.stdout)


def test_wheel_ships_the_assembly_checker_and_its_vendored_contract(wheel):
    import zipfile

    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
    assert [f for f in REQUIRED_WHEEL_FILES if f not in names] == []
    assert not any("fixtures/" in n or n.endswith(".zip") for n in names)  # the reference package is a test fixture only


def test_unvendored_packages_are_not_importable_in_the_clean_venv(installed):
    assert installed["imports"] == {module: "ModuleNotFoundError" for module in MONOREPO_MODULES}


def test_the_installed_assembly_module_is_the_venv_copy_outside_the_repo(installed, venv_python):
    assembly_file = Path(installed["assembly_file"]).resolve()
    assert Path(installed["purelib"]).resolve() in assembly_file.parents  # the venv's site-packages
    assert venv_python.parent.parent.resolve() in assembly_file.parents
    assert REPO_ROOT not in assembly_file.parents
    assert not any(str(REPO_ROOT).lower() in p.lower() for p in installed["sys_path"] if p), installed["sys_path"]


def test_installed_check_of_the_reference_package(installed, outside_dir):
    assert installed["status"] == "PASS"
    assert installed["claim"] == "NO_INTERFERENCE"
    assert installed["checks"] == EXPECTED_CHECKS
    assert installed["reparsed_hash"] == installed["result_hash"]
    # the browser runtime's entry point gives the same document for the archive bytes and for the file map
    assert installed["envelopes"] == {label: {"ok": True, "status": "PASS", "same_file": True, "same_result": True}
                                      for label in ("bytes", "file_map")}
    result = AssemblyLocalValidationV1.model_validate_json((outside_dir / "py_result.json").read_bytes())  # vendored contract
    assert result.result_hash == installed["result_hash"]
    assert result.checker.runtime_kind.value == "PYTHON_LOCAL"
    assert result.package.package_content_hash == installed["package_content_hash"]


def test_installed_result_equals_the_in_repo_result(installed, outside_dir, in_repo_result_bytes):
    # the wheel is built from these sources, so the installed build id is the in-repo one ...
    assert installed["engine_build_id"] == engine_build_id()
    # ... and the whole result (build id, every check, issue, part, tolerance and result_hash) is byte-identical
    assert (outside_dir / "py_result.json").read_bytes() == in_repo_result_bytes
    in_repo = AssemblyLocalValidationV1.model_validate_json(in_repo_result_bytes)
    assert in_repo.checker.engine_build_id == engine_build_id()


# ---- the installed console script -------------------------------------------------------------------------------------------
def test_installed_console_script_verifies_the_package(venv_python, installed, outside_dir):
    proc = _run([str(_exe(venv_python)), "assembly", "verify", PACKAGE_NAME], outside_dir)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    report = json.loads(proc.stdout)
    assert report["status"] == "VERIFIED" and report["problems"] == []
    assert report["package_content_hash"] == installed["package_content_hash"]


def test_installed_console_script_validates_the_package(venv_python, installed, outside_dir, in_repo_result_bytes):
    proc = _run([str(_exe(venv_python)), "assembly", "validate", PACKAGE_NAME, "--out", "out.json"], outside_dir)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]  # PASS -> exit 0
    summary = json.loads(proc.stdout)
    assert summary["status"] == "PASS" and summary["interference"] == "NO_INTERFERENCE"
    assert summary["checks"] == EXPECTED_CHECKS
    assert summary["result_hash"] == installed["result_hash"]
    written = (outside_dir / "out.json").read_bytes()
    assert written == (outside_dir / "py_result.json").read_bytes() == in_repo_result_bytes  # console script == Python API == repo
    AssemblyLocalValidationV1.model_validate_json(written)
    # without --out the full document goes to stdout, same bytes (text mode on Windows only changes line endings)
    full = _run([str(_exe(venv_python)), "assembly", "validate", PACKAGE_NAME], outside_dir)
    assert full.returncode == 0, full.stderr[-2000:]
    assert full.stdout.encode("utf-8").replace(b"\r\n", b"\n") == written


def test_installed_console_script_rejects_a_tampered_package(venv_python, outside_dir):
    import zipfile

    tampered = outside_dir / "tampered"
    tampered.mkdir(exist_ok=True)
    with zipfile.ZipFile(outside_dir / PACKAGE_NAME) as z:
        for name in z.namelist():
            (tampered / name).parent.mkdir(parents=True, exist_ok=True)
            (tampered / name).write_bytes(z.read(name))
    scene = tampered / "scene.json"
    scene.write_bytes(scene.read_bytes().replace(b'"thickness_mm": 12.7', b'"thickness_mm": 12.8'))
    verify = _run([str(_exe(venv_python)), "assembly", "verify", "tampered"], outside_dir)
    assert verify.returncode == 66, verify.stdout + verify.stderr
    assert json.loads(verify.stdout)["problems"] == ["'scene.json': size or SHA-256 does not match the manifest"]
    validate = _run([str(_exe(venv_python)), "assembly", "validate", "tampered", "--out", "never.json"], outside_dir)
    assert validate.returncode == 65, validate.stdout + validate.stderr
    assert json.loads(validate.stdout)["error"]["code"] == "ASSEMBLY_PACKAGE_INVALID"
    assert not (outside_dir / "never.json").exists()
