"""``pip install os-optics`` alone is sufficient for the local Detailed Assembly Drawing (plan v0.12 stage six).

Builds the wheel from the current sources, installs ONLY that wheel (plus its declared PyPI dependencies) into a fresh
venv outside the repository, copies the reference Assembly Preview + Assembly Detail pair next to it and then, from a
directory outside the repository with a scrubbed environment (no ``PYTHONPATH``, no user site) where
the unvendored ``shared_contracts`` package is not importable:

* Python: ``load_assembly_detail_package`` -> ``generate_assembly_drawing`` -> ``write`` -> ``verify_assembly_drawing``
  (and the browser (Pyodide) runtime entry points ``json_api.verify_assembly_detail_package`` / ``json_api.generate_assembly_drawing``);
* the console script: ``os-optics assembly draw`` and ``os-optics assembly verify-drawing``.

The installed outputs must equal the in-repo outputs byte for byte (the wheel carries the same ``engine_build_id``).

The path after a commercial resolution: without the ``protected`` extra the wheel still imports
``os_optics.local``, ``os_optics.protected_delivery`` and ``os_optics.client_artifacts`` (``cryptography``
absent; opening a delivery refuses with ``CRYPTO_UNAVAILABLE`` naming the extra; artifact uploads still build). A second
venv installs ``<wheel>[protected]`` and opens the committed protected delivery vectors and the sealed chain-L
reference pair through ``os_optics.protected_delivery``, draws, and builds both uploads, byte-identical to the
in-repo uploads. Mark: ``install``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from os_optics.assembly_drawing import generate_assembly_drawing
from os_optics.manifest import engine_build_id

pytestmark = pytest.mark.install

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
FIXTURES = ROOT / "tests/fixtures/assembly_drawing"
PREVIEW = "reference.assembly-preview.zip"
DETAIL = "reference.assembly-detail.json"
MONOREPO_MODULES = ("shared_contracts",)
REQUIRED_WHEEL_FILES = (
    "os_optics/assembly_drawing/__init__.py",
    "os_optics/assembly_drawing/generate.py",
    "os_optics/assembly_drawing/hidden_lines.py",
    "os_optics/assembly_drawing/sheet.py",
    "os_optics/assembly_drawing/pdf.py",
    "os_optics/assembly_drawing/svg.py",
    "os_optics/_vendor/shared_contracts/assembly_detail/__init__.py",
    "os_optics/_vendor/shared_contracts/assembly_detail/detail.py",
    "os_optics/_vendor/shared_contracts/assembly_detail/drawing.py",
    "os_optics/_vendor/shared_contracts/assembly_detail/verify.py",
)
DRAWING_FILES = ("assembly_drawing.svg", "assembly_drawing.pdf", "drawing_result.json", "drawing_manifest.json")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("drawing_dist")
    proc = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out), str(ROOT)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (whl,) = out.glob("os_optics-*.whl")
    return whl


@pytest.fixture(scope="module")
def venv_python(wheel, tmp_path_factory) -> Path:
    venv = tmp_path_factory.mktemp("drawing_venv") / "venv"
    assert REPO_ROOT not in venv.resolve().parents, venv
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    proc = subprocess.run([str(py), "-m", "pip", "install", "--only-binary=:all:", "--no-cache-dir", str(wheel)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, "pip install of the wheel failed (PyPI reachable?):\n" + proc.stdout[-2000:] + proc.stderr[-2000:]
    return py


@pytest.fixture(scope="module")
def outside_dir(tmp_path_factory) -> Path:
    work = tmp_path_factory.mktemp("drawing_outside_repo")
    assert REPO_ROOT not in work.resolve().parents, work
    shutil.copyfile(FIXTURES / "reference.assembly-preview.zip", work / PREVIEW)
    shutil.copyfile(FIXTURES / "reference.assembly-detail.json", work / DETAIL)
    return work


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP"}}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(args: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=_clean_env(), capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def _exe(venv_python: Path) -> Path:
    exe = venv_python.parent / ("os-optics.exe" if os.name == "nt" else "os-optics")
    assert exe.is_file(), exe
    return exe


INSTALLED_SCRIPT = textwrap.dedent(
    f"""
    import importlib, json, sys, sysconfig
    from pathlib import Path

    imports = {{}}
    for module in {MONOREPO_MODULES!r}:
        try:
            importlib.import_module(module)
            imports[module] = "IMPORTED"
        except ImportError as exc:
            imports[module] = type(exc).__name__

    import os_optics.assembly_drawing as drawing_module
    from os_optics.assembly import load_assembly_preview
    from os_optics.assembly_drawing import generate_assembly_drawing, load_assembly_detail_package, verify_assembly_drawing
    from os_optics.local import json_api
    from os_optics.manifest import engine_build_id

    preview = load_assembly_preview({PREVIEW!r})
    detail = load_assembly_detail_package({DETAIL!r}, preview)
    drawing = generate_assembly_drawing(preview, detail)
    out = drawing.write("py_drawing")
    reply = json_api.generate_assembly_drawing(Path({PREVIEW!r}).read_bytes(), Path({DETAIL!r}).read_bytes())
    check = json_api.verify_assembly_detail_package(Path({PREVIEW!r}).read_bytes(), Path({DETAIL!r}).read_bytes())
    print(json.dumps({{
        "imports": imports,
        "module_file": drawing_module.__file__,
        "purelib": sysconfig.get_paths()["purelib"],
        "sys_path": sys.path,
        "engine_build_id": engine_build_id(),
        "status": drawing.result.drawing_status.value,
        "delivery_ready": drawing.result.delivery_ready,
        "verify": list(verify_assembly_drawing(out)),
        "json_api_ok": reply["ok"],
        "json_api_same_files": reply["files"] == dict(drawing.files),
        "detail_ok": check["ok"] and len(check["dimensions"]) == len(detail.detail.dimensions),
    }}, sort_keys=True))
    """
)


@pytest.fixture(scope="module")
def installed(venv_python, outside_dir) -> dict:
    script = outside_dir / "drawing_check.py"
    script.write_text(INSTALLED_SCRIPT, encoding="utf-8")
    proc = _run([str(venv_python), str(script)], outside_dir)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def in_repo_files() -> dict[str, bytes]:
    return dict(generate_assembly_drawing(FIXTURES / "reference.assembly-preview.zip",
                                          FIXTURES / "reference.assembly-detail.json").files)


def test_wheel_ships_the_drawing_generator_and_its_vendored_contract(wheel):
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
    assert [f for f in REQUIRED_WHEEL_FILES if f not in names] == []
    assert not any("fixtures/" in n or n.endswith((".zip", ".svg", ".pdf")) for n in names)


def test_unvendored_packages_are_not_importable_and_the_module_is_the_venv_copy(installed, venv_python):
    assert installed["imports"] == {module: "ModuleNotFoundError" for module in MONOREPO_MODULES}
    module_file = Path(installed["module_file"]).resolve()
    assert Path(installed["purelib"]).resolve() in module_file.parents
    assert venv_python.parent.parent.resolve() in module_file.parents and REPO_ROOT not in module_file.parents
    assert not any(str(REPO_ROOT).lower() in p.lower() for p in installed["sys_path"] if p), installed["sys_path"]


def test_installed_drawing_equals_the_in_repo_drawing(installed, outside_dir, in_repo_files):
    assert installed["status"] == "DETAILED_ASSEMBLY_DRAWING" and installed["delivery_ready"] is True
    assert installed["verify"] == [] and installed["json_api_ok"] and installed["json_api_same_files"] and installed["detail_ok"]
    assert installed["engine_build_id"] == engine_build_id()
    assert {name: (outside_dir / "py_drawing" / name).read_bytes() for name in DRAWING_FILES} == in_repo_files


def test_installed_console_script_draws_and_verifies(venv_python, installed, outside_dir, in_repo_files):
    proc = _run([str(_exe(venv_python)), "assembly", "draw", PREVIEW, "--detail", DETAIL, "--out", "cli_drawing"], outside_dir)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    summary = json.loads(proc.stdout)
    assert summary["status"] == "DETAILED_ASSEMBLY_DRAWING" and summary["delivery_ready"] is True
    assert {name: (outside_dir / "cli_drawing" / name).read_bytes() for name in DRAWING_FILES} == in_repo_files
    verify = _run([str(_exe(venv_python)), "assembly", "verify-drawing", "cli_drawing"], outside_dir)
    assert verify.returncode == 0 and json.loads(verify.stdout)["status"] == "VERIFIED"
    (outside_dir / "cli_drawing" / "assembly_drawing.svg").write_bytes(b"<svg/>")
    tampered = _run([str(_exe(venv_python)), "assembly", "verify-drawing", "cli_drawing"], outside_dir)
    assert tampered.returncode == 66 and json.loads(tampered.stdout)["status"] == "INVALID"


# ---- protected delivery opening and client artifact uploads from the installed wheel ------------------------------------
PROTECTED_WHEEL_FILES = (
    "os_optics/protected_delivery.py",
    "os_optics/client_artifacts.py",
    "os_optics/_vendor/shared_contracts/protected_assembly_delivery/__init__.py",
    "os_optics/_vendor/shared_contracts/protected_assembly_delivery/_common.py",
    "os_optics/_vendor/shared_contracts/protected_assembly_delivery/envelope.py",
    "os_optics/_vendor/shared_contracts/protected_assembly_delivery/detail_envelope.py",
    "os_optics/_vendor/shared_contracts/client_assembly_artifact/__init__.py",
    "os_optics/_vendor/shared_contracts/client_assembly_artifact/manifest.py",
)
DELIVERY_FILES = ("preview-delivery.json", "detail-delivery.json", "preview-vector.json", "detail-vector.json")


def test_wheel_ships_the_delivery_and_artifact_modules_and_declares_the_protected_extra(wheel):
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        meta = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
    assert [f for f in PROTECTED_WHEEL_FILES if f not in names] == []
    assert "Provides-Extra: protected" in meta
    requires = [line.split(":", 1)[1].strip() for line in meta.splitlines() if line.startswith("Requires-Dist:")]
    crypto = [r for r in requires if r.startswith("cryptography")]
    assert crypto and all("extra ==" in r for r in crypto) and any("'protected'" in r or '"protected"' in r for r in crypto)


NO_EXTRA_SCRIPT = textwrap.dedent(
    f"""
    import importlib, json
    from pathlib import Path

    try:
        importlib.import_module("cryptography")
        crypto = "IMPORTED"
    except ImportError as exc:
        crypto = type(exc).__name__
    import os_optics.local
    from os_optics.assembly_drawing import generate_assembly_drawing
    from os_optics.client_artifacts import drawing_artifact_upload, validation_artifact_upload
    from os_optics.protected_delivery import ProtectedDeliveryError, cryptography_available, open_preview_delivery

    document = json.loads(Path("preview-vector.json").read_text(encoding="utf-8"))
    try:
        open_preview_delivery(document["delivery_document"])
        refusal = None
    except ProtectedDeliveryError as exc:
        refusal = {{"code": exc.code.value, "text": str(exc)}}
    drawing = generate_assembly_drawing({PREVIEW!r}, {DETAIL!r})
    uploads = [validation_artifact_upload({PREVIEW!r}, drawing.validation), drawing_artifact_upload(drawing)]
    print(json.dumps({{"cryptography": crypto, "available": cryptography_available(), "refusal": refusal,
                      "catalog": os_optics.local.load_public_catalog().catalog_version,
                      "manifests": [u.manifest_json_bytes().decode("utf-8") for u in uploads]}}, sort_keys=True))
    """
)

PROTECTED_SCRIPT = textwrap.dedent(
    f"""
    import importlib, json
    from pathlib import Path

    imports = {{}}
    for module in {MONOREPO_MODULES!r}:
        try:
            importlib.import_module(module)
            imports[module] = "IMPORTED"
        except ImportError as exc:
            imports[module] = type(exc).__name__

    import cryptography
    import os_optics.protected_delivery as pd
    from os_optics._vendor.shared_contracts.protected_assembly_delivery import b64url_decode, parse_utc
    from os_optics.assembly import load_assembly_preview
    from os_optics.assembly_drawing import generate_assembly_drawing
    from os_optics.client_artifacts import drawing_artifact_upload, validation_artifact_upload

    preview_vector = json.loads(Path("preview-vector.json").read_text(encoding="utf-8"))
    detail_vector = json.loads(Path("detail-vector.json").read_text(encoding="utf-8"))
    opened_preview = pd.open_preview_delivery(json.dumps(preview_vector["delivery_document"]), now=parse_utc(preview_vector["now"]))
    vector_preview = load_assembly_preview(b64url_decode(detail_vector["preview_archive"]))
    opened_detail = pd.open_detail_delivery(detail_vector["delivery_document"], preview=vector_preview,
                                            now=parse_utc(detail_vector["now"]))

    preview = pd.open_preview_delivery(Path("preview-delivery.json").read_bytes())
    detail = pd.open_detail_delivery(Path("detail-delivery.json").read_bytes(), preview=preview)
    drawing = generate_assembly_drawing(preview, detail)
    uploads = [validation_artifact_upload(preview, drawing.validation), drawing_artifact_upload(drawing)]
    print(json.dumps({{
        "imports": imports,
        "module_file": pd.__file__,
        "cryptography": cryptography.__version__,
        "vector_preview_ok": opened_preview.archive_bytes() == b64url_decode(preview_vector["plaintext"]),
        "vector_detail_ok": opened_detail.detail_bytes == b64url_decode(detail_vector["plaintext"]),
        "status": drawing.result.drawing_status.value,
        "files_equal_reference": dict(drawing.files) == dict(generate_assembly_drawing({PREVIEW!r}, {DETAIL!r}).files),
        "manifests": [u.manifest_json_bytes().decode("utf-8") for u in uploads],
        "fields": [[name, part[0], part[2], len(part[1])] for name, part in uploads[1].multipart_fields()],
        "cwd_files": sorted(p.name for p in Path(".").iterdir()),
    }}, sort_keys=True))
    """
)


def _write_delivery_files(directory: Path) -> None:
    """What the scripts read: the sealed chain-L reference pair as delivery documents (valid for the next minutes, so
    written right before a script runs) and the protected delivery vectors (delivery document + fixed ``now`` +
    plaintext; the detail vector also carries its preview archive)."""

    from delivery_helpers import DETAIL_VECTOR_FILE, PREVIEW_VECTOR_FILE, load_vector, sealed_reference_deliveries, vector_document

    preview_delivery, detail_delivery = sealed_reference_deliveries()
    preview_vector, detail_vector = load_vector(PREVIEW_VECTOR_FILE), load_vector(DETAIL_VECTOR_FILE)
    documents = {
        "preview-delivery.json": preview_delivery,
        "detail-delivery.json": detail_delivery,
        "preview-vector.json": {"delivery_document": vector_document(preview_vector, "ASSEMBLY_PREVIEW"),
                                "now": preview_vector["now"], "plaintext": preview_vector["plaintext"]},
        "detail-vector.json": {"delivery_document": vector_document(detail_vector, "ASSEMBLY_DETAIL"), "now": detail_vector["now"],
                               "plaintext": detail_vector["plaintext"], "preview_archive": detail_vector["preview_archive"]},
    }
    for name, document in documents.items():
        (directory / name).write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture(scope="module")
def in_repo_uploads() -> list[str]:
    from os_optics.client_artifacts import drawing_artifact_upload, validation_artifact_upload

    drawing = generate_assembly_drawing(FIXTURES / "reference.assembly-preview.zip",
                                        FIXTURES / "reference.assembly-detail.json")
    uploads = [validation_artifact_upload(FIXTURES / "reference.assembly-preview.zip", drawing.validation),
               drawing_artifact_upload(drawing)]
    return [u.manifest_json_bytes().decode("utf-8") for u in uploads]


@pytest.fixture(scope="module")
def protected_venv_python(wheel, tmp_path_factory) -> Path:
    venv = tmp_path_factory.mktemp("protected_venv") / "venv"
    assert REPO_ROOT not in venv.resolve().parents, venv
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    proc = subprocess.run([str(py), "-m", "pip", "install", "--only-binary=:all:", "--no-cache-dir", f"{wheel}[protected]"],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, "pip install of wheel[protected] failed (PyPI reachable?):\n" + proc.stdout[-2000:] + proc.stderr[-2000:]
    return py


def test_without_the_extra_the_wheel_imports_everything_and_names_the_extra(venv_python, installed, outside_dir, in_repo_uploads):
    _write_delivery_files(outside_dir)
    script = outside_dir / "no_extra_check.py"
    script.write_text(NO_EXTRA_SCRIPT, encoding="utf-8")
    proc = _run([str(venv_python), str(script)], outside_dir)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout)
    assert result["cryptography"] == "ModuleNotFoundError" and result["available"] is False
    assert result["refusal"]["code"] == "CRYPTO_UNAVAILABLE" and 'pip install "os-optics[protected]"' in result["refusal"]["text"]
    assert result["catalog"]
    assert result["manifests"] == in_repo_uploads  # building uploads needs no cryptography


def test_with_the_extra_the_wheel_opens_deliveries_draws_and_builds_uploads(protected_venv_python, tmp_path_factory, in_repo_uploads):
    work = tmp_path_factory.mktemp("protected_outside_repo")
    assert REPO_ROOT not in work.resolve().parents, work
    shutil.copyfile(FIXTURES / "reference.assembly-preview.zip", work / PREVIEW)
    shutil.copyfile(FIXTURES / "reference.assembly-detail.json", work / DETAIL)
    _write_delivery_files(work)
    script = work / "protected_check.py"
    script.write_text(PROTECTED_SCRIPT, encoding="utf-8")
    proc = _run([str(protected_venv_python), str(script)], work)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout)
    assert result["imports"] == {module: "ModuleNotFoundError" for module in MONOREPO_MODULES}
    module_file = Path(result["module_file"]).resolve()
    assert protected_venv_python.parent.parent.resolve() in module_file.parents and REPO_ROOT not in module_file.parents
    assert result["cryptography"].startswith("50.")
    assert result["vector_preview_ok"] and result["vector_detail_ok"]
    assert result["status"] == "DETAILED_ASSEMBLY_DRAWING" and result["files_equal_reference"]
    assert result["manifests"] == in_repo_uploads
    assert [f[:3] for f in result["fields"]] == [["manifest", "manifest.json", "application/json"],
                                                 ["file", "assembly_drawing.pdf", "application/pdf"],
                                                 ["file", "assembly_drawing.svg", "image/svg+xml"],
                                                 ["file", "drawing_manifest.json", "application/json"],
                                                 ["file", "drawing_result.json", "application/json"]]
    # nothing decrypted was written next to the deliveries
    assert result["cwd_files"] == sorted([PREVIEW, DETAIL, "protected_check.py", *DELIVERY_FILES])
