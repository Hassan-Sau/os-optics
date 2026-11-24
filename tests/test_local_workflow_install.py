"""``pip install os-optics`` alone is sufficient for the local pure-optical design workflow.

Builds the wheel from the current sources, checks what it ships (the vendored public design bundle, the Optical
Simulation View, the catalog store / sync code -- and no catalog, nothing private), installs ONLY that wheel (plus its
PyPI dependencies) into a fresh venv, and then runs the whole local design workflow there from a directory outside the
repository, where ``shared_contracts``, ``opensci_data`` and ``opensci_optical_view`` are not importable. The public
optical catalog is synchronized data: the fresh install has none (``CATALOG_NOT_AVAILABLE``) until the pinned public
catalog release is imported into a store of its own with ``os-optics catalog import`` (no service is contacted; automatic
checks are off):

public catalog -> Project -> ONE real Engine run -> simulation result + ``LocalValidationRecordV1`` + Optical Simulation
View (scene JSON + GLB) -> run directory -> binding / cross-artifact verification, from Python and from the console
script. The emitted artifacts are scanned for private identifiers too. Mark: ``install``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from os_optics.local import example_project, lower_project
from os_optics.manifest import engine_build_id

from catalog_store_helpers import CATALOG_URL_VARIABLES, REAL_CATALOG

pytestmark = pytest.mark.install

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
CANONICAL_CATALOG = REAL_CATALOG
# The installed wheel builds and runs the same reference design (example_project(), products chosen by catalog query) as
# these sources from the same catalog release: same Project, same Engine input.
EXAMPLE_PROJECT = example_project(CANONICAL_CATALOG)
EXAMPLE_INPUT_HASH = lower_project(EXAMPLE_PROJECT, CANONICAL_CATALOG).engine_input_hash
INSTALLED_STORE = "catalog-store"

REQUIRED_WHEEL_FILES = (
    "os_optics/_vendor/MANIFEST.json",
    "os_optics/_vendor/shared_contracts/design_catalog/manifest.py",
    "os_optics/catalog/__init__.py",
    "os_optics/catalog/service.py",
    "os_optics/catalog/store.py",
    "os_optics/catalog/remote.py",
    "os_optics/deployment.toml",
    "os_optics/cli_catalog.py",
    "os_optics/_vendor/shared_contracts/optical_design/project.py",
    "os_optics/local/run.py",
    "os_optics/local/artifacts.py",
    "os_optics/validation/ray_paths.py",
    "os_optics/optical_view/builder.py",
    "os_optics/optical_view/glb.py",
    "os_optics/_vendor/shared_contracts/optical_view/scene.py",
    "os_optics/cli_view.py",
    "os_optics/viewer/__init__.py",
    "os_optics/viewer/raster.py",
    "os_optics/viewer/server.py",
    "os_optics/viewer/assets/index.html",
    "os_optics/viewer/assets/viewer.js",
    "os_optics/viewer/assets/viewer.css",
    "os_optics/viewer/assets/three.bundle.js",
    "os_optics/viewer/assets/THREE_LICENSE.txt",
)
RUN_FILE_NAMES = {"manifest.json", "project.json", "engine_request.json", "simulation_result.json", "ray_data.json", "local_validation.json",
                  "optical_scene.json", "optical_scene.glb"}
# Mechanical / supplier vocabulary that must not appear in any emitted artifact of a design-stage optical run.
MECHANICAL_WORDS = ("mount", "post", "holder", "breadboard", "stage", "supplier", "supplierpart", "fastener", "screw", "bracket")
# Module / file names that belong to a private catalog pipeline and must never be shipped.
FORBIDDEN_PATH_PARTS = ("private_source", "design_catalog_source", "projector", "leak", "supplier")
# Identifiers of private records (case-insensitive): the demo identity namespace, the field names of private catalog
# records and synthetic stand-ins for supplier and part names. Generic words such as "supplier" or "mount" legitimately
# appear in contract docstrings and field-forbidding code. (The scan for the real private identities runs where those
# are kept.)
PRIVATE_TOKENS = (
    "acme-optics",
    "xq-412b",
    "zk-75a",
    "os-demo",
    "supplier_realizations",
    "internal_commercial_sku",
    "private_source",
)
# The vendored Assembly Preview scene contract states the identity-namespace rule ("an OS-DEMO-* identity must be declared
# DEMO_PILOT_SKU"): the bare prefix followed by a wildcard or a closing quote is that rule, not a demo identity.
NAMESPACE_RULE = re.compile(r"os-demo-(?=[*'\"])")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("local_dist")
    proc = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out), str(ROOT)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = list(out.glob("os_optics-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="module")
def venv_python(wheel, tmp_path_factory) -> Path:
    venv = tmp_path_factory.mktemp("local_venv") / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    proc = subprocess.run([str(py), "-m", "pip", "install", "--only-binary=:all:", "--no-cache-dir", str(wheel)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, "pip install of the wheel failed (PyPI reachable?):\n" + proc.stdout[-2000:] + proc.stderr[-2000:]
    return py


@pytest.fixture(scope="module")
def outside_dir(tmp_path_factory) -> Path:
    work = tmp_path_factory.mktemp("outside_repo")
    assert REPO_ROOT not in work.resolve().parents, work
    return work


def _clean_env(cwd: Path, store: str = INSTALLED_STORE) -> dict[str, str]:
    """The environment of the clean install: its own catalog store under ``cwd``, automatic catalog checks off and no
    catalog service configured (nothing is fetched)."""

    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP", *CATALOG_URL_VARIABLES}}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["OPENSCI_CATALOG_DIR"] = str(cwd / store)
    env["OPENSCI_CATALOG_CHECK_INTERVAL"] = "0"
    return env


def _run(args: list[str], cwd: Path, timeout: int = 300, store: str = INSTALLED_STORE) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=_clean_env(cwd, store), capture_output=True, text=True, encoding="utf-8", timeout=timeout)


# ---- wheel content ---------------------------------------------------------------------------------------------------------------
def test_wheel_ships_the_public_design_bundle_and_no_catalog_and_nothing_private(wheel):
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        meta = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
    assert [f for f in REQUIRED_WHEEL_FILES if f not in names] == []
    top_level = {n.split("/", 1)[0] for n in names}
    assert {t for t in top_level if not t.endswith(".dist-info")} == {"os_optics"}, top_level
    assert not any(n.startswith(("shared_contracts/", "opensci_data/", "opensci_optical_view/")) for n in names)
    assert not any("opensci_data" in part for n in names for part in n.split("/")[:-1])
    assert [n for n in names if any(bad in n.lower() for bad in FORBIDDEN_PATH_PARTS)] == []
    requires = [line.split(":", 1)[1].strip() for line in meta.splitlines() if line.startswith("Requires-Dist:") and "extra ==" not in line]
    assert sorted(r.split("<")[0].split(">")[0].split("=")[0].split(";")[0].strip() for r in requires) == ["numpy", "pydantic", "scipy", "shapely"]
    # no catalog is bundled: the catalog is versioned data synchronized into a local store
    assert [n for n in names if n.endswith(".json") and n.startswith("os_optics/") and n != "os_optics/_vendor/MANIFEST.json"] == []


def test_no_text_file_in_the_wheel_contains_a_private_identifier(wheel):
    offenders: list[str] = []
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            data = z.read(name)
            try:
                text = data.decode("utf-8").lower()
            except UnicodeDecodeError:
                continue  # the wheel is pure Python + JSON; nothing binary is expected, and nothing binary can be read as text
            text = NAMESPACE_RULE.sub("<namespace-rule>", text)
            for token in PRIVATE_TOKENS:
                at = text.find(token)
                if at >= 0:
                    offenders.append(f"{name}: {token!r} in ...{text[max(0, at - 50):at + 50]!r}...")
    assert offenders == [], "\n".join(offenders)


# ---- clean install -------------------------------------------------------------------------------------------------------------
def test_unvendored_packages_are_not_importable_in_the_clean_venv(venv_python, outside_dir):
    for module in ("shared_contracts", "opensci_data", "opensci_optical_view"):
        proc = _run([str(venv_python), "-c", f"import {module}"], outside_dir)
        assert proc.returncode != 0 and f"ModuleNotFoundError: No module named '{module}'" in proc.stderr, (module, proc.stderr)


WORKFLOW_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    import os_optics
    from os_optics.manifest import engine_build_id
    from os_optics.local import (
        check_local_validation_binding, example_project, load_public_catalog, run_optical_project, verify_run_artifacts,
    )

    catalog = load_public_catalog()
    lenses = catalog.select(category="LENS", wavelength_nm=780)
    project = example_project()  # the reference design: its lens is chosen by catalog query
    lens = catalog.get(next(c for c in project.components if c.instance_id == "L1").product.optical_sku)
    project_file = Path("chain_l_like.json")
    project_file.write_text(json.dumps(project.payload(), indent=2, sort_keys=True), encoding="utf-8")

    run = run_optical_project(project, output_dir="run")
    scene = run.optical_view
    print(json.dumps({
        "engine_file": os_optics.__file__,
        "selected": [p.optical_sku for p in lenses],
        "lens_sku": lens.optical_sku,
        "lens_models": [m.model_kind for m in lens.simulation_models],
        "project": project.payload(),
        "status": run.status.value,
        "result_issue_codes": [issue.code.value for issue in run.result.issues],
        "record": run.local_validation.payload(),
        "binding": list(check_local_validation_binding(run.local_validation, project)),
        "installed_engine_build_id": engine_build_id(),
        "artifacts": {role: str(path) for role, path in run.artifacts.paths.items()},
        "verify": list(verify_run_artifacts("run")),
        "scene": scene.payload(),
        "scene_hash": scene.content_hash(),
        "ray_paths": {"input_hash": run.ray_paths.input_hash, "trace_hash": run.ray_paths.trace_hash,
                      "paths": len(run.ray_paths.paths), "version": run.ray_paths.ray_paths_version},
        "result_trace_hash": run.result.manifest.trace_hash,
    }, sort_keys=True))
    """
)


def test_a_fresh_install_has_no_catalog_until_one_is_imported(venv_python, outside_dir):
    """Without a stored release (and with automatic access off) the catalog commands say what to do."""

    exe = venv_python.parent / ("os-optics.exe" if os.name == "nt" else "os-optics")
    status = _run([str(exe), "catalog", "status"], outside_dir, store="empty-store")
    assert status.returncode == 1 and json.loads(status.stdout)["available"] is False, status.stdout + status.stderr
    example = _run([str(exe), "project", "example"], outside_dir, store="empty-store")
    assert example.returncode == 65, example.stdout + example.stderr
    (problem,) = json.loads(example.stdout)["problems"]
    assert problem["code"] == "CATALOG_NOT_AVAILABLE" and "os-optics catalog import" in problem["message"]
    doctor = _run([str(exe), "doctor"], outside_dir, store="empty-store")
    report = json.loads(doctor.stdout)
    # (the catalog is its own readiness fact: the interpreter can run a Project, the catalog work cannot start)
    assert doctor.returncode == 0 and report["catalog"]["ok"] is False
    assert (report["ready"]["local_design"], report["ready"]["catalog"]) == (True, False)
    step = next(s for s in report["next_steps"] if s["check"] == "catalog")
    assert step["commands"] == ["os-optics catalog sync", "os-optics catalog import <catalog file>"]


@pytest.fixture(scope="module")
def installed_catalog(venv_python, outside_dir) -> dict:
    """The seed release imported into the clean install's own store (``os-optics catalog import``)."""

    exe = venv_python.parent / ("os-optics.exe" if os.name == "nt" else "os-optics")
    proc = _run([str(exe), "catalog", "import", str(CANONICAL_CATALOG)], outside_dir)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    status = json.loads(proc.stdout)
    assert status["active"]["catalog_version"] == json.loads(CANONICAL_CATALOG.read_bytes())["catalog_version"]
    verify = _run([str(exe), "catalog", "verify", "--deep"], outside_dir)
    assert verify.returncode == 0 and json.loads(verify.stdout)["ok"] is True, verify.stdout
    return status


@pytest.fixture(scope="module")
def workflow(venv_python, outside_dir, installed_catalog) -> dict:
    script = outside_dir / "local_workflow.py"
    script.write_text(WORKFLOW_SCRIPT, encoding="utf-8")
    proc = _run([str(venv_python), str(script)], outside_dir)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(proc.stdout)


def test_mode_a_needs_no_optional_extra(venv_python, workflow, outside_dir):
    """The plain wheel (no ``[protected]`` extra, so no ``cryptography``) runs the local design workflow and still imports the
    delivery-opening and artifact-upload modules; nothing of them is loaded by ``os_optics.local``."""

    code = ("import importlib.util, json, sys; import os_optics, os_optics.local; "
            "loaded = sorted(m for m in sys.modules if m in ('os_optics.protected_delivery', 'os_optics.client_artifacts')); "
            "import os_optics.protected_delivery as pd, os_optics.client_artifacts; "
            "print(json.dumps({'crypto_spec': importlib.util.find_spec('cryptography') is not None, 'loaded': loaded, "
            "'available': pd.cryptography_available()}))")
    proc = _run([str(venv_python), "-c", code], outside_dir)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout) == {"crypto_spec": False, "loaded": [], "available": False}
    # The real public glass has no stated bulk loss; geometric targets of the local workflow remain decidable.
    assert workflow["status"] == "UNKNOWN"
    assert {t["status"] for t in workflow["record"]["simulation"]["target_outcomes"]} == {"PASS"}


def test_installed_workflow_uses_the_installed_copy_outside_the_repo(workflow, venv_python):
    engine_file = Path(workflow["engine_file"]).resolve()
    assert venv_python.parent.parent.resolve() in engine_file.parents  # the venv's site-packages, not the source tree
    assert REPO_ROOT not in engine_file.parents


def test_installed_catalog_discovery(workflow):
    assert workflow["lens_sku"] in workflow["selected"]
    assert workflow["lens_models"] == ["SURFACE_SEQUENCE"]


def test_installed_workflow_builds_the_reference_project_on_the_installed_release(workflow):
    """The installed wheel builds the same reference design as these sources, pinned to the imported catalog release."""
    assert workflow["project"] == EXAMPLE_PROJECT.payload()
    lens = next(c for c in workflow["project"]["components"] if c["instance_id"] == "L1")
    assert lens["product"]["catalog_version"] == json.loads(CANONICAL_CATALOG.read_text(encoding="utf-8"))["catalog_version"]


def test_installed_workflow_runs_to_a_bound_pass_record(workflow):
    record = workflow["record"]
    assert workflow["status"] == "UNKNOWN"
    assert record["validation_status"] == "UNKNOWN" and record["simulation"]["status"] == "UNKNOWN"
    assert {t["target_id"]: t["status"] for t in record["simulation"]["target_outcomes"]} == {"focus_position": "PASS", "spot_radius": "PASS"}
    assert "BULK_LOSS_NOT_ESTABLISHED" in workflow["result_issue_codes"]
    assert any(o["quantity_state"] == "UNKNOWN" for o in record["simulation"]["observation_quantities"])
    assert workflow["binding"] == []
    assert record["input_hash"]["digest"] == EXAMPLE_INPUT_HASH
    assert record["runtime"]["engine_build_id"] == workflow["installed_engine_build_id"]
    # the wheel is built from these sources, so the installed build id is the in-repo one
    assert workflow["installed_engine_build_id"] == engine_build_id()
    assert [(p["instance_id"], p["optical_sku"], p["simulation_model"]["model_kind"]) for p in record["simulated_parts"]] == [
        ("L1", workflow["lens_sku"], "SURFACE_SEQUENCE")]


def test_installed_console_script_catalog_and_project_run(venv_python, workflow, outside_dir):
    exe = venv_python.parent / ("os-optics.exe" if os.name == "nt" else "os-optics")
    assert exe.is_file(), exe
    listing = _run([str(exe), "catalog", "list", "--category", "LENS", "--wavelength-nm", "780"], outside_dir)
    assert listing.returncode == 0, listing.stderr
    assert workflow["lens_sku"] in [row["optical_sku"] for row in json.loads(listing.stdout)["products"]]
    record_file = outside_dir / "record.json"
    proc = _run([str(exe), "project", "run", "chain_l_like.json", "-o", str(record_file)], outside_dir)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    record = json.loads(record_file.read_text(encoding="utf-8"))
    assert record["validation_status"] == workflow["record"]["validation_status"] == "UNKNOWN"
    assert record["input_hash"]["digest"] == EXAMPLE_INPUT_HASH
    assert record["project"] == workflow["record"]["project"]


def test_installed_doctor_reports_a_ready_local_workflow(venv_python, outside_dir, installed_catalog):
    """`os-optics doctor` from the clean install: supported Python, required packages in range, the imported catalog, and
    (the wheel was installed without the [protected] extra) the exact command that adds it; exit 0."""
    exe = venv_python.parent / ("os-optics.exe" if os.name == "nt" else "os-optics")
    env = {k: v for k, v in _clean_env(outside_dir).items() if k not in {"OPENSCI_API_URL", "OPENSCI_TOKEN", "OPENSCI_TOKEN_FILE"}}
    for args in ([str(exe), "doctor"], [str(venv_python), "-m", "os_optics", "doctor"]):
        proc = subprocess.run(args, cwd=outside_dir, env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        report = json.loads(proc.stdout)
        assert report["python"]["supported"] is True and report["install"]["kind"] == "virtualenv"
        assert report["engine"]["distribution_installed"] is True and report["engine"]["build_id"] == engine_build_id()
        assert all(entry["ok"] for entry in report["packages"].values()) and set(report["packages"]) == {
            "numpy", "scipy", "pydantic", "shapely"}
        assert report["catalog"]["ok"] and report["ready"]["local_design"] is True and report["ready"]["catalog"] is True
        assert report["catalog"]["catalog_version"] == installed_catalog["active"]["catalog_version"]
        assert report["protected_extra"]["installed"] is False
        extra = next(step for step in report["next_steps"] if step["check"] == "protected_extra")
        (command,) = extra["commands"]
        assert command.endswith(f' -m pip install "os-optics[protected]=={report["engine"]["version"]}"')
        assert report["python"]["executable"] in command


# ---- the integrated run directory and Optical Simulation View ---------------------------------------------------------------------
def test_installed_run_writes_the_complete_run_directory(workflow, outside_dir):
    run_dir = outside_dir / "run"
    assert {p.name for p in run_dir.iterdir()} == RUN_FILE_NAMES
    assert {Path(p).name for role, p in workflow["artifacts"].items()} == RUN_FILE_NAMES
    assert workflow["verify"] == []
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]["validation_status"] == "UNKNOWN" and manifest["binding"]["cross_artifact_problems"] == []
    assert manifest["engine_run"]["engine_build_id"] == workflow["installed_engine_build_id"]


def test_installed_scene_and_record_come_from_the_same_engine_run(workflow):
    scene, record = workflow["scene"], workflow["record"]
    assert scene["engine_run"]["input_hash"] == record["input_hash"]
    assert scene["engine_run"]["result_hash"] == record["result_hash"]
    assert scene["project"] == record["project"]
    assert scene["engine_run"]["trace_hash"] == workflow["result_trace_hash"] == workflow["ray_paths"]["trace_hash"]
    assert workflow["ray_paths"]["input_hash"] == record["input_hash"]["digest"] == EXAMPLE_INPUT_HASH


def test_installed_view_shows_source_neutral_lens_engine_beam_and_target_plane(workflow, outside_dir):
    scene = workflow["scene"]
    assert scene["view_kind"] == "OPTICAL_SIMULATION_VIEW"
    assert [(e["element_id"], e["geometry"]["kind"]) for e in scene["elements"]] == [("SRC", "SOURCE_EMITTER"), ("L1", "OPTIC_BODY")]
    lens = scene["elements"][1]
    assert lens["optical_sku"] == workflow["lens_sku"] and lens["geometry"]["basis"] == "PUBLIC_CATALOG_DESIGN_GEOMETRY"
    (plane,) = scene["observation_planes"]
    assert plane["kind"] == "TARGET" and plane["footprints"]
    (beam,) = scene["beam_paths"]
    assert beam["source_element_id"] == "SRC" and len(beam["chief_ray"]["points_mm"]) >= 3 and beam["pupil_rays"]
    glb = (outside_dir / "run" / "optical_scene.glb").read_bytes()
    assert glb[:4] == b"glTF" and int.from_bytes(glb[8:12], "little") == len(glb)
    assert workflow["scene_hash"].encode() in glb


def test_installed_artifacts_carry_no_private_or_mechanical_identity(outside_dir):
    run_dir = outside_dir / "run"
    offenders = []
    for path in sorted(run_dir.iterdir()):
        data = path.read_bytes()
        if path.suffix == ".glb":  # the JSON chunk is the only text in a GLB
            length = int.from_bytes(data[12:16], "little")
            data = data[20:20 + length]
        text = data.decode("utf-8").lower().replace("design-stage", "")
        offenders += [f"{path.name}: {token!r}" for token in PRIVATE_TOKENS if token in text]
        offenders += [f"{path.name}: {word!r}" for word in MECHANICAL_WORDS if re.search(rf"(?<![a-z0-9_]){word}s?(?![a-z0-9_])", text)]
    assert offenders == [], offenders


def test_installed_console_script_writes_and_verifies_a_run_directory(venv_python, workflow, outside_dir):
    exe = venv_python.parent / ("os-optics.exe" if os.name == "nt" else "os-optics")
    proc = _run([str(exe), "project", "run", "chain_l_like.json", "--out-dir", "cli_run"], outside_dir)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    summary = json.loads(proc.stdout)
    assert summary["status"] == "UNKNOWN" and summary["exit_contract"] == "requirements"
    assert {t["target_id"]: t["status"] for t in summary["targets"]} == {"focus_position": "PASS", "spot_radius": "PASS"}
    assert Path(summary["artifacts"]["optical_scene_glb"]).name == "optical_scene.glb"
    for name in RUN_FILE_NAMES:  # deterministic: the console script and the Python API write the same bytes
        assert (outside_dir / "cli_run" / name).read_bytes() == (outside_dir / "run" / name).read_bytes(), name
    verify = _run([str(exe), "project", "verify", "cli_run"], outside_dir)
    assert verify.returncode == 0 and json.loads(verify.stdout)["status"] == "CONSISTENT", verify.stdout
    help_text = _run([str(exe), "--help"], outside_dir).stdout
    assert "--out-dir" in help_text and "optical_scene.glb" in help_text


def test_installed_viewer_writes_snapshots_and_a_self_contained_page(venv_python, workflow, outside_dir):
    """The wheel ships the viewer and its assets: ``os-optics view`` works from the installed copy alone."""

    exe = venv_python.parent / ("os-optics.exe" if os.name == "nt" else "os-optics")
    proc = _run([str(exe), "view", "run", "--snapshot", "shots", "--size", "400x300"], outside_dir)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert {p.name for p in (outside_dir / "shots").iterdir()} == {"iso.png", "top.png", "front.png", "side.png", "scene_summary.json",
                                                                   "legend.md"}
    summary = json.loads((outside_dir / "shots" / "scene_summary.json").read_text(encoding="utf-8"))
    assert summary["units"] == {"length": "mm"} and "LENS" in summary["parts_by_class"]
    page = _run([str(exe), "view", "run", "--write-html", "view.html"], outside_dir)
    assert page.returncode == 0, page.stderr[-2000:]
    html = (outside_dir / "view.html").read_text(encoding="utf-8")
    assert "OPENSCI_THREE" in html and 'id="opensci-view-document"' in html
