"""``os-optics doctor``: the installation report an Agent reads first (and it never reveals a credential)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from os_optics import __version__, cli_doctor
from os_optics._version import MIN_PYTHON
from os_optics.cli import main
from os_optics.deployment import default_service_origin

TOKEN = "oat_" + "0123456789abcdef" * 2 + "_" + "fedcba9876543210" * 4
ENV_NAMES = ("OPENSCI_API_URL", "OPENSCI_TOKEN", "OPENSCI_TOKEN_FILE")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No commercial configuration and an empty home directory (so ~/.opensci/token does not exist)."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _doctor(capsys) -> tuple[int, dict, str]:
    code = main(["doctor"])
    out = capsys.readouterr().out
    return code, json.loads(out), out


def _checks(report: dict) -> dict[str, dict]:
    return {step["check"]: step for step in report["next_steps"]}


def test_report_of_a_working_installation(clean_env, capsys):
    code, report, _ = _doctor(capsys)
    assert code == 0
    assert report["schema"] == "opensci.engine.doctor" and report["schema_version"] == "2"
    assert report["engine"]["version"] == __version__ and report["engine"]["build_id"].startswith(__version__ + "+")
    python = report["python"]
    assert python["supported"] is True and python["version"] == ".".join(map(str, sys.version_info[:3]))
    assert python["requires"] == ">={}.{}".format(*MIN_PYTHON)
    assert set(report["packages"]) == {"numpy", "scipy", "pydantic", "shapely"}
    assert all(entry["ok"] and entry["installed"] for entry in report["packages"].values())
    assert report["protected_extra"]["installed"] is True  # the test extra installs cryptography
    assert report["catalog"]["ok"] and report["catalog"]["products"] > 0 and report["catalog"]["catalog_version"]
    assert report["platform"]["sys_platform"] == sys.platform
    assert report["ready"] == {"local_design": True, "catalog": True, "protected_deliveries": True,
                               "commercial_configuration": False}
    steps = _checks(report)
    assert set(steps) == {"api_url", "token"}  # only the commercial configuration is missing
    assert all(step["commands"] for step in steps.values())
    assert report["commercial"]["token"]["file"] == str(clean_env / ".opensci" / "token")


def test_token_from_the_environment_is_never_printed(clean_env, monkeypatch, capsys):
    monkeypatch.setenv("OPENSCI_TOKEN", TOKEN)
    monkeypatch.setenv("OPENSCI_API_URL", "http://127.0.0.1:8750")
    code, report, out = _doctor(capsys)
    assert code == 0
    assert TOKEN not in out and TOKEN.split("_")[1] not in out and TOKEN.split("_")[2] not in out
    assert report["commercial"]["token"] == {"configured": True, "source": "OPENSCI_TOKEN", "file": None, "well_formed": True}
    assert report["commercial"]["api_url"] == {"configured": True, "value": "http://127.0.0.1:8750", "valid": True}
    assert report["ready"]["commercial_configuration"] is True
    assert _checks(report)["commercial_access"]["commands"] == ["os-optics account status"]


@pytest.mark.parametrize("via", ["OPENSCI_TOKEN_FILE", "default_token_file"])
def test_token_file_sources(clean_env, monkeypatch, capsys, tmp_path, via):
    path = tmp_path / "secret" / "token" if via == "OPENSCI_TOKEN_FILE" else clean_env / ".opensci" / "token"
    path.parent.mkdir(parents=True)
    path.write_text(TOKEN + "\n", encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    if via == "OPENSCI_TOKEN_FILE":
        monkeypatch.setenv("OPENSCI_TOKEN_FILE", str(path))
    _, report, out = _doctor(capsys)
    token = report["commercial"]["token"]
    assert TOKEN not in out
    assert token["configured"] and token["source"] == via and token["file"] == str(path) and token["well_formed"] is True
    if os.name == "posix":
        assert token["owner_only"] is True
        path.chmod(0o644)
        _, report, _ = _doctor(capsys)
        assert _checks(report)["token_file_permissions"]["commands"] == [f'chmod 600 "{path}"']


def test_missing_and_malformed_token_sources(clean_env, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("OPENSCI_TOKEN_FILE", str(tmp_path / "nowhere"))
    _, report, _ = _doctor(capsys)
    assert report["commercial"]["token"]["configured"] is False and report["commercial"]["token"]["readable"] is False
    assert "token" in _checks(report)
    monkeypatch.setenv("OPENSCI_TOKEN", "oat_not-a-token")
    _, report, out = _doctor(capsys)
    assert report["commercial"]["token"]["well_formed"] is False and "token" in _checks(report)
    assert "oat_not-a-token" not in out


@pytest.mark.parametrize("url", ["https://user:secret@opensci.example/api", "ftp://opensci.example", "not a url",
                                 "https://opensci.example/?key=secret"])
def test_api_url_is_echoed_only_when_well_formed(clean_env, monkeypatch, capsys, url):
    monkeypatch.setenv("OPENSCI_API_URL", url)
    _, report, out = _doctor(capsys)
    assert report["commercial"]["api_url"] == {"configured": True, "value": None, "valid": False}
    assert "secret" not in out and "api_url" in _checks(report)


def test_missing_protected_extra_gives_the_install_command(clean_env, monkeypatch, capsys):
    real = cli_doctor._installed
    monkeypatch.setattr(cli_doctor, "_installed", lambda name: None if name == "cryptography" else real(name))
    code, report, _ = _doctor(capsys)
    assert code == 0 and report["ready"]["protected_deliveries"] is False
    commands = _checks(report)["protected_extra"]["commands"]
    assert len(commands) == 1 and f"os-optics[protected]=={__version__}" in commands[0]


def test_unsupported_interpreter_fails_with_a_fresh_install_path(clean_env, monkeypatch, capsys):
    monkeypatch.setattr(cli_doctor, "MIN_PYTHON", (sys.version_info[0] + 1, 0))
    code, report, _ = _doctor(capsys)
    assert code == 1 and report["python"]["supported"] is False and report["ready"]["local_design"] is False
    assert report["ready"]["catalog"] is True and report["ready"]["protected_deliveries"] is False  # (separate facts)
    step = _checks(report)["python"]
    assert step["commands"][0] == 'uv tool install "os-optics[protected]"'
    assert any(c.startswith('uvx --from "os-optics[protected]"') for c in step["commands"])


def test_missing_required_package_fails(clean_env, monkeypatch, capsys):
    real = cli_doctor._installed
    monkeypatch.setattr(cli_doctor, "_installed", lambda name: None if name == "shapely" else real(name))
    code, report, _ = _doctor(capsys)
    assert code == 1 and report["packages"]["shapely"] == {"required": report["packages"]["shapely"]["required"],
                                                           "installed": None, "ok": False}
    assert "shapely" in _checks(report)["packages"]["message"]


@pytest.mark.parametrize(("version", "spec", "ok"), [
    ("2.4.6", ">=2.3,<3", True), ("2.3", ">=2.3,<3", True), ("3.0.0", ">=2.3,<3", False), ("2.2.9", ">=2.3,<3", False),
    ("2.11.5", ">=2.11.5,<3", True), ("2.11.4", ">=2.11.5,<3", False), ("1.2.3", "~=1.2", True), ("50.0.1", "<51,>=50.0.1", True),
    ("2.0", "===2.0", None),
])
def test_specifier_check(version, spec, ok):
    assert cli_doctor._satisfies(version, spec) is ok


def test_requirement_lines_are_parsed():
    assert cli_doctor._requirement("numpy<3,>=2.3") == ("numpy", "<3,>=2.3", None)
    assert cli_doctor._requirement('cryptography<51,>=50.0.1; extra == "protected"') == ("cryptography", "<51,>=50.0.1", "protected")
    assert cli_doctor._requirement("pydantic (>=2.11.5,<3)") == ("pydantic", ">=2.11.5,<3", None)


def test_python_dash_m_and_help(clean_env):
    env = {k: v for k, v in os.environ.items() if k not in ENV_NAMES}
    proc = subprocess.run([sys.executable, "-m", "os_optics", "doctor"], capture_output=True, text=True, timeout=300,
                          env=env, cwd=clean_env)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["ready"]["local_design"] is True
    helptext = subprocess.run([sys.executable, "-m", "os_optics", "--help"], capture_output=True, text=True, timeout=300)
    assert "doctor" in helptext.stdout
    doctor_help = subprocess.run([sys.executable, "-m", "os_optics", "doctor", "--help"], capture_output=True, text=True,
                                 timeout=300)
    assert doctor_help.returncode == 0 and "OPENSCI_API_URL" in doctor_help.stdout



# ---- the catalog lifecycle: a fresh install is ready; doctor never contacts the service ------------------------------


def test_a_fresh_install_is_ready_and_the_catalog_is_downloaded_on_first_use(clean_env, empty_catalog_store, monkeypatch, capsys):
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "3600")
    code, report, out = _doctor(capsys)
    assert code == 0 and report["ready"]["local_design"] is True and report["ready"]["catalog"] is True
    catalog = report["catalog"]
    assert catalog["status"] == "DOWNLOAD_ON_FIRST_USE" and catalog["ok"] is False and catalog["ready"] is True
    assert f"{default_service_origin()}/api/v1/catalog" in catalog["message"] and "first command that needs it" in catalog["message"]
    step = _checks(report)["catalog"]
    assert step["level"] == "WARN" and step["commands"] == ["os-optics catalog sync"]
    assert "yet" not in out  # the same neutral wording after a store was removed
    # (doctor states once, in its help, that it is local and offline; the report states the fact and the two commands
    # that test the connection)
    assert report["connection"] == {"tested": False, "commands": ["os-optics catalog check", "os-optics account status"]}
    assert step["needed_for"] == "Projects that use catalog products and the catalog commands"


def test_a_failed_download_without_a_catalog_leaves_the_catalog_not_ready(clean_env, empty_catalog_store, monkeypatch, capsys):
    """The catalog is its own readiness fact: the interpreter can run a Project (exit 0), the catalog cannot be used."""
    import socket

    from os_optics.local import LocalRunError, load_public_catalog

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "3600")
    monkeypatch.setenv("OPENSCI_CATALOG_URL", f"http://127.0.0.1:{port}")
    with pytest.raises(LocalRunError):
        load_public_catalog()  # the first use cannot reach the service
    code, report, _ = _doctor(capsys)
    assert code == 0 and report["catalog"]["status"] == "DOWNLOAD_FAILED" and report["catalog"]["ready"] is False
    assert (report["ready"]["local_design"], report["ready"]["catalog"], report["ready"]["protected_deliveries"]) == (True, False, True)
    assert "CATALOG_SERVICE_UNREACHABLE" in report["catalog"]["error"] and f"127.0.0.1:{port}" in report["catalog"]["error"]
    step = _checks(report)["catalog"]
    assert step["level"] == "FAIL" and step["commands"] == ["os-optics catalog sync", "os-optics catalog import <catalog file>"]
    assert step["needed_for"] == "Projects that use catalog products and the catalog commands"
    assert "commercial_access" not in _checks(report)


def test_a_failed_download_from_another_service_is_not_this_one_s(clean_env, empty_catalog_store, monkeypatch, capsys):
    """An offline attempt against one OPENSCI_API_URL, then the real service configured: doctor reports the
    configured service's first use, not the other service's failure under its URL."""

    import socket

    from os_optics.local import LocalRunError, load_public_catalog

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "3600")
    monkeypatch.setenv("OPENSCI_CATALOG_URL", f"http://127.0.0.1:{port}")
    with pytest.raises(LocalRunError):
        load_public_catalog()
    monkeypatch.setenv("OPENSCI_CATALOG_URL", default_service_origin())
    code, report, _ = _doctor(capsys)
    assert code == 0 and report["catalog"]["status"] == "DOWNLOAD_ON_FIRST_USE"
    assert report["catalog"]["service"]["reachable"] is None


def test_first_use_size_and_upgrade_come_only_from_the_configured_service(clean_env, empty_catalog_store, monkeypatch,
                                                                             capsys):
    """Review: service A answered with a release this engine cannot read (its size recorded,
    nothing stored); with service B configured, doctor neither quotes A's size nor asks for an engine upgrade."""

    import json as json_lib

    from catalog_service_stub import CatalogServiceStub
    from catalog_store_helpers import FIXTURE_CATALOG
    from os_optics import catalog as catalog_api
    from os_optics.catalog import CatalogError

    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "3600")
    with CatalogServiceStub() as stub:
        stub.state.publish(FIXTURE_CATALOG.read_bytes(), schema_version="4.0")
        monkeypatch.setenv("OPENSCI_CATALOG_URL", stub.url)
        with pytest.raises(CatalogError):
            catalog_api.sync()  # A: ENGINE_UPGRADE_REQUIRED, nothing stored
        code, report, _ = _doctor(capsys)
        size = report["catalog"]["service"]["current"]["size_bytes"]
        assert report["catalog"]["upgrade_required"] is True and f"{size / 1_000_000:.1f} MB" in json_lib.dumps(report)
    monkeypatch.setenv("OPENSCI_CATALOG_URL", default_service_origin())
    code, report, _ = _doctor(capsys)
    catalog = report["catalog"]
    assert code == 0 and catalog["status"] == "DOWNLOAD_ON_FIRST_USE" and catalog["upgrade_required"] is False
    assert "a few MB" in catalog["message"] and f"{size / 1_000_000:.1f} MB" not in catalog["message"]
    assert "catalog_upgrade" not in _checks(report)


def test_no_catalog_with_automatic_access_off_leaves_the_catalog_not_ready(clean_env, empty_catalog_store, capsys):
    code, report, _ = _doctor(capsys)  # the suite runs with OPENSCI_CATALOG_CHECK_INTERVAL=0
    assert code == 0 and report["catalog"]["status"] == "AUTOMATIC_ACCESS_OFF"
    assert (report["ready"]["local_design"], report["ready"]["catalog"]) == (True, False)
    assert "OPENSCI_CATALOG_CHECK_INTERVAL=0" in report["catalog"]["error"]
    assert _checks(report)["catalog"]["needed_for"] == "Projects that use catalog products and the catalog commands"


def test_a_project_without_catalog_products_runs_where_the_catalog_is_not_ready(clean_env, empty_catalog_store, capsys, tmp_path):
    """What ``ready.local_design`` true with ``ready.catalog`` false states: on an installation whose store is empty, with
    automatic catalog access off, a Project of declared optical models runs and verifies; one that uses a catalog product
    is refused with the command that obtains the catalog."""

    from composition_helpers import build, laser, place, project_of, screen

    from os_optics.local import example_project_of_kind

    code, report, _ = _doctor(capsys)
    assert code == 0 and (report["ready"]["local_design"], report["ready"]["catalog"]) == (True, False)
    lens = build("lens.ideal_thin", {"focal_length_mm": "50", "clear_aperture_diameter_mm": "20"}, model_id="l")
    declared = project_of([laser("SRC", (-100.0, 0.0, 0.0)), place(lens, "L1", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))],
                          [screen("P", (50.0, 0.0, 0.0))], builds=[lens])
    path = tmp_path / "declared.json"
    path.write_text(json.dumps(declared), encoding="utf-8")
    assert main(["project", "run", str(path), "--out-dir", str(tmp_path / "run")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["outcome"]["execution"]["outcome"] == "COMPLETED"
    assert main(["project", "verify", str(tmp_path / "run")]) == 0
    capsys.readouterr()
    assert not empty_catalog_store.exists() or not any(empty_catalog_store.rglob("*.json"))  # (no catalog was obtained)
    assert main(["project", "example"]) == 65
    (problem,) = json.loads(capsys.readouterr().out)["problems"]
    assert problem["code"] == "CATALOG_NOT_AVAILABLE"


def test_a_damaged_active_release_is_reported_with_its_repair(clean_env, empty_catalog_store, capsys):
    from catalog_store_helpers import FIXTURE_CATALOG
    from os_optics.catalog import import_release

    report = import_release(FIXTURE_CATALOG)
    (empty_catalog_store / "releases" / f"{report.active.catalog_version}.json").unlink()
    code, doctor, out = _doctor(capsys)
    assert code == 0 and doctor["catalog"]["status"] == "DAMAGED" and "missing or damaged" in doctor["catalog"]["error"]
    assert (doctor["ready"]["local_design"], doctor["ready"]["catalog"]) == (True, False)
    assert _checks(doctor)["catalog"]["commands"][0] == "os-optics catalog sync" and "yet" not in out


def test_the_help_says_what_doctor_does_not_check():
    assert "does not contact OpenSci or test the connection" in cli_doctor.DOCTOR_DESCRIPTION
    assert "DOWNLOAD_ON_FIRST_USE" in cli_doctor.DOCTOR_DESCRIPTION and "downloaded on first use" in cli_doctor.DOCTOR_EPILOG
    assert "ready.local_design" in cli_doctor.DOCTOR_EPILOG and "The catalog does not set the exit code" in cli_doctor.DOCTOR_EPILOG
    assert "ready.catalog" in cli_doctor.DOCTOR_DESCRIPTION
