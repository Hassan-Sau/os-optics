"""``os-optics doctor``: check this installation and print what is missing, with the commands that fix it.

Prints one JSON document (``opensci.engine.doctor`` version 2)::

    os-optics doctor
    python -m os_optics doctor

Reported: the engine version and build id; the Python interpreter and whether it is supported; the platform; the
installed versions of the required packages against their declared ranges; whether the ``[protected]`` extra (needed to
open Assembly Preview / Assembly Detail deliveries) is installed; the local public optical catalog store (active
release, stored releases, last check against the catalog service, automatic check interval); whether
``OPENSCI_API_URL`` and an Account token source are configured for the commercial steps (the token itself is never
printed); and ``next_steps``: concrete commands for everything that is missing.

The check is local and offline: it never contacts OpenSci, so it does not test the connection (``connection.tested`` is
false and ``connection.commands`` names the two commands that do: ``os-optics catalog check`` asks the catalog service,
``os-optics account status`` checks the token with the service). ``catalog.status`` is one of:

* ``OK``: an active catalog release is stored and intact;
* ``DOWNLOAD_ON_FIRST_USE`` (a warning, not a failure): no release is stored and automatic catalog access is on, so the
  first command that needs the catalog downloads the current release (a few MB); ``os-optics catalog sync`` does
  it now. This is the state of a fresh installation (or of a catalog store that was removed);
* ``DOWNLOAD_FAILED``: no release is stored and the last attempt to download one from the configured service failed (e.g. no access to the
  catalog service): sync when the service is reachable, or ``catalog import`` a release file;
* ``AUTOMATIC_ACCESS_OFF``: no release is stored and automatic catalog access is off
  (``OPENSCI_CATALOG_CHECK_INTERVAL=0``);
* ``NEWEST_STORED_ON_FIRST_USE`` (a warning, not a failure): releases are stored but none is selected; the first command
  that needs the catalog uses the newest stored release (``catalog use`` selects another);
* ``ACTIVE_UNKNOWN``: releases are stored, none is selected and this Engine reads none of them (``catalog sync`` /
  ``import``);
* ``EARLIER_FORMAT``: the active release is written in an earlier catalog schema this Engine no longer reads (kept from
  before an Engine upgrade; ``catalog sync`` replaces it);
* ``DAMAGED``: the active release's file is missing or differs from what was verified (``catalog sync`` repairs it);
* ``UNREADABLE``: the store cannot be read.

``ready`` states four separate facts: ``local_design`` (this interpreter can run a Project: a supported Python and the
required packages), ``catalog`` (a Project that uses catalog products, ``OS-OPT-...``, and the catalog commands can start:
``catalog.status`` ``OK`` or ``DOWNLOAD_ON_FIRST_USE``), ``protected_deliveries`` and ``commercial_configuration``. A
Project whose components are all declared optical models or native components reads no catalog and runs without one.

Exit code 0 when ``ready.local_design`` is true, 1 when Python or a required package is missing. The catalog does not
set the exit code: read ``ready.catalog`` and ``catalog.status``. Doctor cannot know whether this computer is offline; an
offline computer that never had a catalog reports ``DOWNLOAD_ON_FIRST_USE`` until a download was attempted, then
``DOWNLOAD_FAILED`` (``ready.catalog`` false until a catalog is obtained).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import stat
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from ._version import DISTRIBUTION_NAME, MIN_PYTHON, __version__

DOCTOR_SCHEMA = "opensci.engine.doctor"
DOCTOR_SCHEMA_VERSION = "2"
DISTRIBUTION = DISTRIBUTION_NAME
PROTECTED_EXTRA_NAME = "protected"
# Fallback when the distribution metadata is unavailable (a source checkout on sys.path): the required packages.
CORE_DISTRIBUTIONS = ("numpy", "scipy", "pydantic", "shapely")
API_URL_ENV = "OPENSCI_API_URL"
TOKEN_ENV = "OPENSCI_TOKEN"
TOKEN_FILE_ENV = "OPENSCI_TOKEN_FILE"
DEFAULT_TOKEN_FILE = Path(".opensci") / "token"
TOKEN_PATTERN = re.compile(r"oat_[0-9a-f]{32}_[0-9a-f]{64}")
# An http(s) base URL without credentials, query or fragment: scheme, host (name, IPv4 or [IPv6]), optional port and path.
API_URL_PATTERN = re.compile(r"(https?)://([A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::([0-9]{1,5}))?(/[^?#@\s]*)?")
# Permission bits a token file must not grant (group / other access).
TOKEN_FILE_FOREIGN_BITS = stat.S_IRWXG | stat.S_IRWXO
MAX_TOKEN_FILE_BYTES = 4096
DOCTOR_HELP = "check this installation offline (Python, packages, optional extra, local catalog, commercial configuration)"
DOCTOR_DESCRIPTION = (
    "Check this installation and print one JSON document: engine version and build id, Python version and whether it is "
    "supported, platform, installed package versions, whether the [protected] extra is installed, the local public "
    "optical catalog store (catalog.status: OK, or DOWNLOAD_ON_FIRST_USE on a fresh install -- the current release, a "
    "few MB, is downloaded by the first command that needs it or by 'os-optics catalog sync'), whether "
    "OPENSCI_API_URL and an Account token (OPENSCI_TOKEN, OPENSCI_TOKEN_FILE or ~/.opensci/token) are configured, and "
    "next_steps: each with the check, what it is needed_for and the commands that fix it. Two readiness facts stand "
    "apart: ready.local_design says whether this interpreter can run a Project (a supported Python and the required "
    "packages; a Project of declared optical models or native components needs nothing more), ready.catalog whether "
    "Projects that use catalog products (OS-OPT-...) and the catalog commands can start. The token itself is never "
    "printed. Local and offline: doctor does not contact OpenSci or test the connection (connection.tested is false; "
    "connection.commands are 'os-optics catalog check', which asks the catalog service, and 'os-optics account status', "
    "which checks the token with the service)."
)
DOCTOR_EPILOG = ("exit codes: 0 = this interpreter can run a Project (ready.local_design), 1 = Python or a required package is "
                 "missing. The catalog does not set the exit code: ready.catalog and catalog.status state it (on a fresh "
                 "install it is downloaded on first use)")
#: The ``connection`` entry of the report: doctor never contacts the service.
CONNECTION_NOTE = {
    "tested": False,
    "commands": ["os-optics catalog check", "os-optics account status"],
}
#: catalog.status values (module docstring) and whether a Project that uses catalog products, and the catalog commands,
#: can start with them (``ready.catalog``).
CATALOG_READY = {"OK": True, "DOWNLOAD_ON_FIRST_USE": True, "DOWNLOAD_FAILED": False, "AUTOMATIC_ACCESS_OFF": False,
                 "NEWEST_STORED_ON_FIRST_USE": True, "ACTIVE_UNKNOWN": False, "EARLIER_FORMAT": False, "DAMAGED": False,
                 "UNREADABLE": False}
#: What a catalog that cannot be used stops (``next_steps[].needed_for``).
CATALOG_NEEDED_FOR = "Projects that use catalog products and the catalog commands"
#: The longest exception text a doctor report repeats (the report stays small enough for an Agent to read whole).
MAX_ERROR_CHARS = 500
BYTES_PER_MB = 1_000_000


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``doctor`` subcommand to the ``os-optics`` parser."""

    doctor = subparsers.add_parser("doctor", help=DOCTOR_HELP, description=DOCTOR_DESCRIPTION, epilog=DOCTOR_EPILOG)
    doctor.set_defaults(handler=run)


def run(args: argparse.Namespace | None = None) -> int:
    """Print the report; exit code 0 when this interpreter can run a Project (``ready.local_design``)."""

    report = doctor_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["ready"]["local_design"] else 1


def _version_tuple(text: str) -> tuple[int, ...]:
    match = re.match(r"\d+(?:\.\d+)*", text.strip())
    return tuple(int(p) for p in match.group(0).split(".")) if match else ()


def _satisfies(version: str, specifier: str) -> bool | None:
    """Whether ``version`` meets a simple ``>=A,<B`` style specifier; None when a clause is not understood."""

    have = _version_tuple(version)
    for clause in (c.strip() for c in specifier.split(",") if c.strip()):
        match = re.fullmatch(r"(>=|<=|==|!=|>|<|~=)\s*([0-9][0-9.]*)", clause)
        if not match or not have:
            return None
        op, bound = match.group(1), _version_tuple(match.group(2))
        width = max(len(have), len(bound))
        a, b = have + (0,) * (width - len(have)), bound + (0,) * (width - len(bound))
        ok = {">=": a >= b, "<=": a <= b, "==": a == b, "!=": a != b, ">": a > b, "<": a < b,
              "~=": a >= b and a[: len(bound) - 1] == b[: len(bound) - 1]}[op]
        if not ok:
            return False
    return True


def _requirement(line: str) -> tuple[str, str, str | None]:
    """``(name, specifier, extra)`` of one ``Requires-Dist`` line."""

    spec, _, marker = line.partition(";")
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*\(?([^)]*)\)?", spec)
    name, specifier = (match.group(1), match.group(2).strip()) if match else (spec.strip(), "")
    extra = re.search(r"extra\s*==\s*['\"]([^'\"]+)['\"]", marker)
    return name, specifier, extra.group(1) if extra else None


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _distribution() -> dict[str, Any]:
    try:
        dist = metadata.distribution(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return {"installed": False, "requires_python": None, "requires": []}
    return {"installed": True, "version": dist.version, "requires_python": dist.metadata.get("Requires-Python"),
            "requires": list(dist.requires or [])}


def _packages(dist: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(required packages, packages of the [protected] extra) with installed version and range check."""

    core: dict[str, Any] = {}
    protected: dict[str, Any] = {}
    parsed = [_requirement(line) for line in dist["requires"]]
    if not parsed:
        parsed = [(name, "", None) for name in CORE_DISTRIBUTIONS]
    for name, specifier, extra in parsed:
        if extra not in (None, PROTECTED_EXTRA_NAME):
            continue
        installed = _installed(name)
        entry = {"required": specifier or None, "installed": installed,
                 "ok": installed is not None and (not specifier or _satisfies(installed, specifier) is not False)}
        (core if extra is None else protected)[name] = entry
    return core, protected


def _install_kind() -> str:
    prefix = Path(sys.prefix)
    if (prefix / "uv-receipt.toml").is_file():
        return "uv-tool"
    if any(part.startswith("archive-v") for part in prefix.parts):
        return "uvx"
    return "virtualenv" if sys.prefix != sys.base_prefix else "system"


def _pip_available() -> bool:
    from importlib.util import find_spec

    return find_spec("pip") is not None


def _shell_path(path: str) -> str:
    """A path as a command word: bare when it has no spaces, else quoted (PowerShell also needs its call operator)."""

    if " " not in path:
        return path
    return f'& "{path}"' if sys.platform == "win32" else f'"{path}"'


def _install_commands(kind: str, pip: bool) -> list[str]:
    requirement = f'"{DISTRIBUTION}[{PROTECTED_EXTRA_NAME}]=={__version__}"'
    if kind == "uv-tool":
        return [f"uv tool install --force {requirement}"]
    if kind == "uvx":
        return [f'uvx --from "{DISTRIBUTION}[{PROTECTED_EXTRA_NAME}]" os-optics doctor']
    if pip:
        return [f"{_shell_path(sys.executable)} -m pip install {requirement}"]
    return [f'uv pip install --python "{sys.executable}" {requirement}']


def _fresh_install_commands() -> list[str]:
    minimum = "{}.{}".format(*MIN_PYTHON)
    venv = f"py -{minimum} -m venv .venv" if sys.platform == "win32" else f"python{minimum} -m venv .venv"
    python = r".venv\Scripts\python" if sys.platform == "win32" else ".venv/bin/python"
    return [
        f'uv tool install "{DISTRIBUTION}[{PROTECTED_EXTRA_NAME}]"',
        f'uvx --from "{DISTRIBUTION}[{PROTECTED_EXTRA_NAME}]" os-optics doctor',
        f'{venv} && {python} -m pip install "{DISTRIBUTION}[{PROTECTED_EXTRA_NAME}]"',
    ]


def _api_url() -> dict[str, Any]:
    raw = os.environ.get(API_URL_ENV, "").strip()
    if not raw:
        return {"configured": False, "value": None, "valid": None}
    match = API_URL_PATTERN.fullmatch(raw)
    # Only a well-formed URL is echoed back (a malformed value might embed a credential).
    return {"configured": True, "value": raw if match else None, "valid": match is not None}


def _token_file_report(path: Path, source: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"source": source, "file": str(path)}
    try:
        info = path.stat()
        if info.st_size > MAX_TOKEN_FILE_BYTES:
            return {**entry, "configured": False, "readable": True, "well_formed": False}
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {**entry, "configured": False, "readable": False, "well_formed": None}
    entry.update({"configured": True, "readable": True, "well_formed": TOKEN_PATTERN.fullmatch(text) is not None})
    if os.name == "posix":
        entry["owner_only"] = not (info.st_mode & TOKEN_FILE_FOREIGN_BITS)
    return entry


def _token() -> dict[str, Any]:
    """Which credential source the client would use, in its resolution order; the credential itself is never returned."""

    value = os.environ.get(TOKEN_ENV, "").strip()
    if value:
        return {"configured": True, "source": TOKEN_ENV, "file": None, "well_formed": TOKEN_PATTERN.fullmatch(value) is not None}
    named = os.environ.get(TOKEN_FILE_ENV, "").strip()
    if named:
        return _token_file_report(Path(named).expanduser(), TOKEN_FILE_ENV)
    try:
        default = Path.home() / DEFAULT_TOKEN_FILE
    except RuntimeError:
        return {"configured": False, "source": None, "file": None, "well_formed": None}
    if default.is_file():
        return _token_file_report(default, "default_token_file")
    return {"configured": False, "source": None, "file": str(default), "well_formed": None}


def _download_size(service: dict[str, Any]) -> str:
    """The configured service's release size: its confirmed current release, else its own earlier answer; never another
    service's."""

    last_known = service.get("last_known") or {}
    if not _same_service(last_known.get("manifest_url"), service.get("manifest_url")):
        last_known = {}
    current = service.get("current") or last_known.get("current") or {}
    size = current.get("size_bytes")
    return f"about {size / BYTES_PER_MB:.1f} MB" if isinstance(size, int) and size > 0 else "a few MB"


def _same_service(url: str | None, configured: str | None) -> bool:
    from .catalog import remote

    return url is not None and configured is not None and remote.origin_of(url) == remote.origin_of(configured)


def _empty_store(entry: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    """``catalog`` of a store holding no release (module docstring): whether it is downloaded on first use."""

    from .catalog import automatic_access
    from .catalog.service import failure_concerns

    service = document["service"]
    url = service.get("manifest_url") or "the OpenSci catalog service"
    failure = service.get("last_check_error")
    if not failure_concerns(failure, service.get("manifest_url")):
        failure = None  # an attempt to another service (OPENSCI_API_URL changed since) says nothing about this one
    if not automatic_access():
        entry.update(status="AUTOMATIC_ACCESS_OFF",
                     error="no public optical catalog release is stored on this device and automatic catalog access is "
                           "off (OPENSCI_CATALOG_CHECK_INTERVAL=0), so none will be downloaded by itself")
    elif failure:
        entry.update(status="DOWNLOAD_FAILED",
                     error=f"no public optical catalog release is stored on this device, and the last attempt to download "
                           f"one from {url} (at {service.get('checked_at')}) failed: {failure.get('code')}: "
                           f"{failure.get('message')}")
    else:
        entry.update(status="DOWNLOAD_ON_FIRST_USE",
                     message=f"no public optical catalog release is stored on this device: the current release "
                             f"({_download_size(service)}) is downloaded from {url} by the first command that needs it, "
                             "or now with 'os-optics catalog sync'")
    return entry


def _catalog() -> dict[str, Any]:
    """The local catalog store (``os_optics.catalog.status``; no network): ``status`` (module docstring), whether
    the active release is intact (``ok``) and whether local design can start with it (``ready``)."""

    try:
        from pydantic import ValidationError

        from ._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2
        from .catalog import CatalogError, CatalogStore, status
        from .catalog.store import is_retired_schema, validated_here

        report = status()
        document = report.as_dict()
        entry: dict[str, Any] = {"ok": False, "status": None, "store": document["store"], "catalog_version": None,
                                 "products": None, "active": document["active"], "releases": len(document["releases"]),
                                 "service": document["service"], "upgrade_required": document["upgrade_required"],
                                 "automatic_check_interval_s": document["automatic_check_interval_s"]}
        active = report.active
        if active is None:
            entry["stored_releases"] = [r["catalog_version"] for r in document["releases"]]
            newest = report.state.newest_readable()
            if newest is not None:
                entry.update(status="NEWEST_STORED_ON_FIRST_USE", ok=True, catalog_version=newest,
                             message="catalog releases are stored on this device and none is selected as the active one "
                                     f"(stored: {', '.join(entry['stored_releases'])}): the first command that needs the "
                                     f"catalog uses {newest}, the newest stored release")
            elif document["releases"]:
                entry.update(status="ACTIVE_UNKNOWN",
                             error="catalog releases are stored on this device but none is selected as the active one "
                                   f"(stored: {', '.join(entry['stored_releases'])}), and this os-optics reads none of them")
            else:
                _empty_store(entry, document)
            entry["ready"] = CATALOG_READY[entry["status"]]
            return entry
        if is_retired_schema(active.schema_version):  # before parsing: this Engine cannot read it at all
            entry.update(status="EARLIER_FORMAT", ready=False, schema_version=active.schema_version,
                         error=f"the active release {active.catalog_version} is written in catalog schema "
                               f"{active.schema_version}, an earlier format this os-optics no longer reads (it reads "
                               f"schema {', '.join(document['supported_schema_versions'])})")
            return entry
        try:
            data = CatalogStore(report.store).read_release(active)
        except CatalogError as exc:
            entry.update(status="DAMAGED", ready=False,
                         error=f"the active release {active.catalog_version} in {report.store} is missing or damaged: {exc.message}")
            return entry
        if validated_here(active):
            # (the bytes are the release this Engine's catalog contract validated when it was stored: counted, not re-validated)
            header = json.loads(data)
            entry.update(ok=True, status="OK", ready=True, catalog_version=header["catalog_version"], products=len(header["products"]))
            return entry
        catalog = PublicOpticalCatalogV2.model_validate_json(data)
    except ValidationError as exc:  # never the full validation text: a whole catalog can fail with thousands of errors
        return {"ok": False, "status": "UNREADABLE", "ready": False,
                "error": f"the active catalog release is not a valid public optical catalog ({exc.error_count()} errors)"}
    except Exception as exc:  # noqa: BLE001 -- any failure to read the stored catalog is the finding itself
        return {"ok": False, "status": "UNREADABLE", "ready": False,
                "error": f"{type(exc).__name__}: {str(exc)[:MAX_ERROR_CHARS]}"}
    entry.update(ok=True, status="OK", ready=True, catalog_version=catalog.catalog_version, products=len(catalog.products))
    return entry


def _build_id() -> str | None:
    try:
        from .manifest.build import engine_build_id

        return engine_build_id()
    except Exception:  # noqa: BLE001 -- reported as unknown
        return None


def _set_env_commands(name: str, value: str) -> list[str]:
    if sys.platform == "win32":
        return [f'$env:{name} = "{value}"  # PowerShell', f'export {name}="{value}"  # bash']
    return [f'export {name}="{value}"']


def doctor_report() -> dict[str, Any]:
    """The ``opensci.engine.doctor`` document (see the module docstring)."""

    dist = _distribution()
    minimum = "{}.{}".format(*MIN_PYTHON)
    python_ok = sys.version_info >= MIN_PYTHON
    core, protected = _packages(dist)
    packages_ok = all(entry["ok"] for entry in core.values())
    protected_ok = bool(protected) and all(entry["ok"] for entry in protected.values())
    if not protected:
        installed = _installed("cryptography")
        protected = {"cryptography": {"required": None, "installed": installed, "ok": installed is not None}}
        protected_ok = installed is not None
    catalog = _catalog()
    kind, pip = _install_kind(), _pip_available()
    api_url, token = _api_url(), _token()
    # (two separate facts: this interpreter can run a Project; the catalog-dependent work can start)
    engine_ready = python_ok and packages_ok
    catalog_ready = bool(catalog.get("ready"))

    steps: list[dict[str, Any]] = []
    if not python_ok:
        steps.append({"check": "python", "needed_for": "everything",
                      "message": f"os-optics needs Python {minimum} or newer; this is Python {platform.python_version()}. "
                                 "Install it with a supported Python (uv selects or downloads one).",
                      "commands": _fresh_install_commands()})
    if not packages_ok:
        steps.append({"check": "packages", "needed_for": "everything",
                      "message": "a required package is missing or outside its supported range: "
                                 + ", ".join(sorted(n for n, e in core.items() if not e["ok"])),
                      "commands": _install_commands(kind, pip)})
    catalog_status = catalog.get("status")
    if catalog_status == "DOWNLOAD_ON_FIRST_USE":
        steps.append({"check": "catalog", "level": "WARN", "needed_for": CATALOG_NEEDED_FOR,
                      "message": f"{catalog['message']}. Nothing to fix: this is the normal state of a fresh installation",
                      "commands": ["os-optics catalog sync"]})
    elif catalog_status == "NEWEST_STORED_ON_FIRST_USE":
        steps.append({"check": "catalog", "level": "WARN", "needed_for": CATALOG_NEEDED_FOR,
                      "message": f"{catalog['message']}. Nothing to fix; to use another stored release, select it",
                      "commands": ["os-optics catalog use <catalog_version>"]})
    elif catalog_status == "ACTIVE_UNKNOWN":
        steps.append({"check": "catalog", "level": "FAIL", "needed_for": CATALOG_NEEDED_FOR,
                      "message": f"{catalog['error']}: download the service's current release, which becomes the active one",
                      "commands": ["os-optics catalog sync"]})
    elif catalog_status == "EARLIER_FORMAT":
        steps.append({"check": "catalog", "level": "FAIL", "needed_for": CATALOG_NEEDED_FOR,
                      "message": f"{catalog['error']}: download the service's current release, which replaces it as the "
                                 "active one (on a computer without access to the catalog service, import a current release file)",
                      "commands": ["os-optics catalog sync", "os-optics catalog import <catalog file>"]})
    elif catalog_status == "DAMAGED":
        steps.append({"check": "catalog", "level": "FAIL", "needed_for": CATALOG_NEEDED_FOR,
                      "message": f"{catalog['error']}: download the current release again (it replaces a damaged file), or "
                                 "on a computer without access to the catalog service import a release file",
                      "commands": ["os-optics catalog sync", "os-optics catalog import <catalog file>"]})
    elif not catalog["ok"]:
        steps.append({"check": "catalog", "level": "FAIL", "needed_for": CATALOG_NEEDED_FOR,
                      "message": f"no usable public optical catalog on this device ({catalog.get('error')}): download it "
                                 "from the OpenSci catalog service when it is reachable, or on a computer without that "
                                 "access import a catalog release file",
                      "commands": ["os-optics catalog sync", "os-optics catalog import <catalog file>"]})
    elif catalog.get("upgrade_required"):
        steps.append({"check": "catalog_upgrade", "needed_for": "the newest public optical catalog",
                      "message": "the catalog service publishes a catalog release this os-optics cannot read; upgrade "
                                 "os-optics (the stored release stays in use meanwhile)",
                      "commands": [f'{_shell_path(sys.executable)} -m pip install --upgrade "{DISTRIBUTION}[{PROTECTED_EXTRA_NAME}]"',
                                   f"uv tool upgrade {DISTRIBUTION}"]})
    if not protected_ok:
        steps.append({"check": "protected_extra", "needed_for": "opening Assembly Preview and Assembly Detail deliveries",
                      "message": "install the [protected] extra",
                      "commands": _install_commands(kind, pip)})
    if not api_url["configured"] or not api_url["valid"]:
        steps.append({"check": "api_url", "needed_for": "commercial steps (resolution, Order, deliveries, artifacts)",
                      "message": f"set {API_URL_ENV} to the OpenSci service URL your OpenSci Account shows with your "
                                 "Agent token; the local design workflow does not need it",
                      "commands": _set_env_commands(API_URL_ENV, "<OpenSci service URL from your OpenSci Account>")})
    if not token["configured"] or token.get("well_formed") is False:
        default_file = f"~/{DEFAULT_TOKEN_FILE.as_posix()}"
        steps.append({"check": "token", "needed_for": "commercial steps (resolution, Order, deliveries, artifacts)",
                      "message": f"make the Account token for this Agent available: save it to {default_file} "
                                 f"(readable only by you), or point {TOKEN_FILE_ENV} at the file holding it, or set "
                                 f"{TOKEN_ENV}; the value must have the form oat_<32 hex>_<64 hex>",
                      "commands": _set_env_commands(TOKEN_FILE_ENV, "<path of the token file>")})
    elif token.get("owner_only") is False:
        steps.append({"check": "token_file_permissions", "needed_for": "keeping the Account token private",
                      "message": "the token file is readable by other users",
                      "commands": [f'chmod 600 "{token["file"]}"']})
    if engine_ready and catalog_ready and api_url.get("valid") and token["configured"] and token.get("well_formed"):
        steps.append({"check": "commercial_access", "needed_for": "commercial steps",
                      "message": "local configuration is complete; confirm the token with the service",
                      "commands": ["os-optics account status"]})

    return {
        "schema": DOCTOR_SCHEMA,
        "schema_version": DOCTOR_SCHEMA_VERSION,
        "engine": {"version": __version__, "build_id": _build_id(), "distribution_installed": dist["installed"],
                   "module_path": str(Path(__file__).resolve().parent)},
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "executable": sys.executable, "requires": dist["requires_python"] or f">={minimum}",
                   "supported": python_ok},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                     "sys_platform": sys.platform},
        "install": {"kind": kind, "prefix": sys.prefix, "pip_available": pip},
        "packages": core,
        "protected_extra": {"installed": protected_ok, "packages": protected},
        "catalog": catalog,
        "commercial": {"api_url": api_url, "token": token},
        "connection": CONNECTION_NOTE,
        "ready": {"local_design": engine_ready, "catalog": catalog_ready,
                  "protected_deliveries": engine_ready and protected_ok,
                  "commercial_configuration": bool(api_url.get("valid")) and bool(token["configured"])
                  and token.get("well_formed") is not False},
        "next_steps": steps,
    }
