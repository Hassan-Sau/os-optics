"""Repository hygiene tests: forbidden dependencies, no I/O / clocks / global state in the core, no placeholders, no scattered
tolerance literals."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "os_optics"
FILES = sorted(SRC.rglob("*.py"))
# ``_vendor/`` is the vendored contract bundle (the design, catalog, view, assembly and delivery contracts the Engine shares
# with the OpenSci service; ``MANIFEST.json`` pins every file, test_vendored_contracts.py checks it). It is governed by
# those contracts' own review and tests, so the Engine-authoring rules below (placeholder words, named tolerance
# literals, the Engine's stdlib allow-list) apply to CORE_FILES only; forbidden imports, I/O and global state are checked
# everywhere, and the vendored imports have their own allow-list.
VENDOR = SRC / "_vendor"
CORE_FILES = [p for p in FILES if VENDOR not in p.parents]
VENDORED_FILES = [p for p in FILES if VENDOR in p.parents]

FORBIDDEN_IMPORTS = {
    # databases / web / cloud / heavy compute / concurrency
    "sqlalchemy", "psycopg", "psycopg2", "requests", "httpx", "fastapi", "flask", "django", "redis", "celery", "kafka", "torch",
    "tensorflow", "multiprocessing", "cupy", "numba", "boto3", "oss2", "minio",
    # network / OS-level state, threads, subprocesses (offline, deterministic core)
    "socket", "ssl", "urllib", "http", "asyncio", "threading", "concurrent", "subprocess", "sched", "signal",
    # wall-clock and non-reproducible randomness
    "time", "datetime", "random", "uuid", "secrets",
    # dependency policy (docs/PYODIDE_COMPATIBILITY.md): not adopted
    "trimesh", "networkx", "rtree", "matplotlib",
}
# cli.py is the only module that touches stdio / process exit codes. Besides it, only the local workflow touches files, and only
# files a caller names explicitly: local/catalog.py reads catalog / Project files a caller names; local/artifacts.py writes and
# re-reads a run directory (pathlib only). The physics core and the optical view do no I/O.
IO_ALLOWED = {"cli.py", "cli_view.py", "__main__.py"}
# `os-optics doctor` (cli_doctor.py) inspects the installation: interpreter, platform, installed distributions, the
# commercial-access environment variables and the token file's presence / permissions (never printing the credential).
DOCTOR_MODULE = "cli_doctor.py"
DOCTOR_STDLIB = {"os", "platform", "stat"}
IO_ALLOWED = IO_ALLOWED | {DOCTOR_MODULE}
# manifest/build.py computes the build identity once per process (``functools.lru_cache`` on ``engine_build_id`` alone).
BUILD_ID_MODULE = "manifest/build.py"
# cli.py reads PYTHONIOENCODING to decide whether its JSON output to a file or pipe is UTF-8 (``os.environ`` only).
CLI_MODULE = "cli.py"
CLI_STDLIB = {"os"}
# deployment.py parses the packaged deployment defaults (deployment.toml, read through importlib.resources).
DEPLOYMENT_MODULE = "deployment.py"
DEPLOYMENT_STDLIB = {"tomllib"}
# Opening a protected Assembly Preview / Assembly Detail delivery checks a short-lived decryption grant's
# validity window against the current time, so exactly these modules may import ``datetime``: the vendored canonical
# delivery contract (which only compares a caller-supplied ``now``) and ``protected_delivery.py`` (whose default ``now``
# is the current UTC time). The physics core, layout, assembly and drawing code stay clock-free.
PROTECTED_DELIVERY_VENDORED = "_vendor/shared_contracts/protected_assembly_delivery/"
CLOCK_ALLOWED = {"protected_delivery.py", *(PROTECTED_DELIVERY_VENDORED + name for name in ("_common.py", "envelope.py", "detail_envelope.py"))}
# ``cryptography`` (the optional ``protected`` extra, AES-256-GCM) may be imported only lazily -- inside a function, never
# at module level -- and only by these modules, so ``import os_optics`` and every other module work without it.
CRYPTOGRAPHY_ALLOWED = {"protected_delivery.py", PROTECTED_DELIVERY_VENDORED + "_common.py"}
# The commercial client (``os_optics.client``) is the package's one network boundary: it talks HTTP to the OpenSci
# service with the standard library (``urllib``), reads the ``OPENSCI_*`` environment variables, writes the user's token
# file owner-only (``os``; ``icacls`` via ``subprocess`` on Windows), paces retries (``time``), draws multipart
# boundaries / token-issuance keys (``secrets``) and logs requests without secrets (``logging``). ``cli_commerce.py`` is
# its command-line front end (stdio, like ``cli.py``). Nothing else may import the client (test below).
CLIENT_PACKAGE = "client/"
CLIENT_IMPORTS = {"os", "secrets", "subprocess", "time", "urllib", "ipaddress", "logging"}
IO_ALLOWED |= {"cli_commerce.py"}
# The local 3D viewer (os_optics.viewer) serves one page to this computer's browser: exactly its server module may use
# the standard library's HTTP server on 127.0.0.1, a serving thread, an unguessable path token and the browser launcher.
# Nothing else in the package gains network, thread or randomness access, and the viewer's scene / rasteriser modules stay
# pure. ``cli_view.py`` is the viewer's command-line module (stdio, like cli.py).
VIEWER_SERVER = "viewer/server.py"
VIEWER_SERVER_ALLOWED = {"http", "threading", "secrets", "webbrowser"}
# The local public optical catalog store (``os_optics.catalog``) keeps verified catalog releases on this device and
# synchronizes them with the catalog service: exactly this package may read its store directory and environment (``os``:
# OPENSCI_CATALOG_DIR / OPENSCI_CATALOG_URL / OPENSCI_CATALOG_CHECK_INTERVAL, atomic renames), stamp checks with the wall
# clock (``time``: the heartbeat interval) and -- in ``catalog/remote.py`` only -- make unauthenticated HTTP GETs
# (``urllib``, ``ipaddress`` for the loopback rule, ``logging``). ``cli_catalog.py`` is its command-line front end. Only
# ``local/catalog.py`` (the catalog resolution of a local run) and the command-line modules import it, and it never imports
# the local workflow or the commercial client (tests below). ``catalog/service.py`` alone writes to stderr: one notice line
# when a command downloaded a catalog release by itself (standard output is never written).
CATALOG_PACKAGE = "catalog/"
CATALOG_IMPORTS = {"os", "time"}
CATALOG_REMOTE = "catalog/remote.py"
CATALOG_REMOTE_IMPORTS = {"urllib", "ipaddress", "logging"}
# ``catalog/locking.py`` only: the store's inter-process lock (``msvcrt.locking`` / ``fcntl.flock`` on a lock file) and
# its in-process re-entrant lock (``threading``), so concurrent processes and threads never lose store state.
CATALOG_LOCKING = "catalog/locking.py"
CATALOG_LOCKING_IMPORTS = {"threading", "msvcrt", "fcntl"}
IO_ALLOWED |= {"cli_catalog.py"}
# ``os-optics results`` / ``export`` (cli_results.py) print a run directory's results and write an export file a caller
# names. ``results/export.py`` builds the dense-grid ``.npz`` export in memory (``io.BytesIO`` + ``zipfile`` with fixed
# timestamps); it never touches the filesystem (the I/O checks still apply to it).
IO_ALLOWED |= {"cli_results.py"}
# ``os-optics layout draw`` (cli_layout.py) prints the summary of an Optical Layout Drawing it wrote to --out.
IO_ALLOWED |= {"cli_layout.py"}
# ``os-optics project solids`` (cli_solids.py) reads the Project file it is given and prints the optical-solid report.
IO_ALLOWED |= {"cli_solids.py"}
# ``os-optics project import-prescription`` (cli_prescription.py) reads the prescription / Project / catalog files a
# caller names, prints the import and writes the updated Project to --out.
IO_ALLOWED |= {"cli_prescription.py"}
# ``os-optics native build --into PROJECT.json`` (cli_native.py) reads and rewrites the Project file a caller names.
IO_ALLOWED |= {"cli_native.py"}
# ``os-optics project network`` (cli_network.py) reads the Project file it is given and prints / writes the network JSON.
IO_ALLOWED |= {"cli_network.py"}
# ``os-optics interop list|show`` (cli_interop.py) prints the interop capability catalog.
IO_ALLOWED |= {"cli_interop.py"}
# ``os-optics project paraxial`` (cli_paraxial.py) reads the Project (and catalog) file it is given and prints the analysis.
IO_ALLOWED |= {"cli_paraxial.py"}
# ``os-optics workspace ...`` (cli_workspace.py) reads the workspace / module / request files a caller names, prints
# the operation's JSON reply and writes the resulting workspace or module.
IO_ALLOWED |= {"cli_workspace.py"}
# ``os-optics capabilities`` / ``project check`` / ``project update`` (cli_project.py) read the Project / operation
# files a caller names, print JSON and write the updated Project to the -o file a caller names.
IO_ALLOWED |= {"cli_project.py"}
# ``_command.py``: a Ctrl-C while the os-optics command is still importing ends it with one line on stderr (exit code 130).
IO_ALLOWED |= {"_command.py"}
RESULTS_EXPORT = "results/export.py"
RESULTS_EXPORT_STDLIB = {"io", "zipfile"}
# A run's progress, cancellation and resource limits: ``run_control.py`` is the Engine's side -- no clock, no
# I/O; a context variable names the run's monitor, whose checkpoints may only stop the run -- and ``run_monitor.py`` the
# host's: exactly it reads the monotonic clock (``time``: progress intervals, elapsed seconds) and this process's memory
# (``ctypes`` on Windows, ``resource`` and ``/proc/self/statm`` elsewhere) to report progress and enforce the limits a
# caller sets. Neither enters a number of a run.
RUN_CONTROL = "run_control.py"
RUN_CONTROL_STDLIB = {"contextlib", "contextvars"}
RUN_MONITOR = "run_monitor.py"
RUN_MONITOR_IMPORTS = {"time", "ctypes", "resource"}
# ``cli_exit.py`` formats the traceback of an internal error (an exception no command documents; exit code 70), which the
# command-line modules write to standard error. Nothing else formats a traceback.
CLI_EXIT_MODULE = "cli_exit.py"
CLI_EXIT_STDLIB = {"traceback"}
# ``local/prepared.py`` keeps the prepared evaluation handles of the JSON API: a lock (``threading``) orders the calls a
# threaded host makes on them (no thread is started; the browser worker runs one call at a time), and a handle id is random
# (``secrets``) so a handle issued before the runtime started again is never taken for a new one. Neither enters a number.
PREPARED_MODULE = "local/prepared.py"
PREPARED_IMPORTS = {"threading", "secrets"}


def _catalog_allowance(rel: str) -> set[str]:
    if not rel.startswith(CATALOG_PACKAGE):
        return set()
    return (CATALOG_IMPORTS | (CATALOG_REMOTE_IMPORTS if rel == CATALOG_REMOTE else set())
            | (CATALOG_LOCKING_IMPORTS if rel == CATALOG_LOCKING else set()))


def _imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_no_forbidden_imports(path):
    rel = path.relative_to(SRC).as_posix()
    forbidden = FORBIDDEN_IMPORTS - {"datetime"} if rel in CLOCK_ALLOWED else FORBIDDEN_IMPORTS
    forbidden = forbidden - CLIENT_IMPORTS if rel.startswith(CLIENT_PACKAGE) else forbidden
    forbidden = forbidden - VIEWER_SERVER_ALLOWED if rel == VIEWER_SERVER else forbidden
    forbidden = forbidden - RUN_MONITOR_IMPORTS if rel == RUN_MONITOR else forbidden
    forbidden = forbidden - PREPARED_IMPORTS if rel == PREPARED_MODULE else forbidden
    forbidden = forbidden - _catalog_allowance(rel)
    bad = _imports(ast.parse(path.read_text(encoding="utf-8"))) & forbidden
    assert not bad, f"{path.name} imports forbidden modules: {sorted(bad)}"


def test_only_declared_third_party_dependencies_are_imported():
    stdlib = {"__future__", "ast", "argparse", "collections", "heapq", "dataclasses", "enum", "functools", "hashlib", "importlib", "json", "math",
              "pathlib", "sys", "typing", "builtins", "copy", "itertools", "operator", "decimal",
              "re", "struct",  # re: id patterns; struct: the optical view's GLB byte layout (both pure, no I/O)
              "base64", "zlib"}  # the viewer's in-memory page document (base64 arrays) and PNG encoder (zlib)
    allowed = {"numpy", "scipy", "pydantic", "shapely", "os_optics"} | stdlib
    for path in CORE_FILES:
        rel = path.relative_to(SRC).as_posix()
        permitted = allowed | ({"datetime"} if rel in CLOCK_ALLOWED else set()) | ({"cryptography"} if rel in CRYPTOGRAPHY_ALLOWED else set())
        permitted |= CLIENT_IMPORTS if rel.startswith(CLIENT_PACKAGE) else set()
        permitted |= DOCTOR_STDLIB if rel == DOCTOR_MODULE else set()
        permitted |= CLI_STDLIB if rel == CLI_MODULE else set()
        permitted |= DEPLOYMENT_STDLIB if rel == DEPLOYMENT_MODULE else set()
        permitted |= RESULTS_EXPORT_STDLIB if rel == RESULTS_EXPORT else set()
        permitted |= VIEWER_SERVER_ALLOWED if rel == VIEWER_SERVER else set()
        permitted |= RUN_CONTROL_STDLIB if rel == RUN_CONTROL else set()
        permitted |= RUN_MONITOR_IMPORTS if rel == RUN_MONITOR else set()
        permitted |= CLI_EXIT_STDLIB if rel == CLI_EXIT_MODULE else set()
        permitted |= PREPARED_IMPORTS if rel == PREPARED_MODULE else set()
        permitted |= _catalog_allowance(rel)
        extra = _imports(ast.parse(path.read_text(encoding="utf-8"))) - permitted
        assert not extra, f"{path.relative_to(SRC)} imports undeclared modules {sorted(extra)}"


def test_vendored_contracts_import_only_pydantic_and_the_standard_library():
    """The vendored contracts need nothing beyond the declared pydantic dependency (pydantic_core ships with it).

    ``assembly_preview/package.py`` (the Assembly Preview Package contract) additionally uses the pure standard library
    ``io``, ``json``, ``struct``, ``zipfile``, ``dataclasses`` and ``types``: it builds, reads and verifies package bytes and
    the single-file ZIP archive entirely in memory (``io.BytesIO``), parses GLB byte layouts, and never touches the
    filesystem (the I/O checks below still apply to it).

    ``protected_assembly_delivery/`` additionally uses ``base64`` (the strict base64url wire encoding),
    ``datetime`` (comparing the grant window with a caller-supplied ``now``) and, lazily, ``cryptography`` (AES-256-GCM;
    the ``protected`` extra): see ``CLOCK_ALLOWED`` / ``CRYPTOGRAPHY_ALLOWED``.
    """
    assert VENDORED_FILES, "the public design bundle is missing from the package"
    allowed = {"__future__", "collections", "decimal", "enum", "hashlib", "math", "re", "typing", "pydantic", "pydantic_core",
               "dataclasses", "io", "json", "struct", "types", "zipfile",  # assembly_preview/package.py: in-memory bytes/ZIP only
               "unicodedata"}  # common/text.py: NFC form and character categories of author text (pure, no I/O)
    for path in VENDORED_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(SRC).as_posix()
        permitted = allowed | ({"base64", "datetime"} if rel.startswith(PROTECTED_DELIVERY_VENDORED) else set())
        permitted |= {"cryptography"} if rel in CRYPTOGRAPHY_ALLOWED else set()
        extra = _imports(tree) - permitted
        assert not extra, f"{path.relative_to(SRC)} imports undeclared modules {sorted(extra)}"
        absolute = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.startswith("opensci")]
        assert not absolute, f"{path.relative_to(SRC)} must use relative imports only: {absolute}"


def test_cryptography_is_imported_lazily_and_only_by_the_protected_delivery_modules():
    """No module imports ``cryptography`` at module level; only the protected-delivery modules import it at all."""

    for path in FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(SRC).as_posix()
        top_level = _imports(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))], type_ignores=[]))
        assert "cryptography" not in top_level, f"{rel} imports cryptography at module level"
        if rel not in CRYPTOGRAPHY_ALLOWED:
            assert "cryptography" not in _imports(tree), f"{rel} imports cryptography"
    assert all((SRC / rel).is_file() for rel in CRYPTOGRAPHY_ALLOWED | CLOCK_ALLOWED)


def test_only_the_command_line_imports_the_commercial_client():
    """The Engine core, the local workflow, assembly and drawing code never import ``os_optics.client`` (the network
    boundary); only the client package itself and the command-line modules do."""

    offenders = []
    for path in FILES:
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith(CLIENT_PACKAGE) or rel.startswith("cli"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            module = node.module or "" if isinstance(node, ast.ImportFrom) else ""
            # (the client package, not a sibling module whose name starts with "client", e.g. client_artifacts)
            if isinstance(node, ast.ImportFrom) and (module == "os_optics.client" or module.startswith("os_optics.client.")
                                                     or (node.level and (module == "client" or module.startswith("client.")))):
                offenders.append(rel)
    assert offenders == []


def _engine_imports(path: Path) -> set[str]:
    """``os_optics`` subpackages / modules ``path`` imports (absolute or relative), as ``a/b`` paths."""

    rel_parts = path.relative_to(SRC).parent.parts
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(rel_parts[: len(rel_parts) - (node.level - 1)])
                target = base + (node.module.split(".") if node.module else [])
            elif (node.module or "").startswith("os_optics"):
                target = node.module.split(".")[1:]
            else:
                continue
            found.add("/".join(target))
    return found


def test_only_the_local_catalog_resolution_and_the_command_line_import_the_catalog_store():
    """The catalog store / synchronization package is reached from exactly one place of the local workflow
    (``local/catalog.py``) and from the command line; it imports neither the local workflow nor the commercial client."""

    importers = sorted(p.relative_to(SRC).as_posix() for p in CORE_FILES
                       if not p.relative_to(SRC).as_posix().startswith(CATALOG_PACKAGE)
                       and any(m == "catalog" or m.startswith("catalog/") for m in _engine_imports(p)))
    assert [rel for rel in importers if not rel.startswith("cli")] == ["local/catalog.py"]
    for path in (SRC / "catalog").rglob("*.py"):
        assert not {m for m in _engine_imports(path) if m.startswith(("local", "client", "cli"))}, path.name
    network = sorted(p.relative_to(SRC).as_posix() for p in (SRC / "catalog").rglob("*.py") if "urllib" in _imports(ast.parse(p.read_text(encoding="utf-8"))))
    assert network == [CATALOG_REMOTE]
    writers = sorted(p.relative_to(SRC).as_posix() for p in (SRC / "catalog").rglob("*.py") if "sys.stdout" in p.read_text(encoding="utf-8")
                     or "sys.stderr" in p.read_text(encoding="utf-8"))
    assert writers == ["catalog/service.py"] and "sys.stdout" not in (SRC / "catalog" / "service.py").read_text(encoding="utf-8")


def test_no_catalog_is_bundled_with_the_engine():
    """The public optical catalog is versioned data synchronized into a local store, never a package resource."""

    assert not [p for p in SRC.rglob("*.json") if "catalog" in p.name and p.parent.name == "_vendor"]
    assert not (SRC / "_vendor" / "public_optical_catalog.v2.json").exists()


def test_shapely_is_imported_only_by_the_layout_package():
    offenders = [p.relative_to(SRC).as_posix() for p in FILES if "shapely" in _imports(ast.parse(p.read_text(encoding="utf-8")))
                 and not p.relative_to(SRC).as_posix().startswith("layout/")]
    assert offenders == []


def test_pydantic_is_not_used_in_numerical_inner_loops():
    """math/, polarization/, surfaces/ and gaussian/ ops are pure NumPy (Pydantic only at the contract boundary)."""
    for sub in ("math", "optics/polarization", "optics/surfaces"):
        for path in (SRC / sub).rglob("*.py"):
            assert "pydantic" not in _imports(ast.parse(path.read_text(encoding="utf-8"))), path.name


@pytest.mark.parametrize("path", [p for p in FILES if p.name not in IO_ALLOWED], ids=lambda p: p.relative_to(SRC).as_posix())
def test_no_io_environment_or_global_state(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "print", "input", "exec", "eval", "globals"}:
            pytest.fail(f"{path.name}:{node.lineno} calls {node.func.id}()")
        if isinstance(node, ast.Global):
            pytest.fail(f"{path.name}:{node.lineno} uses the 'global' statement")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os"                 and not path.relative_to(SRC).as_posix().startswith((CLIENT_PACKAGE, CATALOG_PACKAGE)):
            pytest.fail(f"{path.name}:{node.lineno} touches os.{node.attr}")
        if isinstance(node, ast.Attribute) and node.attr in {"lru_cache", "cache", "cached_property"}:
            if path.relative_to(SRC).as_posix() == BUILD_ID_MODULE and node.attr == "lru_cache":
                continue  # (the build identity, the package's one per-process cache: the test below holds it to that)
            pytest.fail(f"{path.name}:{node.lineno} uses a cache decorator (no global caches allowed)")
        if isinstance(node, ast.ImportFrom) and node.module == "functools" and any(a.name in {"lru_cache", "cache", "cached_property"} for a in node.names):
            pytest.fail(f"{path.name}:{node.lineno} imports a cache decorator")


def test_the_build_identity_is_the_only_per_process_cache():
    """``manifest/build.py`` caches exactly ``engine_build_id()``, a function of no argument: the identity of the sources
    the running process imported. It is no input of any result; nothing else in the package is cached."""
    tree = ast.parse((SRC / BUILD_ID_MODULE).read_text(encoding="utf-8"))
    cached = [node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.decorator_list]
    assert cached == ["engine_build_id"]
    (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "engine_build_id"]
    assert not function.args.args and not function.args.kwonlyargs and len(function.decorator_list) == 1
    assert ast.unparse(function.decorator_list[0]) == "functools.lru_cache(maxsize=1)"


def test_module_level_mutable_containers_are_constants_only():
    for path in FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if isinstance(value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
                    for t in targets:
                        name = t.id if isinstance(t, ast.Name) else ""
                        assert name == "__all__" or re.fullmatch(r"_?[A-Z][A-Z0-9_]*", name), (
                            f"{path.relative_to(SRC)}:{node.lineno}: module-level mutable container {name!r} must be an ALL_CAPS constant")


PLACEHOLDER_WORDS = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b|NotImplementedError|placeholder|mock physics|fake result|hard-?coded success", re.IGNORECASE)


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_no_placeholders_or_stub_functions(path):
    text = path.read_text(encoding="utf-8")
    m = PLACEHOLDER_WORDS.search(text)
    assert m is None, f"{path.name}: placeholder marker {m.group(0)!r}"
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
            assert not (len(body) == 1 and isinstance(body[0], ast.Pass)), f"{path.name}:{node.lineno}: stub function {node.name}"


def _literal_hits(path: Path) -> list[tuple[int, object]]:
    """Numeric literals that look like tolerances / limits: floats outside a tiny allow-list and ints > 8, excluding docstrings and
    values assigned to module-level ALL_CAPS constants (which are named, documented values)."""
    allowed_floats = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 0.25, -1.0, -0.5, 0.1}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if all(isinstance(t, ast.Name) and re.fullmatch(r"_?[A-Z][A-Z0-9_]*", t.id) for t in targets):
                skip.update(id(n) for n in ast.walk(node))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) not in skip:
            v = node.value
            if (isinstance(v, float) and v not in allowed_floats) or (isinstance(v, int) and not isinstance(v, bool) and abs(v) > 8):
                hits.append((node.lineno, v))
    return hits


SCANNED = [p for p in CORE_FILES if p.name not in {"config.py", "_version.py", "units.py", "cli.py"} and "contracts" not in p.parts]


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: p.relative_to(SRC).as_posix())
def test_no_scattered_numeric_tolerances_outside_config(path):
    """Tolerances / thresholds / limits live in config.py (documented) or in named module constants."""
    assert _literal_hits(path) == [], f"{path.relative_to(SRC)}: unnamed numeric literals {_literal_hits(path)}"


def test_config_documents_every_tolerance():
    from os_optics.config import NumericalConfig

    undocumented = [n for n, f in NumericalConfig.model_fields.items() if n != "config_version" and not f.description]
    assert undocumented == []


def test_the_result_path_adds_floats_with_correctly_rounded_sum():
    """The built-in ``sum`` compensates floats from Python 3.12, so a value it adds differs in its last bits between Python 3.11 and
    the Pyodide line (NUMERICAL_CONVENTIONS section 4). The trace, the validation and the result queries use
    ``correctly_rounded_sum`` or ``math.fsum``; counts may stay, and so may arrays, which add elementwise with ``+`` on every Python."""
    integer_elements = {"1", "outcome.not_followed", "e.sample_count * len(e.spectrum)"}
    array_arguments = {"per_beam[1:]"}
    array_elements = {"np.abs(field) ** 2"}
    offenders = []
    for package in ("optics", "results", "scene", "validation"):
        for path in sorted((SRC / package).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum":
                    argument = node.args[0]
                    generator = isinstance(argument, ast.GeneratorExp)
                    element = ast.get_source_segment(text, argument.elt) if generator else None
                    counts = element in integer_elements or (generator and isinstance(argument.elt, ast.Call) and getattr(argument.elt.func, "id", "") == "len")
                    if not counts and element not in array_elements and ast.get_source_segment(text, argument) not in array_arguments:
                        offenders.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")
    assert offenders == []


def test_a_release_after_0_9_0_moves_the_contract_schema_version():
    """0.9.0 was published with SCHEMA_VERSION 1.6.0; the public contracts gained fields after it (coating group-delay
    bounds, bounded coherence results), so the next ENGINE_VERSION must come with SCHEMA_VERSION 1.7.0 or later."""

    from os_optics._version import ENGINE_VERSION, SCHEMA_VERSION

    if ENGINE_VERSION != "0.9.0":
        assert tuple(int(p) for p in SCHEMA_VERSION.split(".")) >= (1, 7, 0)


def test_a_release_after_0_10_0_moves_the_contract_schema_version():
    """0.10.0 was published with SCHEMA_VERSION 1.7.0; the frozen PlaneSurface gained an interaction kind after it (the Faraday
    rotator), so the next ENGINE_VERSION must come with SCHEMA_VERSION 1.8.0 or later."""

    from os_optics._version import ENGINE_VERSION, SCHEMA_VERSION

    if ENGINE_VERSION != "0.10.0":
        assert tuple(int(p) for p in SCHEMA_VERSION.split(".")) >= (1, 8, 0)


def test_version_single_source_of_truth():
    import tomllib

    from os_optics import __version__
    from os_optics._version import ENGINE_VERSION, MIN_PYTHON

    assert __version__ == ENGINE_VERSION
    project = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in project["project"]["dynamic"] and project["tool"]["hatch"]["version"]["path"].endswith("_version.py")
    assert project["project"]["requires-python"] == ">={}.{}".format(*MIN_PYTHON)
    names = {d.split(">")[0].split("<")[0].split("=")[0].strip() for d in project["project"]["dependencies"]}
    assert names == {"numpy", "scipy", "pydantic", "shapely"}
