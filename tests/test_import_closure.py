"""The import closure of the JSON API: what ``import os_optics.local.json_api`` loads (tests/fixtures/import_closure.json).

A headless client importing this JSON API pays for all dependencies loaded at startup.
Its cost is decided by the modules it loads, not by bytes: one eager ``import scipy.signal`` adds more than
150 modules (scipy.signal, stats, ndimage, integrate) and seconds of startup. The number of os_optics modules is not
limited (it grows with every feature, so a limit would only ever be raised). This test imports the module in a fresh interpreter (the test process has already
imported much more) and requires the third-party packages and the scipy subpackages it loads to equal the committed
declaration, so a new one is a deliberate, reviewed decision instead of a slower boot nobody chose.

Private names (a leading underscore) are ignored: they are implementation modules of the declared packages and differ
between their releases and builds. Modules loaded by the interpreter's own startup (``site``, ``sitecustomize``, a
virtualenv's ``.pth`` hooks) are subtracted, so the result is the same on every host and supported Python.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DECLARATION_FILE = Path(__file__).resolve().parent / "fixtures" / "import_closure.json"
DECLARATION = json.loads(DECLARATION_FILE.read_text(encoding="utf-8"))
DECLARATION_PATH = "tests/fixtures/import_closure.json"

# Runs in the fresh interpreter. A meta-path hook records, for every module imported for the first time, the innermost
# os_optics module executing at that moment, so a failure can name the line that pulled a package in.
_PROBE = r"""
import json, sys

class _Origin:
    origin = {}
    def find_spec(self, name, path=None, target=None):
        if name not in self.origin:
            frame = sys._getframe(1)
            while frame is not None:
                module = frame.f_globals.get("__name__", "")
                if module == "os_optics" or module.startswith("os_optics."):
                    self.origin[name] = f"{module} (line {frame.f_lineno})"
                    break
                frame = frame.f_back
        return None

hook = _Origin()
startup = set(sys.modules)
sys.meta_path.insert(0, hook)
import importlib
importlib.import_module(MODULE)
sys.meta_path.remove(hook)
loaded = sorted(set(sys.modules) - startup)
files = {name for name in loaded if getattr(sys.modules[name], "__file__", None) or hasattr(sys.modules[name], "__path__")}
print(json.dumps({"loaded": loaded, "with_source": sorted(files), "origin": hook.origin}))
"""


def _public(name: str) -> bool:
    return not name.startswith("_")


def measure(module: str) -> dict[str, object]:
    """Import ``module`` in a fresh interpreter; the third-party packages and scipy subpackages it loads."""
    probe = f"MODULE = {module!r}\n{_PROBE}"
    # -I: no current directory, PYTHONPATH or user site on sys.path, so only the installed environment is imported.
    result = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=300,
                            check=False)
    assert result.returncode == 0, f"`import {module}` failed in a fresh interpreter:\n{result.stderr}"
    data = json.loads(result.stdout)
    loaded: list[str] = data["loaded"]
    stdlib = set(sys.stdlib_module_names)
    packages = {name for name in data["with_source"]
                if "." not in name and name not in stdlib and name != "os_optics" and _public(name)}
    scipy = {name.split(".")[1] for name in loaded if name.startswith("scipy.") and _public(name.split(".")[1])}
    counts = {top: sum(1 for name in loaded if name.split(".")[0] == top) for top in packages | {"os_optics"}}
    return {"packages": packages, "scipy": scipy, "counts": counts, "total": len(loaded), "origin": data["origin"]}


def _differences(kind: str, key: str, measured: set[str], declared: set[str], origin: dict[str, str],
                 prefix: str = "") -> tuple[list[str], list[str]]:
    added = []
    for name in sorted(measured - declared):
        where = origin.get(prefix + name)
        added.append(f"- {kind} {prefix}{name}" + (f" (first imported while {where} executed)" if where else "")
                     + f'; declaration entry: "{key}": {{"{name}": "<reason>"}}')
    removed = [f'- {kind} {prefix}{name}: remove "{name}" from "{key}"' for name in sorted(declared - measured)]
    return added, removed


def test_json_api_import_loads_exactly_the_declared_packages():
    module = DECLARATION["module"]
    measured = measure(module)
    origin = measured["origin"]
    packages = _differences("third-party package", "third_party_packages", measured["packages"],
                            set(DECLARATION["third_party_packages"]), origin)
    scipy = _differences("scipy subpackage", "scipy_subpackages", measured["scipy"],
                         set(DECLARATION["scipy_subpackages"]), origin, "scipy.")
    added, removed = packages[0] + scipy[0], packages[1] + scipy[1]
    if not added and not removed:
        return
    sizes = ", ".join(f"{top} {count}" for top, count in sorted(measured["counts"].items(), key=lambda kv: -kv[1]))
    message = [f"`import {module}` (the headless JSON API startup) no longer loads "
               f"the declared import closure in {DECLARATION_PATH}. Modules loaded now: {measured['total']} ({sizes})."]
    if added:
        message += ["", "Newly loaded at import time:", *added, "",
                    "Each one is paid for by every cold start of a JSON host. If only some methods need it, move the "
                    "import into the functions that use it (a local `import` inside the function), so it loads on "
                    "first use. If it must load at import time, add the declaration entry shown, with the reason, and "
                    "say why in the change's description."]
    if removed:
        message += ["", "Declared but no longer loaded (keep the declaration exact):", *removed]
    pytest.fail("\n".join(message), pytrace=False)


def test_declaration_gives_a_reason_for_every_entry():
    for key in ("third_party_packages", "scipy_subpackages"):
        entries = DECLARATION[key]
        assert entries and all(isinstance(reason, str) and reason.strip() for reason in entries.values()), key
        assert all(_public(name) and "." not in name for name in entries), key
    assert "scipy" in DECLARATION["third_party_packages"] or not DECLARATION["scipy_subpackages"]
