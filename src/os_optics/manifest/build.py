"""Build identity of the installed engine (read-only; computed once per process)."""

from __future__ import annotations

import functools
import hashlib
import sys
from importlib import resources
from typing import Any

import numpy as np
import pydantic

from .._version import ENGINE_VERSION

BUILD_ID_HEX_CHARS = 16


def _hash_sources(root: Any) -> str:
    """``<version>+<sha256[:16] of the .py sources under root>`` (``root``: the package directory, a ``Traversable``); every
    call walks and hashes the files. Line endings are normalized so a wheel built on Windows and on Linux from the same
    sources agrees."""
    entries: list[tuple[str, bytes]] = []
    stack = [(root, "")]
    while stack:
        node, prefix = stack.pop()
        for child in node.iterdir():
            name = child.name
            if child.is_dir():
                if name != "__pycache__":
                    stack.append((child, f"{prefix}{name}/"))
            elif name.endswith(".py"):
                entries.append((f"{prefix}{name}", child.read_bytes().replace(b"\r\n", b"\n")))
    entries.sort(key=lambda e: e[0])
    digest = hashlib.sha256()
    for rel, data in entries:
        digest.update(rel.encode("utf-8") + b"\0" + data + b"\0")
    return f"{ENGINE_VERSION}+{digest.hexdigest()[:BUILD_ID_HEX_CHARS]}"


@functools.lru_cache(maxsize=1)
def engine_build_id() -> str:
    """``<version>+<sha256[:16] of the package .py sources>``; identical for the same wheel on every runtime.

    Computed once per process: the installed sources of a running process are those it imported. Falls back to the bare
    version if the package files cannot be enumerated (e.g. a zipped import)."""
    try:
        return _hash_sources(resources.files("os_optics"))
    except (OSError, ModuleNotFoundError, AttributeError):  # pragma: no cover - environment dependent
        return ENGINE_VERSION


def runtime_info() -> dict[str, str]:
    """Descriptive only (never influences results)."""
    return {
        "python": ".".join(str(p) for p in sys.version_info[:3]),
        "platform": sys.platform,
        "numpy": np.__version__,
        "pydantic": pydantic.VERSION,
    }
