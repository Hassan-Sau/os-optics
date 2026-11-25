"""The viewer page: the packaged HTML / CSS / JS assets and the bundled three.js library, served or inlined.

* Served (``page_files``): ``index.html`` references ``viewer.css``, ``three.bundle.js``, ``viewer.js`` and fetches
  ``view.json`` from the same local address.
* Self-contained (``standalone_html``): one HTML file with everything inlined, including the view document, for a
  local optical run that can be shared as a single file.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any

ASSETS = ("index.html", "viewer.css", "viewer.js", "three.bundle.js", "THREE_LICENSE.txt")
MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
}
DOCUMENT_FILE = "view.json"
SCRIPT_CLOSE = re.compile(r"</(script)", re.IGNORECASE)


def asset_bytes(name: str) -> bytes:
    if name not in ASSETS:
        raise KeyError(name)
    return resources.files(__package__).joinpath("assets", name).read_bytes()


def media_type(name: str) -> str:
    return MEDIA_TYPES.get(name[name.rfind("."):], "application/octet-stream")


def document_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def page_files(document: dict[str, Any]) -> dict[str, bytes]:
    """Every file the served page needs, in memory: the assets and ``view.json``."""

    html = asset_bytes("index.html").decode("utf-8")
    html = html.replace("<!--STYLE-->", '<link rel="stylesheet" href="viewer.css">')
    html = html.replace("<!--DOCUMENT-->", "")
    html = html.replace("<!--SCRIPTS-->", '<script src="three.bundle.js"></script>\n<script src="viewer.js"></script>')
    files = {name: asset_bytes(name) for name in ASSETS if name != "index.html"}
    files["index.html"] = html.encode("utf-8")
    files[DOCUMENT_FILE] = document_bytes(document)
    return files


def _inline_script(code: str) -> str:
    return "<script>\n" + SCRIPT_CLOSE.sub(r"<\\/\1", code) + "\n</script>"


def standalone_html(document: dict[str, Any]) -> bytes:
    """One self-contained HTML file (no network access needed): styles, three.js, the viewer and the view document."""

    html = asset_bytes("index.html").decode("utf-8")
    css = asset_bytes("viewer.css").decode("utf-8")
    data = document_bytes(document).decode("utf-8").replace("</", "<\\/").replace("<!--", "<\\u0021--")
    html = html.replace("<!--STYLE-->", "<style>\n" + css + "\n</style>")
    html = html.replace("<!--DOCUMENT-->", '<script id="opensci-view-document" type="application/json">' + data + "</script>")
    scripts = _inline_script(asset_bytes("three.bundle.js").decode("utf-8")) + "\n" + _inline_script(asset_bytes("viewer.js").decode("utf-8"))
    html = html.replace("<!--SCRIPTS-->", scripts)
    return html.encode("utf-8")
