"""A local-only web server for one viewer page: 127.0.0.1 only, an ephemeral port, an unguessable path.

Every file it serves is held in memory (the page assets and the view document); it serves nothing else, writes nothing,
and stops when ``close()`` is called or the Python process ends (the serving thread is a daemon thread). Requests with
another host name, any other path, or a method other than GET / HEAD are refused.
"""

from __future__ import annotations

import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .page import media_type

LOOPBACK = "127.0.0.1"
TOKEN_BYTES = 24  # 192 bits of randomness in the page path
LOCAL_HOST_NAMES = (LOOPBACK, "localhost")
SECURITY_HEADERS = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data: blob:; "
                                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
)


class _Handler(BaseHTTPRequestHandler):
    server_version = "OpenSciViewer"
    sys_version = ""
    files: dict[str, bytes] = {}
    base_path = "/"
    port = 0

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        return host in {f"{name}:{self.port}" for name in LOCAL_HOST_NAMES}

    def _send(self, status: HTTPStatus, body: bytes = b"", content_type: str = "text/plain; charset=utf-8", *, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _serve(self, head: bool) -> None:
        if not self._host_ok():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", head=head)
            return
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == self.base_path.rstrip("/"):
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", self.base_path)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not path.startswith(self.base_path):
            self._send(HTTPStatus.NOT_FOUND, b"not found", head=head)
            return
        name = path[len(self.base_path):] or "index.html"
        body = self.files.get(name)
        if body is None:
            self._send(HTTPStatus.NOT_FOUND, b"not found", head=head)
            return
        self._send(HTTPStatus.OK, body, media_type(name), head=head)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self._serve(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(head=True)

    def _refuse(self) -> None:
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, b"method not allowed")

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse  # noqa: N815

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - silent: nothing is logged
        return


class ViewerServer:
    """Serves one viewer page from memory. ``url`` is the local address; ``close()`` stops the server.

    Use as a context manager, or keep the object alive while the page is open (``wait()`` blocks until Ctrl+C)."""

    def __init__(self, files: dict[str, bytes], *, port: int = 0):
        token = secrets.token_urlsafe(TOKEN_BYTES)
        handler = type("OpenSciViewerHandler", (_Handler,), {"files": dict(files), "base_path": f"/{token}/"})
        self._httpd = ThreadingHTTPServer((LOOPBACK, port), handler)
        self._httpd.daemon_threads = True
        handler.port = self._httpd.server_address[1]
        self._base_path = handler.base_path
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="opensci-viewer", daemon=True)
        self._thread.start()
        self._closed = False

    @property
    def host(self) -> str:
        return LOOPBACK

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}{self._base_path}"

    def open_browser(self) -> bool:
        """Open the page in the default browser; False when no browser could be started."""

        try:
            return bool(webbrowser.open(self.url, new=2))
        except webbrowser.Error:
            return False

    def wait(self) -> None:
        """Block until Ctrl+C (or ``close()`` from another thread), then stop the server."""

        stop = threading.Event()
        try:
            while self._thread.is_alive() and not stop.wait(1.0):
                continue
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> ViewerServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<ViewerServer {self.url} {'closed' if self._closed else 'running'}>"
