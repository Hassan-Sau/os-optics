"""HTTP transport of the OpenSci commercial client (standard library ``urllib`` only).

* The service URL is ``OPENSCI_API_URL`` (or the ``api_url`` argument): the OpenSci service URL your OpenSci Account
  shows with your Agent token; requests go to ``<api_url>/api/v1/...``. Plain ``http`` is accepted for this device (loopback) only; any other host needs ``https``.
* The Account token is sent only in the ``Authorization`` header of requests to that service: never in a URL, never
  to another host (redirects are not followed).
* Transient conditions are repeated with bounded exponential back-off, always with the same idempotency key (so a
  repeated create replays instead of creating twice): a request that failed to reach the service, ``502`` / ``503`` /
  ``504`` (including ``503 RESOLUTION_CAPACITY_BUSY``: the service is at capacity for resolutions right now), and
  ``409 IDEMPOTENCY_IN_PROGRESS`` (the request with this key is still being processed). ``503
  ARTIFACT_STORAGE_NOT_CONFIGURED`` (recovery ``CONTACT_OPENSCI``: the service has no artifact storage) is not repeated.

  - At most ``retries`` repeats (default 3). The wait before repeat ``n`` (0-based) is drawn with "equal jitter" from
    ``[c / 2, c]``, where ``c = min(8, 0.5 * 2 ** n)`` seconds (0.5, 1, 2, 4, 8, 8, ... s).
  - A ``Retry-After`` header in delta-seconds form replaces that wait (no jitter), at least 0.25 s and at most 30 s;
    any other form is ignored.
  - The waits of one call add up to at most ``max_retry_seconds`` (default 60 s): when the next wait would pass it,
    the client stops and raises the last refusal. The socket ``timeout`` of each attempt is separate.

  Every other refusal (``400``, ``401``, ``403``, ``404``, ``409`` with another code such as
  ``IDEMPOTENCY_KEY_CONFLICT``, ``413``, ``422``, ...) is never repeated. A refusal is raised exactly as the service
  stated it (``code``, ``message``, ``details``, ``recovery``, ``http_status``, ``request_id``); after the last repeat
  it is the last refusal received.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .._version import DISTRIBUTION_NAME, __version__
from .credentials import AccountToken
from .errors import (
    API_URL_INVALID,
    NON_TRANSIENT_CODES,
    RETRYABLE_CODES,
    RETRYABLE_STATUSES,
    SERVICE_UNREACHABLE,
    UNEXPECTED_RESPONSE,
    OpenSciApiError,
    OpenSciConfigurationError,
    OpenSciConnectionError,
    Recovery,
    error_from_response,
)

ENV_API_URL = "OPENSCI_API_URL"
API_PREFIX = "/api/v1"
IDEMPOTENCY_HEADER = "Idempotency-Key"
REQUEST_ID_HEADER = "x-request-id"
RETRY_AFTER_HEADER = "retry-after"
USER_AGENT = f"{DISTRIBUTION_NAME}/{__version__}"
DEFAULT_TIMEOUT_SECONDS = 300.0
#: The most repeats of one request after a transient condition (``retries``).
DEFAULT_RETRIES = 3
#: Exponential back-off: the wait ceiling before repeat ``n`` is ``min(RETRY_MAX_DELAY_SECONDS,
#: RETRY_BASE_DELAY_SECONDS * RETRY_BACKOFF_FACTOR ** n)``; ``RETRY_BACKOFF_SECONDS`` lists these ceilings up to the cap
#: (the last one repeats for later attempts).
RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_BACKOFF_FACTOR = 2
RETRY_MAX_DELAY_SECONDS = 8.0
RETRY_BACKOFF_SECONDS = tuple(min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * RETRY_BACKOFF_FACTOR ** n)
                              for n in range(math.ceil(math.log(RETRY_MAX_DELAY_SECONDS / RETRY_BASE_DELAY_SECONDS, RETRY_BACKOFF_FACTOR)) + 1))
#: ``Retry-After`` (delta-seconds) is honoured within these bounds.
RETRY_AFTER_MIN_SECONDS = 0.25
RETRY_AFTER_MAX_SECONDS = 30.0
RETRY_AFTER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
#: The sum of all waits of one call never exceeds this (``max_retry_seconds``).
DEFAULT_MAX_RETRY_SECONDS = 60.0
LOOPBACK_NAMES = frozenset({"localhost"})
HTTP_OK_MIN = 200
HTTP_OK_MAX = 299
#: urllib runs request pre-processors in this order; after its own (500) so the prepared request gets the header.
BEARER_HANDLER_ORDER = 900
MAX_REASON_CHARS = 300

logger = logging.getLogger("os_optics.client")


def _is_loopback(host: str) -> bool:
    name = host.strip("[]").lower()
    if name in LOOPBACK_NAMES or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def normalize_api_url(url: str, *, allow_insecure_http: bool = False) -> str:
    """The service origin (scheme, host, port) of ``url``; a trailing ``/api/v1`` is accepted and dropped."""

    def invalid(reason: str) -> OpenSciConfigurationError:
        return OpenSciConfigurationError(API_URL_INVALID, f"the OpenSci service URL is not usable: {reason}",
                                         details={"variable": ENV_API_URL}, recovery=Recovery.FIX_REQUEST)

    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise invalid("expected http(s)://host[:port]")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise invalid("it must not carry credentials, a query or a fragment")
    path = parts.path.rstrip("/")
    if path not in ("", API_PREFIX):
        raise invalid(f"expected the service origin (optionally followed by {API_PREFIX})")
    if parts.scheme == "http" and not allow_insecure_http and not _is_loopback(parts.hostname):
        raise invalid("plain http is accepted for this device (localhost) only; use https")
    try:
        port = parts.port
    except ValueError:
        raise invalid("the port is not a number") from None
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}" + (f":{port}" if port is not None else "")


class HttpResponse:
    """One 2xx response. ``body`` holds the raw bytes (never shown by ``repr``)."""

    __slots__ = ("body", "headers", "status")

    def __init__(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
        self.status = status
        self.headers = {name.lower(): value for name, value in headers.items()}
        self.body = body

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise OpenSciApiError(UNEXPECTED_RESPONSE, "the service answered with a body that is not JSON",
                                  family="SERVER_ERROR", recovery=Recovery.CONTACT_OPENSCI, http_status=self.status,
                                  request_id=self.headers.get(REQUEST_ID_HEADER)) from None

    def take_body(self) -> bytes:
        """The body, dropped from this response (for bodies that carry a secret)."""

        body, self.body = self.body, b""
        return body

    def __repr__(self) -> str:
        return f"HttpResponse(status={self.status}, bytes={len(self.body)})"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any) -> None:
        return None


class _Bearer(urllib.request.BaseHandler):
    """Adds the ``Authorization`` header while the request is prepared, so no caller frame holds the header text."""

    handler_order = BEARER_HANDLER_ORDER

    def __init__(self, token: AccountToken) -> None:
        self._token = token

    def _add(self, request: urllib.request.Request) -> urllib.request.Request:
        if request.get_header("X-opensci-authorize") is not None:
            request.remove_header("X-opensci-authorize")
            request.add_unredirected_header("Authorization", self._token.authorization())
        return request

    http_request = _add
    https_request = _add

    def __repr__(self) -> str:
        return f"_Bearer({self._token!r})"


class Transport:
    """Sends requests to one OpenSci service with one Account token.

    ``retries`` is the most repeats of one request and ``max_retry_seconds`` the most time spent waiting between its
    attempts (see the module docstring). ``sleep`` (default ``time.sleep``) waits that many seconds; ``jitter``
    (default: a system random source) returns a fraction in ``[0, 1]`` that places each back-off wait within
    ``[c / 2, c]``."""

    def __init__(self, api_url: str, token: AccountToken | None, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 retries: int = DEFAULT_RETRIES, allow_insecure_http: bool = False,
                 max_retry_seconds: float = DEFAULT_MAX_RETRY_SECONDS, sleep: Callable[[float], Any] | None = None,
                 jitter: Callable[[], float] | None = None) -> None:
        self.api_url = normalize_api_url(api_url, allow_insecure_http=allow_insecure_http)
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.max_retry_seconds = max(0.0, float(max_retry_seconds))
        self._sleep = sleep if sleep is not None else time.sleep
        self._jitter = jitter if jitter is not None else secrets.SystemRandom().random
        host = urllib.parse.urlsplit(self.api_url).hostname or ""
        handlers: list[Any] = [_NoRedirect()]
        if token is not None:
            handlers.append(_Bearer(token))
        if _is_loopback(host):
            handlers.append(urllib.request.ProxyHandler({}))  # a proxy cannot reach this device's loopback service
        self._opener = urllib.request.build_opener(*handlers)
        self._has_token = token is not None

    def __repr__(self) -> str:
        return f"Transport(api_url={self.api_url!r})"

    def url(self, path: str, query: Mapping[str, str] | None = None) -> str:
        suffix = ("?" + urllib.parse.urlencode(dict(query))) if query else ""
        return f"{self.api_url}{API_PREFIX}{path}{suffix}"

    def request(self, method: str, path: str, *, query: Mapping[str, str] | None = None, json_body: Any = None,
                data: bytes | None = None, content_type: str | None = None, idempotency_key: str | None = None,
                accept: str = "application/json", authorized: bool = True) -> HttpResponse:
        """Send one request (repeated on a transient condition, with the same idempotency key) and return its 2xx
        response; raise ``OpenSciApiError`` for anything else (the service's refusal unchanged)."""

        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            content_type = "application/json"
        headers = {"Accept": accept, "User-Agent": USER_AGENT}
        if content_type:
            headers["Content-Type"] = content_type
        if idempotency_key is not None:
            headers[IDEMPOTENCY_HEADER] = idempotency_key
        if authorized and self._has_token:
            headers["X-opensci-authorize"] = "1"
        url = self.url(path, query)
        log_path = urllib.parse.urlsplit(url).path
        attempt = 0
        waited = 0.0
        while True:
            outcome = self._send(method, url, data, headers)
            if isinstance(outcome, HttpResponse):
                return outcome
            if not _is_transient(outcome) or attempt >= self.retries:
                break
            delay = self._delay(attempt, outcome.retry_after_seconds)
            if waited + delay > self.max_retry_seconds:
                logger.debug("%s %s -> %s: retry budget of %.1f s spent, not repeating", method, log_path,
                             _condition(outcome), self.max_retry_seconds)
                break
            logger.debug("%s %s -> %s: repeat %d of %d in %.2f s", method, log_path, _condition(outcome), attempt + 1,
                         self.retries, delay)
            self._sleep(delay)
            waited += delay
            attempt += 1
        raise outcome

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        """The wait before repeat ``attempt`` (0-based): the service's ``Retry-After`` within its bounds, else the
        equal-jitter exponential back-off."""

        if retry_after is not None:
            return min(RETRY_AFTER_MAX_SECONDS, max(RETRY_AFTER_MIN_SECONDS, retry_after))
        ceiling = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
        fraction = min(1.0, max(0.0, float(self._jitter())))
        return ceiling / 2 + fraction * ceiling / 2

    def _send(self, method: str, url: str, data: bytes | None, headers: Mapping[str, str]) -> HttpResponse | OpenSciApiError:
        # Failures are returned, not raised inside the except blocks, so the raised error chains to no urllib exception
        # (whose frames hold the prepared request headers).
        request = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
        status: int | None = None
        failure: str | None = None
        response_headers: Mapping[str, str] = {}
        body = b""
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                status, response_headers, body = response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:
            status, response_headers = exc.code, dict(exc.headers or {})
            try:
                body = exc.read()
            except OSError:
                body = b""
            exc.close()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            failure = type(exc).__name__ + (f": {exc.reason}" if isinstance(exc, urllib.error.URLError) else "")
        del request
        path = urllib.parse.urlsplit(url).path
        if failure is not None or status is None:
            logger.debug("%s %s -> unreachable", method, path)
            return OpenSciConnectionError(SERVICE_UNREACHABLE, "the OpenSci service could not be reached",
                                          details={"api_url": self.api_url, "reason": str(failure)[:MAX_REASON_CHARS]},
                                          recovery=Recovery.RETRY)
        lowered = {k.lower(): v for k, v in response_headers.items()}
        request_id = lowered.get(REQUEST_ID_HEADER)
        logger.debug("%s %s -> %s (request %s)", method, path, status, request_id)
        if HTTP_OK_MIN <= status <= HTTP_OK_MAX:
            return HttpResponse(status, response_headers, body)
        error = error_from_response(status, body, request_id)
        error.retry_after_seconds = parse_retry_after(lowered.get(RETRY_AFTER_HEADER))
        return error


def parse_retry_after(value: str | None) -> float | None:
    """Seconds of a ``Retry-After`` header in delta-seconds form; ``None`` for no header or any other form."""

    text = (value or "").strip()
    if not RETRY_AFTER_PATTERN.fullmatch(text):
        return None
    seconds = float(text)
    return seconds if math.isfinite(seconds) else None


def _is_transient(error: OpenSciApiError) -> bool:
    """Whether the condition is repeated: unreachable, ``502`` / ``503`` / ``504``, or a transient service code -- never a
    code the service declares not transient (``NON_TRANSIENT_CODES``)."""

    if isinstance(error, OpenSciConnectionError):
        return True
    if error.http_status is None or error.code in NON_TRANSIENT_CODES:
        return False
    return error.http_status in RETRYABLE_STATUSES or error.code in RETRYABLE_CODES


def _condition(error: OpenSciApiError) -> str:
    return error.code if error.http_status is None else f"{error.http_status} {error.code}"


def encode_multipart(fields: Sequence[tuple[str, tuple[str, bytes, str]]], boundary: str) -> tuple[bytes, str]:
    """``multipart/form-data`` of ``(name, (filename, bytes, media_type))`` parts, and its ``Content-Type``."""

    chunks: list[bytes] = []
    for name, (filename, content, media_type) in fields:
        chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                       f"Content-Type: {media_type}\r\n\r\n").encode("utf-8"))
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


__all__ = ["API_PREFIX", "ENV_API_URL", "IDEMPOTENCY_HEADER", "HttpResponse", "Transport", "encode_multipart",
           "normalize_api_url", "parse_retry_after"]
