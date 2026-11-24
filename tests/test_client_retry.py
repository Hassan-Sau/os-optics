"""Bounded exponential back-off of the commercial client transport (``os_optics.client.transport``).

A tiny scripted HTTP server on 127.0.0.1 answers a fixed sequence of responses and records every request; sleeping and
jitter are injected, so no test waits for real.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from os_optics.client import OpenSciApiError, OpenSciConnectionError, Recovery
from os_optics.client import transport as transport_module
from os_optics.client.errors import RETRYABLE_CODES, RESOLUTION_CAPACITY_BUSY, derive_recovery
from os_optics.client.transport import Transport, parse_retry_after

KEY = "retry-test-key-0001"
FAMILIES = {400: "REJECTED", 401: "UNAUTHENTICATED", 404: "NOT_FOUND", 409: "CONFLICT", 422: "REJECTED",
            502: "SERVER_ERROR", 503: "SERVER_ERROR", 504: "SERVER_ERROR"}


def refusal(status, code, *, recovery=None, details=None, retry_after=None, request_id=None):
    error = {"code": code, "family": FAMILIES[status], "message": f"the service says {code.lower()}",
             "details": details if details is not None else {"field": "x", "n": 1}}
    if recovery is not None:
        error["recovery"] = recovery
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return status, {"error": error, "request_id": request_id or f"req-{status}-{code.lower()}"}, headers


def ok(status=200, document=None):
    return status, document if document is not None else {"ok": True}, {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return None

    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                                     "body": body})
        status, document, headers = self.server.script.pop(0) if self.server.script else ok()
        payload = json.dumps(document).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _answer
    do_POST = _answer


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.script, httpd.requests = [], []
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    httpd.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


class Sleeps(list):
    def __call__(self, seconds):
        self.append(seconds)


def transport(url, *, retries=4, jitter=1.0, **kw):
    sleeps = Sleeps()
    return Transport(url, None, retries=retries, sleep=sleeps, jitter=lambda: jitter, **kw), sleeps


def post(t):
    return t.request("POST", "/commercial-submissions", json_body={"a": 1}, idempotency_key=KEY)


def keys(server):
    return [r["headers"].get("Idempotency-Key") for r in server.requests]


# ------------------------------------------------------------------------------------------ schedule


def test_backoff_schedule_constants():
    assert transport_module.RETRY_BACKOFF_SECONDS == (0.5, 1.0, 2.0, 4.0, 8.0)
    assert transport_module.DEFAULT_RETRIES == 3 and transport_module.DEFAULT_MAX_RETRY_SECONDS == 60.0
    assert RESOLUTION_CAPACITY_BUSY in RETRYABLE_CODES and "IDEMPOTENCY_IN_PROGRESS" in RETRYABLE_CODES
    assert derive_recovery(503, RESOLUTION_CAPACITY_BUSY, {}) is Recovery.RETRY_LATER


@pytest.mark.parametrize(("value", "seconds"), [("3", 3.0), (" 12 ", 12.0), ("0", 0.0), ("1.5", 1.5), (None, None),
                                                ("", None), ("-1", None), ("soon", None), ("inf", None), ("nan", None),
                                                ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("1e3", None)])
def test_retry_after_is_read_in_delta_seconds_form_only(value, seconds):
    assert parse_retry_after(value) == seconds


def test_503_then_200_repeats_once_with_the_same_key(server):
    server.script[:] = [refusal(503, "SERVICE_UNAVAILABLE"), ok(201, {"created": True})]
    t, sleeps = transport(server.url)
    response = post(t)
    assert response.status == 201 and response.json() == {"created": True}
    assert len(server.requests) == 2 and keys(server) == [KEY, KEY]
    assert server.requests[0]["body"] == server.requests[1]["body"]
    assert sleeps == [0.5]


@pytest.mark.parametrize(("jitter", "expected"), [(1.0, [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]),
                                                  (0.0, [0.25, 0.5, 1.0, 2.0, 4.0, 4.0]),
                                                  (0.5, [0.375, 0.75, 1.5, 3.0, 6.0, 6.0])])
def test_the_waits_follow_the_equal_jitter_exponential_schedule(server, jitter, expected):
    server.script[:] = [refusal(503, "SERVICE_UNAVAILABLE")] * 6 + [ok()]
    t, sleeps = transport(server.url, retries=6, jitter=jitter)
    assert post(t).status == 200
    assert sleeps == expected and len(server.requests) == 7 and set(keys(server)) == {KEY}


@pytest.mark.parametrize("status", [502, 504])
def test_502_and_504_are_repeated(server, status):
    server.script[:] = [refusal(status, "UPSTREAM_UNAVAILABLE"), refusal(status, "UPSTREAM_UNAVAILABLE"), ok()]
    t, sleeps = transport(server.url)
    assert post(t).status == 200
    assert len(server.requests) == 3 and sleeps == [0.5, 1.0] and set(keys(server)) == {KEY}


def test_resolution_capacity_busy_waits_the_advised_retry_after(server):
    server.script[:] = [refusal(503, RESOLUTION_CAPACITY_BUSY, retry_after="3"), ok()]
    t, sleeps = transport(server.url, jitter=0.0)
    assert post(t).status == 200
    assert sleeps == [3.0] and keys(server) == [KEY, KEY]


@pytest.mark.parametrize(("header", "expected"), [("99999", 30.0), ("0", 0.25), ("later", 0.5)])
def test_retry_after_is_bounded_and_unparseable_values_fall_back_to_backoff(server, header, expected):
    server.script[:] = [refusal(503, RESOLUTION_CAPACITY_BUSY, retry_after=header), ok()]
    t, sleeps = transport(server.url)
    assert post(t).status == 200
    assert sleeps == [expected]


def test_idempotency_in_progress_is_repeated_with_the_same_key(server):
    server.script[:] = [refusal(409, "IDEMPOTENCY_IN_PROGRESS", recovery="RETRY_WITH_SAME_KEY", retry_after="2"),
                        refusal(409, "IDEMPOTENCY_IN_PROGRESS", recovery="RETRY_WITH_SAME_KEY"),
                        ok(201, {"replayed": True})]
    t, sleeps = transport(server.url)
    assert post(t).json() == {"replayed": True}
    assert keys(server) == [KEY, KEY, KEY] and sleeps == [2.0, 1.0]


def test_idempotency_in_progress_without_a_key_is_repeated_without_one(server):
    server.script[:] = [refusal(409, "IDEMPOTENCY_IN_PROGRESS"), ok()]
    t, sleeps = transport(server.url)
    assert t.request("POST", "/account/token", json_body={}).status == 200
    assert keys(server) == [None, None] and sleeps == [0.5]


def test_idempotency_key_conflict_is_not_repeated_and_raised_unchanged(server):
    details = {"idempotency_key": KEY, "first_request": "other"}
    server.script[:] = [refusal(409, "IDEMPOTENCY_KEY_CONFLICT", details=details, request_id="req-conflict")]
    t, sleeps = transport(server.url)
    with pytest.raises(OpenSciApiError) as refused:
        post(t)
    error = refused.value
    assert (error.code, error.family, error.message, error.details, error.recovery, error.http_status, error.request_id) \
        == ("IDEMPOTENCY_KEY_CONFLICT", "CONFLICT", "the service says idempotency_key_conflict", details,
            Recovery.USE_NEW_IDEMPOTENCY_KEY, 409, "req-conflict")
    assert len(server.requests) == 1 and sleeps == []


@pytest.mark.parametrize(("status", "code", "recovery"), [
    (400, "REQUEST_INVALID", "FIX_REQUEST"), (401, "AUTHORIZATION_INVALID", "REQUEST_NEW_TOKEN"),
    (404, "RESOLUTION_NOT_FOUND", "FIX_REQUEST"), (422, "SUBMISSION_REJECTED", "FIX_REQUEST"),
    (409, "ORDER_ALREADY_EXISTS", "NONE")])
def test_other_refusals_are_not_repeated_and_raised_unchanged(server, status, code, recovery):
    details = {"problems": [{"path": "a.b", "reason": "bad"}]}
    server.script[:] = [refusal(status, code, recovery=recovery, details=details, retry_after="1",
                                request_id=f"req-{status}")]
    t, sleeps = transport(server.url)
    with pytest.raises(OpenSciApiError) as refused:
        post(t)
    error = refused.value
    assert (error.code, error.message, error.details, error.recovery.value, error.http_status, error.request_id) == (
        code, f"the service says {code.lower()}", details, recovery, status, f"req-{status}")
    assert error.retry_after_seconds == 1.0
    assert len(server.requests) == 1 and sleeps == []


def test_artifact_storage_not_configured_is_not_repeated_though_it_is_a_503(server):
    """The service answers an instance without artifact storage with 503 ARTIFACT_STORAGE_NOT_CONFIGURED (recovery
    CONTACT_OPENSCI, no Retry-After) and declares that sending the request again does not help: one request, no wait. A
    plain 503 and 503 RESOLUTION_CAPACITY_BUSY are still repeated (the tests above)."""
    from os_optics.client.errors import ARTIFACT_STORAGE_NOT_CONFIGURED, NON_TRANSIENT_CODES

    assert ARTIFACT_STORAGE_NOT_CONFIGURED in NON_TRANSIENT_CODES and not NON_TRANSIENT_CODES & RETRYABLE_CODES
    server.script[:] = [refusal(503, ARTIFACT_STORAGE_NOT_CONFIGURED, recovery="CONTACT_OPENSCI"), ok()]
    t, sleeps = transport(server.url)
    with pytest.raises(OpenSciApiError) as refused:
        post(t)
    assert (refused.value.code, refused.value.recovery, refused.value.http_status) == (
        ARTIFACT_STORAGE_NOT_CONFIGURED, Recovery.CONTACT_OPENSCI, 503)
    assert len(server.requests) == 1 and sleeps == []


def test_exhausted_retries_raise_the_last_refusal(server):
    server.script[:] = [refusal(503, "SERVICE_UNAVAILABLE", request_id="req-first"),
                        refusal(503, "SERVICE_UNAVAILABLE", request_id="req-second"),
                        refusal(503, RESOLUTION_CAPACITY_BUSY, request_id="req-last", details={"queue": "full"})]
    t, sleeps = transport(server.url, retries=2)
    with pytest.raises(OpenSciApiError) as refused:
        post(t)
    error = refused.value
    assert (error.code, error.recovery, error.http_status, error.request_id, error.details) == (
        RESOLUTION_CAPACITY_BUSY, Recovery.RETRY_LATER, 503, "req-last", {"queue": "full"})
    assert error.family == "SERVER_ERROR" and error.retry_after_seconds is None
    assert len(server.requests) == 3 and sleeps == [0.5, 1.0] and set(keys(server)) == {KEY}


def test_retries_zero_sends_once(server):
    server.script[:] = [refusal(503, "SERVICE_UNAVAILABLE")]
    t, sleeps = transport(server.url, retries=0)
    with pytest.raises(OpenSciApiError):
        post(t)
    assert len(server.requests) == 1 and sleeps == []


def test_the_total_wait_never_exceeds_the_budget(server):
    server.script[:] = [refusal(503, RESOLUTION_CAPACITY_BUSY, retry_after="4")] * 10
    t, sleeps = transport(server.url, retries=9, max_retry_seconds=10.0)
    with pytest.raises(OpenSciApiError) as refused:
        post(t)
    assert refused.value.code == RESOLUTION_CAPACITY_BUSY
    assert sleeps == [4.0, 4.0] and sum(sleeps) <= 10.0 and len(server.requests) == 3


def test_the_budget_bounds_the_backoff_waits_too(server):
    server.script[:] = [refusal(503, "SERVICE_UNAVAILABLE")] * 10
    t, sleeps = transport(server.url, retries=9, max_retry_seconds=5.0)
    with pytest.raises(OpenSciApiError):
        post(t)
    assert sleeps == [0.5, 1.0, 2.0] and sum(sleeps) <= 5.0 and len(server.requests) == 4


def test_the_client_constructor_passes_the_budget(server):
    from os_optics.client import OpenSciClient

    client = OpenSciClient(server.url, "oat_" + "0" * 32 + "_" + "1" * 64, environ={}, max_retry_seconds=0.0)
    assert client._http.retries == 3 and client._http.max_retry_seconds == 0.0
    server.script[:] = [refusal(503, "SERVICE_UNAVAILABLE")]
    with pytest.raises(OpenSciApiError) as refused:
        client.token_status()
    assert refused.value.http_status == 503 and len(server.requests) == 1


def test_retries_are_logged_without_secrets(server, caplog):
    server.script[:] = [refusal(503, RESOLUTION_CAPACITY_BUSY, retry_after="2"), ok()]
    t, _ = transport(server.url, retries=3)
    with caplog.at_level("DEBUG", logger="os_optics.client"):
        post(t)
    lines = [r.getMessage() for r in caplog.records]
    assert any("POST /api/v1/commercial-submissions -> 503 RESOLUTION_CAPACITY_BUSY: repeat 1 of 3 in 2.00 s" in line
               for line in lines)
    assert not any(KEY in line for line in lines)


def test_an_unreachable_service_is_repeated_with_backoff():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    t, sleeps = transport(f"http://127.0.0.1:{port}", retries=1, timeout=5)
    with pytest.raises(OpenSciConnectionError) as failed:
        post(t)
    assert failed.value.code == "SERVICE_UNREACHABLE" and failed.value.recovery is Recovery.RETRY
    assert sleeps == [0.5]


def test_the_services_retry_later_recovery_is_kept():
    """The service states ``RETRY_LATER`` for ``RESOLUTION_CAPACITY_BUSY`` (with Retry-After); the client keeps it."""

    from os_optics.client.errors import Recovery, error_from_response

    body = b'{"error": {"code": "RESOLUTION_CAPACITY_BUSY", "family": "SERVER_ERROR", "message": "busy", "recovery": "RETRY_LATER"}}'
    error = error_from_response(503, body, "req-1")
    assert (error.code, error.recovery, error.family) == ("RESOLUTION_CAPACITY_BUSY", Recovery.RETRY_LATER, "SERVER_ERROR")
    assert "Retry-After" in error.hint


def test_a_mechanical_refusal_maps_to_its_design_recovery() -> None:
    """Assembly Resolution refusals carry a supplier-neutral corrective action: the client keeps the
    service's recovery and, when none is stated, derives it from ``details.suggestion.action``."""

    from os_optics.client.errors import RECOVERY_HINTS, Recovery, derive_recovery

    for name in ("ADJUST_COMPONENT_SPACING", "MOVE_COMPONENT_OFF_BEAM", "ADJUST_COMPONENT_HEIGHT", "RELAX_RETAINED_DOF",
                 "REDUCE_TILT", "CHANGE_COMPONENT", "ALLOW_LARGER_BREADBOARD", "SIMPLIFY_DESIGN"):
        assert Recovery(name) in RECOVERY_HINTS
    details = {"instance_id": "SP532", "reason": "NO_FIXING_HOLE", "suggestion": {"action": "ADJUST_COMPONENT_SPACING"}}
    assert derive_recovery(422, "MECHANICAL_LAYOUT_CONFLICT", details) is Recovery.ADJUST_COMPONENT_SPACING
    assert derive_recovery(422, "MECHANICAL_LAYOUT_CONFLICT", {"suggestion": {"action": "NOT_A_RECOVERY"}}) is Recovery.FIX_REQUEST


def test_a_recovery_the_client_does_not_know_falls_back_to_the_derived_one() -> None:
    """The retired coherence refusal's recovery (retired with integrity policy version 2) is no client value any more:
    an answer naming it, or any unknown recovery, gets the recovery derived from status and code."""

    from os_optics.client.errors import Recovery, derive_recovery, error_from_response

    assert "DECLARE_SOURCE_COHERENCE" not in Recovery.__members__
    body = {"error": {"code": "SOURCE_COHERENCE_UNDECLARED", "family": "REJECTED", "message": "set the source coherence",
                      "details": {"targets": ["fringe_visibility"], "source_instance_ids": ["SRC"]},
                      "recovery": "DECLARE_SOURCE_COHERENCE"}, "request_id": "req-1"}
    error = error_from_response(422, json.dumps(body).encode(), "req-1")
    assert error.recovery is derive_recovery(422, "SOURCE_COHERENCE_UNDECLARED", body["error"]["details"]) is Recovery.FIX_REQUEST


def test_an_optical_system_submitted_as_a_complete_system_is_told_to_buy_the_optics_only() -> None:
    """DESIGN_INTENT_HAS_NO_LAB_ASSEMBLY answers CHOOSE_OPTICS_ONLY; the client keeps (or derives) it and its hint
    names the optics-only purchase."""

    from os_optics.client.errors import Recovery, derive_recovery, error_from_response

    body = {"error": {"code": "DESIGN_INTENT_HAS_NO_LAB_ASSEMBLY", "family": "REJECTED", "message": "buy the optics only",
                      "details": {"design_intent": "OPTICAL_SYSTEM", "supported_purchase_scopes": ["OPTICS_ONLY"]},
                      "recovery": "CHOOSE_OPTICS_ONLY"}, "request_id": "req-1"}
    error = error_from_response(422, json.dumps(body).encode(), "req-1")
    assert (error.code, error.recovery) == ("DESIGN_INTENT_HAS_NO_LAB_ASSEMBLY", Recovery.CHOOSE_OPTICS_ONLY)
    assert "--optics-only" in error.hint
    assert derive_recovery(422, "DESIGN_INTENT_HAS_NO_LAB_ASSEMBLY", {}) is Recovery.CHOOSE_OPTICS_ONLY


def test_an_engine_that_is_not_accepted_is_told_the_upgrade_path() -> None:
    """ENGINE_VERSION_NOT_ACCEPTED answers UPGRADE_ENGINE_AND_RERUN; the hint is the same path as the service's
    message (upgrade, catalog sync, project update-catalog, run again, submit)."""

    from os_optics.client.errors import Recovery, derive_recovery, error_from_response

    body = {"error": {"code": "ENGINE_VERSION_NOT_ACCEPTED", "family": "REJECTED", "message": "upgrade",
                      "details": {"accepted_engine_versions": ["9.9.9"]}, "recovery": "UPGRADE_ENGINE_AND_RERUN"},
            "request_id": "req-1"}
    error = error_from_response(422, json.dumps(body).encode(), "req-1")
    assert error.recovery is Recovery.UPGRADE_ENGINE_AND_RERUN
    for step in ('python -m pip install --upgrade "os-optics[protected]"', "os-optics catalog sync",
                 "os-optics project update-catalog PROJECT --out PROJECT", "--include-changed", "run the Project again",
                 'the same command applies to an installation under the earlier name opensci-engine',
                 "can no longer be obtained or read is moved and listed as unverified",
                 "if the status is NEEDS_ATTENTION, add --include-changed"):
        assert step in error.hint, step
    # (the flag is the step of a NEEDS_ATTENTION report, not of every unverified component: os_optics.local.catalog_update)
    assert "when it lists components as unverified" not in error.hint and "until you add --include-changed" not in error.hint
    assert derive_recovery(422, "ENGINE_VERSION_NOT_ACCEPTED", {}) is Recovery.UPGRADE_ENGINE_AND_RERUN


def test_the_catalog_and_supply_recoveries_are_known_to_the_client() -> None:
    """The server's catalog / supply refusals keep their own recovery instead of degrading to CONTACT_OPENSCI."""

    from os_optics.client.errors import RECOVERY_HINTS, Recovery

    for name in ("UPDATE_CATALOG_AND_RERUN", "CHOOSE_ANOTHER_PRODUCT", "RESUBMIT_DESIGN"):
        assert Recovery(name) in RECOVERY_HINTS


@pytest.mark.parametrize(("scope", "who"), [("MEMBER", "of this Account member are"),
                                            ("ORGANIZATION", "of this Account member's organization are")])
def test_paused_artifact_uploads_name_whose_uploads_and_for_how_long(scope, who):
    """``ARTIFACT_UPLOAD_SUSPENDED`` (429, about 24 h) used to carry the generic RETRY_LATER hint (busy, or a
    rate limit was reached); its hint says whose artifact uploads are paused and for how long, from the service's
    details (else the Retry-After wait)."""

    from os_optics.client.errors import error_from_response

    body = json.dumps({"error": {"code": "ARTIFACT_UPLOAD_SUSPENDED", "family": "RATE_LIMITED", "message": "paused",
                                 "recovery": "RETRY_LATER", "details": {"scope": scope, "retry_after_seconds": 86384}}})
    error = error_from_response(429, body.encode("utf-8"), "req-9")
    assert error.recovery is Recovery.RETRY_LATER
    assert error.hint == (f"Artifact uploads {who} paused for 24.0 h (retry_after_seconds 86384); upload again after "
                          "that. Requests that upload no artifact are not affected.")
    assert "busy" not in error.hint and error.as_dict()["hint"] == error.hint
    bare = error_from_response(429, b'{"error": {"code": "ARTIFACT_UPLOAD_SUSPENDED", "recovery": "RETRY_LATER"}}', "r")
    bare.retry_after_seconds = 600.0
    assert "paused for 10 min (retry_after_seconds 600)" in bare.hint
    bare.retry_after_seconds = None
    assert "until the time the service gives (Retry-After)" in bare.hint
