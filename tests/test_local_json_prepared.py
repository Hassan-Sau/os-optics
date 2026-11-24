"""Prepared evaluation handles of the JSON API (``prepare_evaluation`` / ``evaluate_prepared`` / ``cancel_evaluation`` /
``release_evaluation`` / ``evaluation_handles``; ``os_optics.local.prepared``): a reply through a handle equals the
reply of a fresh ``evaluate_project`` for the same document and selection, the handles are bounded with least recently
used eviction reported as a notice, a handle the runtime does not hold is refused with the reason, and cancellation acts on
one handle. The run monitors of the cancellation tests are the host's own."""

from __future__ import annotations

import copy
import json
import threading

import pytest

from os_optics.local import json_api
from os_optics.local.evaluate import EvaluationSession
from os_optics.local.prepared import (
    DEFAULT_MAX_HANDLES,
    EVALUATION_HANDLE_BUSY,
    EVALUATION_HANDLE_EVICTED,
    EVALUATION_HANDLE_LIMIT,
    EVALUATION_HANDLE_UNKNOWN,
    MAX_HANDLES_LIMIT,
    PREPARED_EVALUATIONS,
)
from os_optics.run_control import monitored
from os_optics.run_monitor import RunMonitor

#: What a reply through a handle adds to the reply of ``evaluate_project``.
HANDLE_KEYS = {"handle", "reuse", "notices"}


@pytest.fixture(autouse=True)
def _no_handles():
    PREPARED_EVALUATIONS.clear()
    yield
    PREPARED_EVALUATIONS.clear()


def _project(kind: str | None = None) -> dict:
    return json_api.example_project_document(kind=kind)["project"]


def _same(prepared: dict, fresh: dict) -> None:
    """The reply through a handle is the fresh reply, value for value (JSON text equal: every float as written), plus the
    handle's own keys."""

    assert prepared["ok"] is True and fresh["ok"] is True, (prepared, fresh)
    assert set(prepared) - set(fresh) == HANDLE_KEYS
    rest = {k: v for k, v in prepared.items() if k not in HANDLE_KEYS}
    assert json.dumps(rest, sort_keys=True, allow_nan=False) == json.dumps(fresh, sort_keys=True, allow_nan=False)
    assert prepared["identity"]["result_hash"] == fresh["identity"]["result_hash"] is not None


def _moved(document: dict, offset_mm: float) -> dict:
    candidate = copy.deepcopy(document)
    lens = next(c for c in candidate["components"] if c["optical_role"] == "lens")
    lens["nominal_pose"]["position_mm"][0] += offset_mm
    return candidate


@pytest.mark.parametrize("kind", ["focus", "two_beam_interference", "dichroic_routing"])
@pytest.mark.parametrize("outputs", [None, ["readings", "requirements"], ["complete"]])
def test_a_reply_through_a_handle_equals_a_fresh_evaluation(kind, outputs):
    document = _project(kind)
    prepared = json_api.prepare_evaluation(document)
    assert prepared["ok"] is True and prepared["handle"].startswith("evh_") and prepared["notices"] == []
    fresh = json_api.evaluate_project(document, outputs)
    assert prepared["project_hash"] == fresh["identity"]["project_hash"]
    assert prepared["engine_input_hash"] == fresh["identity"]["engine_input_hash"]
    first = json_api.evaluate_prepared(prepared["handle"], outputs)
    _same(first, fresh)
    again = json_api.evaluate_prepared(prepared["handle"], outputs)
    _same(again, fresh)
    # (the second call read the Engine's products the first one left: the beam trace is not made again)
    assert "beam_trace" in first["reuse"]["executed"] and "beam_trace" in again["reuse"]["reused"]
    assert "beam_trace" not in again["reuse"]["executed"]


def test_a_candidate_loop_through_one_handle_equals_fresh_evaluations():
    """Moved lenses, a changed threshold and a return to an earlier candidate, each against its own fresh evaluation."""

    document = _project()
    handle = json_api.prepare_evaluation(document)["handle"]
    loosened = copy.deepcopy(document)
    loosened["targets"][1]["maximum"] = 0.5
    candidates = [_moved(document, 0.4), _moved(document, -1.25), loosened, document, _moved(document, 0.4)]
    for candidate in candidates:
        outputs = ["readings", "requirements"]
        _same(json_api.evaluate_prepared(handle, outputs, document=candidate), json_api.evaluate_project(candidate, outputs))
    # (the last candidate is the handle's current Project: a call without a document evaluates it)
    _same(json_api.evaluate_prepared(handle, ["readings"]), json_api.evaluate_project(candidates[-1], ["readings"]))
    (row,) = json_api.evaluation_handles()["handles"]
    assert row["evaluations"] == len(candidates) + 1 and row["evaluating"] is False


def test_the_scene_of_a_view_through_a_handle_equals_the_fresh_one():
    """``view`` through a handle: the same ``optical_scene`` as ``evaluate_project``, bound to the same result; the view
    settings draw what they draw there, and invalid ones are refused as there, the handle staying usable."""

    document = _project()
    handle = json_api.prepare_evaluation(document)["handle"]
    for outputs, settings in ((["view"], None), (["view"], {"pupil_ray_selection": "ALL"}), (["readings", "view"], None)):
        fresh = json_api.evaluate_project(document, outputs, None, None, settings)
        prepared = json_api.evaluate_prepared(handle, outputs, None, None, settings)
        _same(prepared, fresh)
        assert prepared["optical_scene"]["engine_run"]["result_hash"]["digest"] == prepared["identity"]["result_hash"]
    # (a candidate's scene is the candidate's)
    moved = _moved(document, 0.4)
    _same(json_api.evaluate_prepared(handle, ["view"], document=moved), json_api.evaluate_project(moved, ["view"]))
    refused = json_api.evaluate_prepared(handle, ["view"], None, None, {"pupil_ray_selection": "SOME"})
    assert refused["error"] == json_api.evaluate_project(document, ["view"], None, None, {"pupil_ray_selection": "SOME"})["error"]
    assert refused["handle"] == handle and "optical_scene" not in json_api.evaluate_prepared(handle, ["readings"])


#: Along +x (the optical axis of a component's local frame).
Q_ALONG_X = [0.5, 0.5, 0.5, 0.5]

#: A thin lens whose declared band (400-1100 nm) does not cover its 1200 nm source, two targets on one plane.
OUT_OF_BAND = {
    "schema": "opensci.project", "project_schema_version": "2.0",
    "components": [
        {"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Source"},
         "nominal_pose": {"position_mm": [-100, 0.15, -0.05], "quaternion_wxyz": Q_ALONG_X},
         "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 1200, "power_w": 0.001, "waist_radius_mm": 0.5,
                          "waist_position_mm": 0, "m2": 1, "polarization": {"kind": "LINEAR", "angle_deg": 0}}},
        {"instance_id": "LENS", "optical_role": "lens", "custom_part": {"custom_part_key": "lens", "description": "Thin lens"},
         "optical_model_id": "thin", "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": Q_ALONG_X}},
    ],
    "observations": [{"observation_id": "early", "pose": {"position_mm": [50, 0, 0], "quaternion_wxyz": Q_ALONG_X},
                      "aperture": {"shape": "circle", "radius_mm": 5}}],
    "targets": [
        {"target_id": "spot", "metric": "spot_rms_mm", "observation_id": "early", "maximum": 1, "unit": "mm"},
        {"target_id": "radius", "metric": "beam_radius_mm", "observation_id": "early", "maximum": 5, "unit": "mm"},
    ],
    "optical_models": [{"model_id": "thin", "model": {
        "model_kind": "IDEAL_THIN_LENS", "model_version": "1", "focal_length_mm": "100", "clear_radius_mm": "10",
        "edge_radius_mm": "12.7", "valid_band_nm": {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"},
        "status": "SPECIFIED"}}],
    "compute_budget": {"max_paths": 128, "max_trace_steps": 2048, "max_interactions_per_path": 64},
}


@pytest.mark.parametrize("outputs, targets", [(["readings", "requirements"], None), (["readings"], ["spot"]),
                                              (["readings"], None), (["complete"], None)])
def test_a_reply_through_a_handle_carries_the_fresh_model_scope_marks_and_coverage(outputs, targets):
    """Through a handle, the targets read through a lens outside its declared band carry the same ESTIMATE basis, validity
    entry and issue code as the fresh reply, and the coverage (``requirements.not_assessed``, ``status``) is the fresh one,
    for a full, a partial and an empty target selection."""

    handle = json_api.prepare_evaluation(OUT_OF_BAND)["handle"]
    fresh = json_api.evaluate_project(OUT_OF_BAND, outputs, None, targets)
    prepared = json_api.evaluate_prepared(handle, outputs, None, targets)
    _same(prepared, fresh)
    assessed = {t["target_id"] for t in prepared["targets"]}
    assert assessed | {row["target_id"] for row in prepared["requirements"]["not_assessed"]} == {"spot", "radius"}
    for row in prepared["targets"]:
        assert "MODEL_WAVELENGTH_OUTSIDE_BAND" in row["issue_codes"] and row["assessment"] != "MET"
        assert row["value_basis"]["value"]["kind"] == "ESTIMATE"
    if not assessed:
        # (nothing assessed never reads PASS; here the model-scope finding names the status)
        assert prepared["status"] == "OUT_OF_MODEL_SCOPE"


def test_targets_exclude_and_plan_refusals_through_a_handle():
    document = _project()
    handle = json_api.prepare_evaluation(document)["handle"]
    _same(json_api.evaluate_prepared(handle, None, None, ["spot_radius"]), json_api.evaluate_project(document, None, None, ["spot_radius"]))
    refused = json_api.evaluate_prepared(handle, ["no_such_capability"])
    assert refused["ok"] is False and refused["error"]["code"] == "UNKNOWN_CAPABILITY" and refused["handle"] == handle
    assert refused["error"] == json_api.evaluate_project(document, ["no_such_capability"])["error"]
    refused = json_api.evaluate_prepared(handle, ["record"], ["model_scope"])
    assert refused["error"]["code"] == "EXCLUDED_CAPABILITY_REQUIRED" and refused["handle"] == handle
    refused = json_api.evaluate_prepared(handle, [3])
    assert refused["error"]["code"] == "INPUT_INVALID"
    # (a malformed selection is refused like every other selection refusal: with the handle and its notices)
    assert refused["handle"] == handle and isinstance(refused["notices"], list)
    for malformed in ({"exclude": 5}, {"targets": [None]}):
        refused = json_api.evaluate_prepared(handle, **malformed)
        assert refused["error"]["code"] == "INPUT_INVALID" and refused["handle"] == handle and isinstance(refused["notices"], list)
    # (a refused call leaves the handle usable)
    _same(json_api.evaluate_prepared(handle), json_api.evaluate_project(document))
    # a released handle is refused as such (prepare again), whatever else the call holds
    json_api.release_evaluation(handle)
    unknown = json_api.evaluate_prepared(handle, "readings", None, 5)
    assert unknown["ok"] is False and unknown["error"]["code"] == "EVALUATION_HANDLE_UNKNOWN"
    assert unknown["error"]["problems"][0]["context"]["reason"] == "RELEASED"


def test_a_project_that_cannot_be_simulated_makes_no_handle_and_a_bad_candidate_leaves_the_handle_as_it_was():
    broken = _project()
    broken["components"] = [c for c in broken["components"] if c["optical_role"] != "source"]
    refused = json_api.prepare_evaluation(broken)
    assert refused["ok"] is False and refused["error"] == json_api.evaluate_project(broken)["error"]
    assert refused["error"]["code"] == "LOCAL_RUN_ERROR" and json_api.evaluation_handles()["handles"] == []
    document = _project()
    handle = json_api.prepare_evaluation(document)["handle"]
    refused = json_api.evaluate_prepared(handle, document=broken)
    assert refused["error"]["code"] == "LOCAL_RUN_ERROR" and refused["handle"] == handle
    assert json_api.evaluate_prepared(handle, document={"schema": "not a project"})["error"]["code"] == "LOCAL_RUN_ERROR"
    # (the handle's current Project is still the one it was prepared with)
    _same(json_api.evaluate_prepared(handle), json_api.evaluate_project(document))


def test_the_handles_are_bounded_and_the_least_recently_used_is_evicted_with_a_notice():
    assert json_api.evaluation_handles()["max_handles"] == DEFAULT_MAX_HANDLES
    assert json_api.evaluation_handles(2)["max_handles"] == 2
    document = _project()
    first, second = (json_api.prepare_evaluation(document)["handle"] for _ in range(2))
    assert json_api.evaluate_prepared(first)["ok"]  # (the first handle is now the most recently used)
    third = json_api.prepare_evaluation(document)
    (notice,) = third["notices"]
    assert notice["code"] == EVALUATION_HANDLE_EVICTED and notice["handle"] == second and notice["max_handles"] == 2
    assert third["handles"] == 2
    assert [row["handle"] for row in json_api.evaluation_handles()["handles"]] == [first, third["handle"]]
    refused = json_api.evaluate_prepared(second)
    assert refused["ok"] is False and refused["error"]["code"] == EVALUATION_HANDLE_UNKNOWN
    assert refused["error"]["problems"][0]["context"] == {"handle": second, "reason": "EVICTED"}
    assert "prepare_evaluation(document)" in refused["error"]["message"]
    # (a lower bound evicts the least recently used idle handles at once, each named)
    lowered = json_api.evaluation_handles(1)
    assert [n["handle"] for n in lowered["notices"]] == [first] and [row["handle"] for row in lowered["handles"]] == [third["handle"]]


@pytest.mark.parametrize("value", [0, MAX_HANDLES_LIMIT + 1, True, "2", 2.0, -1])
def test_a_bound_outside_the_range_is_refused(value):
    refused = json_api.evaluation_handles(value)
    assert refused["ok"] is False and refused["error"]["code"] == "INPUT_INVALID"
    assert json_api.evaluation_handles()["max_handles"] == DEFAULT_MAX_HANDLES


def test_a_released_or_unknown_handle_is_refused_with_its_reason():
    handle = json_api.prepare_evaluation(_project())["handle"]
    assert json_api.release_evaluation(handle) == {"ok": True, "handle": handle, "released": True, "cancelled": False}
    for call in (lambda: json_api.evaluate_prepared(handle), lambda: json_api.release_evaluation(handle),
                 lambda: json_api.cancel_evaluation(handle)):
        refused = call()
        assert refused["error"]["code"] == EVALUATION_HANDLE_UNKNOWN
        assert refused["error"]["problems"][0]["context"] == {"handle": handle, "reason": "RELEASED"}
    for unknown in ("evh_000000000000000000000000", None, 7):
        refused = json_api.evaluate_prepared(unknown)
        assert refused["error"]["code"] == EVALUATION_HANDLE_UNKNOWN
        assert refused["error"]["problems"][0]["context"]["reason"] == "NOT_HELD"


def test_a_prepare_is_refused_when_every_handle_at_the_bound_is_evaluating():
    json_api.evaluation_handles(1)
    handle = json_api.prepare_evaluation(_project())["handle"]
    entry = PREPARED_EVALUATIONS.acquire(handle, None)  # (as an evaluation in progress holds it)
    try:
        refused = json_api.prepare_evaluation(_project())
        assert refused["ok"] is False and refused["error"]["code"] == EVALUATION_HANDLE_LIMIT
        assert json_api.evaluate_prepared(handle)["error"]["code"] == EVALUATION_HANDLE_BUSY
        # (a lower bound never evicts a handle while it evaluates)
        json_api.evaluation_handles(1)
        assert [row["evaluating"] for row in json_api.evaluation_handles()["handles"]] == [True]
    finally:
        PREPARED_EVALUATIONS.finish(entry, evaluated=False)
    assert json_api.prepare_evaluation(_project())["ok"] is True


def test_an_evaluation_that_ends_beyond_a_lowered_bound_evicts_the_least_recently_used_idle_handle():
    """Handles A and B evaluating (A the least recently used) when the bound is lowered to 1: neither is evicted while it
    evaluates. When A's evaluation ends, A is the least recently used idle handle beyond the bound and is evicted at once,
    named in the notices of that evaluation's end; a prepare while B evaluates is refused, since the bound holds. When B
    ends, the bound is met; the next prepare evicts B, the least recently used, and keeps the handle it returns, which
    evaluates. No listing holds an idle handle beyond the bound."""

    def bounded() -> list[str]:
        listing = json_api.evaluation_handles()
        rows = listing["handles"]
        assert len(rows) <= listing["max_handles"] or all(row["evaluating"] for row in rows), rows
        return [row["handle"] for row in rows]

    document = _project()
    first, second = (json_api.prepare_evaluation(document)["handle"] for _ in range(2))
    a, b = PREPARED_EVALUATIONS.acquire(first, None), PREPARED_EVALUATIONS.acquire(second, None)  # (two evaluations in progress)
    assert json_api.evaluation_handles(1)["notices"] == []
    assert bounded() == [first, second]
    ended = PREPARED_EVALUATIONS.finish(a, evaluated=True)
    assert [(n["code"], n["handle"], n["max_handles"]) for n in ended] == [(EVALUATION_HANDLE_EVICTED, first, 1)]
    assert bounded() == [second]
    assert json_api.prepare_evaluation(document)["error"]["code"] == EVALUATION_HANDLE_LIMIT
    assert PREPARED_EVALUATIONS.finish(b, evaluated=True) == []
    assert bounded() == [second]
    third = json_api.prepare_evaluation(document)
    assert third["ok"] is True and [n["handle"] for n in third["notices"]] == [second]
    assert bounded() == [third["handle"]]
    assert json_api.evaluate_prepared(third["handle"])["ok"] is True
    assert bounded() == [third["handle"]]
    refused = json_api.evaluate_prepared(first)
    assert refused["error"]["problems"][0]["context"] == {"handle": first, "reason": "EVICTED"}


def test_cancellation_stops_one_handle_and_the_handle_stays_usable():
    """Two handles evaluated from two threads: while one waits inside its trace, the other evaluates, a second evaluation
    of the waiting handle is refused as busy, and cancelling the idle one stops nothing; cancelling the waiting one stops
    it alone. The stopped handle then evaluates to the fresh reply."""

    document = _project()
    waiting, other = (json_api.prepare_evaluation(document)["handle"] for _ in range(2))
    in_trace, go = threading.Event(), threading.Event()
    replies: dict[str, dict] = {}

    def hold(event: dict) -> None:
        if event.get("phase") == "trace" and not in_trace.is_set():
            in_trace.set()
            go.wait(timeout=60)

    def evaluate() -> None:
        with monitored(RunMonitor(hold)):
            replies["waiting"] = json_api.evaluate_prepared(waiting, ["readings"])

    thread = threading.Thread(target=evaluate)
    thread.start()
    try:
        assert in_trace.wait(timeout=60)
        busy = json_api.evaluate_prepared(waiting)
        assert busy["error"]["code"] == EVALUATION_HANDLE_BUSY and busy["error"]["problems"][0]["context"] == {"handle": waiting}
        _same(json_api.evaluate_prepared(other, ["readings"]), json_api.evaluate_project(document, ["readings"]))
        assert json_api.cancel_evaluation(other) == {"ok": True, "handle": other, "cancelled": False}
        assert json_api.cancel_evaluation(waiting) == {"ok": True, "handle": waiting, "cancelled": True}
    finally:
        go.set()
        thread.join(timeout=120)
    stopped = replies["waiting"]
    assert stopped["ok"] is False and stopped["error"]["code"] == "RUN_STOPPED" and stopped["handle"] == waiting
    assert stopped["stopped"]["stop"]["cause"] == "CALLER" and stopped["stopped"]["stop"]["phase"] == "trace"
    _same(json_api.evaluate_prepared(waiting, ["readings"]), json_api.evaluate_project(document, ["readings"]))
    (row,) = [r for r in json_api.evaluation_handles()["handles"] if r["handle"] == waiting]
    assert row["evaluations"] == 1 and row["evaluating"] is False


def test_an_unmonitored_evaluate_prepared_is_stopped_by_cancel_evaluation(monkeypatch):
    """``evaluate_prepared`` called with no host monitor (as the Designer worker and a plain JSON host call it) runs under
    a monitor of its own: ``cancel_evaluation`` from another thread reaches it, the call answers ``RUN_STOPPED`` with cause
    ``CALLER``, and the handle then evaluates to the fresh reply."""

    document = _project()
    handle = json_api.prepare_evaluation(document)["handle"]
    inside, go = threading.Event(), threading.Event()
    original = EvaluationSession.evaluation

    def held(self, *args, **kwargs):
        if not inside.is_set():
            inside.set()
            go.wait(timeout=60)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EvaluationSession, "evaluation", held)
    replies: dict[str, dict] = {}

    def evaluate() -> None:
        replies["held"] = json_api.evaluate_prepared(handle, ["readings"])

    thread = threading.Thread(target=evaluate)
    thread.start()
    try:
        assert inside.wait(timeout=60)
        assert json_api.cancel_evaluation(handle) == {"ok": True, "handle": handle, "cancelled": True}
    finally:
        go.set()
        thread.join(timeout=120)
    assert not thread.is_alive()
    stopped = replies["held"]
    assert stopped["ok"] is False and stopped["error"]["code"] == "RUN_STOPPED" and stopped["handle"] == handle
    assert stopped["stopped"]["stop"]["cause"] == "CALLER"
    _same(json_api.evaluate_prepared(handle, ["readings"]), json_api.evaluate_project(document, ["readings"]))
    (row,) = json_api.evaluation_handles()["handles"]
    assert row["handle"] == handle and row["evaluations"] == 1 and row["evaluating"] is False


def test_an_unmonitored_evaluation_runs_under_a_monitor_of_its_own_that_cancel_reaches():
    handle = json_api.prepare_evaluation(_project())["handle"]
    monitor = RunMonitor(preflight=False)
    entry = PREPARED_EVALUATIONS.acquire(handle, monitor)
    try:
        assert json_api.cancel_evaluation(handle)["cancelled"] is True and monitor.cancelled
    finally:
        PREPARED_EVALUATIONS.finish(entry, evaluated=False)
    # (a release while evaluating stops the evaluation too)
    monitor = RunMonitor(preflight=False)
    entry = PREPARED_EVALUATIONS.acquire(handle, monitor)
    assert json_api.release_evaluation(handle)["cancelled"] is True and monitor.cancelled
    assert PREPARED_EVALUATIONS.finish(entry, evaluated=False) == []
    assert json_api.evaluation_handles()["handles"] == []


def test_a_monitored_evaluation_reports_its_progress_and_a_stop_while_preparing_makes_no_handle():
    document = _project()
    handle = json_api.prepare_evaluation(document)["handle"]
    with monitored(RunMonitor()):
        reply = json_api.evaluate_prepared(handle)
    assert reply["ok"] and "trace" in reply["progress"]["completed_phases"]
    fresh = json_api.evaluate_project(document)
    assert {k: v for k, v in reply.items() if k not in HANDLE_KEYS | {"progress"}} == fresh
    monitor = RunMonitor(lambda event: monitor.cancel("stop") if event.get("phase") == "lower" else None)
    with monitored(monitor):
        stopped = json_api.prepare_evaluation(document)
    assert stopped["ok"] is False and stopped["error"]["code"] == "RUN_STOPPED" and stopped["stopped"]["stop"]["phase"] == "lower"
    assert [row["handle"] for row in json_api.evaluation_handles()["handles"]] == [handle]


def test_the_handle_calls_are_registered_and_discoverable():
    caps = json_api.capabilities()
    methods = caps["json_api_methods"]
    for name in json_api.PREPARED_METHODS:
        assert methods[name]["category"] == "run" and "runtime_state" not in methods[name]
    assert methods["prepare_evaluation"]["required"] == ["document"]
    assert methods["evaluate_prepared"]["parameters"] == ["handle", "outputs", "exclude", "targets", "view_settings", "document"]
    discovery = caps["evaluation"]["prepared"]
    assert discovery["calls"] == [json_api.API_METHODS[name].signature() for name in json_api.PREPARED_METHODS]
    assert discovery["default_max_handles"] == DEFAULT_MAX_HANDLES and discovery["max_handles_limit"] == MAX_HANDLES_LIMIT
    assert discovery["error_codes"] == [EVALUATION_HANDLE_UNKNOWN, EVALUATION_HANDLE_BUSY, EVALUATION_HANDLE_LIMIT]
    assert discovery["notice_codes"] == [EVALUATION_HANDLE_EVICTED]
    # (a host that is not a Python program: the same calls through the registry's dispatch, JSON-safe replies)
    document = _project()
    reply, files = json_api.call_method("prepare_evaluation", {"document": document})
    assert reply["ok"] and files is None
    evaluated, files = json_api.call_method("evaluate_prepared", {"handle": reply["handle"], "outputs": ["readings"]})
    assert files is None
    json.dumps(evaluated, allow_nan=False)
    _same(evaluated, json_api.evaluate_project(document, ["readings"]))
    listed, _ = json_api.call_method("evaluation_handles", {})
    assert [row["handle"] for row in listed["handles"]] == [reply["handle"]]
    released, _ = json_api.call_method("release_evaluation", {"handle": reply["handle"]})
    assert released["released"] is True


# ------------------------------------------------------------------------------------------------ the catalog a handle uses


def _changed_release(document: dict, catalog_version: str) -> bytes:
    """The suite's active release as ``catalog_version``, the first surface of the design's lens bent (radius 60 mm)."""

    from catalog_store_helpers import FIXTURE_CATALOG

    catalog = json.loads(FIXTURE_CATALOG.read_text(encoding="utf-8"))
    catalog["catalog_version"] = catalog_version
    sku = next(c["product"]["optical_sku"] for c in document["components"] if c["optical_role"] == "lens")
    product = next(p for p in catalog["products"] if p["optical_sku"] == sku)
    surface = product["simulation_models"][0]["surfaces"][0]
    assert surface["radius_mm"] != "60"
    surface["radius_mm"] = "60"
    return json.dumps(catalog).encode("utf-8")


def _unpinned(document: dict, pin: str | None = None) -> dict:
    document = copy.deepcopy(document)
    for component in document["components"]:
        if component.get("product") is not None:
            component["product"].pop("catalog_version", None)
            if pin is not None:
                component["product"]["catalog_version"] = pin
    return document


@pytest.fixture
def _installed_catalogs():
    from os_optics.local import clear_installed_catalogs

    clear_installed_catalogs()
    yield
    clear_installed_catalogs()


def test_a_handle_of_a_design_that_pins_no_release_follows_the_active_release(_installed_catalogs):
    """A design whose parts name no catalog release runs against the active one: a release installed after the handle was
    prepared is the one the next evaluation through the handle uses, as a fresh evaluate_project does."""

    document = _unpinned(_project("focus"))
    prepared = json_api.prepare_evaluation(document)
    before = json_api.evaluate_prepared(prepared["handle"], ["readings"])
    _same(before, json_api.evaluate_project(document, ["readings"]))
    installed = json_api.install_catalog(_changed_release(document, "2026-01-01.9"))
    assert installed["ok"] is True
    after = json_api.evaluate_prepared(prepared["handle"], ["readings"])
    fresh = json_api.evaluate_project(document, ["readings"])
    _same(after, fresh)
    assert after["identity"]["catalog_version"] == "2026-01-01.9" != before["identity"]["catalog_version"]
    assert after["identity"]["result_hash"] != before["identity"]["result_hash"]
    # (the session holds the release in force, not every release it has used)
    session = PREPARED_EVALUATIONS._held[prepared["handle"]].session
    assert list(session._catalogs) == [("release", "2026-01-01.9", installed["sha256"])]
    assert session.statistics()["catalogs"] == 1


def test_a_handle_evaluates_a_design_against_its_pin_once_that_release_is_installed(_installed_catalogs):
    """A handle given a design whose parts pin a release this runtime does not hold refuses it as evaluate_project does
    (CATALOG_VERSION_MISMATCH: the active release is another); once the pinned release is installed -- and after it another
    active one -- the handle evaluates the design against the release it pins, as a fresh evaluate_project does."""

    from catalog_store_helpers import FIXTURE_CATALOG

    document = _unpinned(_project("focus"))
    pinned = _unpinned(document, pin="2026-01-01.8")
    assert json_api.install_catalog(FIXTURE_CATALOG.read_bytes())["ok"] is True
    prepared = json_api.prepare_evaluation(document)
    refused = json_api.evaluate_prepared(prepared["handle"], ["readings"], document=pinned)
    fresh_refused = json_api.evaluate_project(pinned, ["readings"])
    assert refused["ok"] is False and fresh_refused["ok"] is False
    assert [p["code"] for p in refused["error"]["problems"]] == [p["code"] for p in fresh_refused["error"]["problems"]]
    assert "CATALOG_VERSION_MISMATCH" in [p["code"] for p in refused["error"]["problems"]]
    assert json_api.install_catalog(_changed_release(document, "2026-01-01.8"))["ok"] is True
    assert json_api.install_catalog(FIXTURE_CATALOG.read_bytes())["ok"] is True  # (another release active again)
    after = json_api.evaluate_prepared(prepared["handle"], ["readings"], document=pinned)
    _same(after, json_api.evaluate_project(pinned, ["readings"]))
    assert after["identity"]["catalog_version"] == "2026-01-01.8"
