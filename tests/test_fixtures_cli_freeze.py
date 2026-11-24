"""Versioned fixtures, snapshot regression, CLI, and the FROZEN-contract schema baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from os_optics import SCHEMA_VERSION, ValidationRequest, validate, validate_json
from os_optics.compare import compare_results
from os_optics.manifest import hash_exact

from fixture_scenes import FIXTURES
from generate_fixtures import FREEZE_FILE, REQ_DIR, SNAP_DIR, canonical_request_text, freeze_hashes, snapshot_text
from identity_mask import IDENTITY_BOUND_MANIFEST_HASHES, IDENTITY_MANIFEST_FIELDS, IDENTITY_MASK, mask_request_identity, masked_input_hash, masked_project_hash

NAMES = sorted(FIXTURES)


@pytest.mark.parametrize("name", NAMES)
def test_request_fixture_files_are_current(name):
    """The committed JSON requests are exactly what fixture_scenes.py builds (no stale hand-edited examples)."""
    on_disk = (REQ_DIR / f"{name}.json").read_text(encoding="utf-8")
    assert on_disk == canonical_request_text(FIXTURES[name]()), f"run: python tests/generate_fixtures.py  ({name} is stale)"


@pytest.mark.parametrize("name", NAMES)
def test_a_request_file_takes_the_schema_of_the_engine_reading_it(name):
    """The file stores no stamp and a factory sets none of its own, so a reader stamps the current SCHEMA_VERSION."""
    text = (REQ_DIR / f"{name}.json").read_text(encoding="utf-8")
    assert "project_schema_version" not in json.loads(text)["project"]
    assert ValidationRequest.model_validate_json(text).project.project_schema_version == SCHEMA_VERSION
    assert FIXTURES[name]().project.project_schema_version == SCHEMA_VERSION


def _next_minor(version: str) -> str:
    major, minor, _ = version.split(".")
    return f"{major}.{int(minor) + 1}.0"


@pytest.mark.parametrize("name", NAMES)
def test_manifest_hashes_bind_the_stamped_request(name):
    """The binding the snapshots no longer pin. ``input_hash`` and ``project_hash`` are recomputed from their parts and move with
    the stamp; the masked digests and every hash the snapshots keep comparing do not move with it."""
    request = ValidationRequest.model_validate_json((REQ_DIR / f"{name}.json").read_text(encoding="utf-8"))
    manifest = validate(request).manifest
    assert manifest.input_hash == hash_exact(request.model_dump(mode="json"))
    if manifest.project_hash:  # a request refused before its Project was read has none
        assert manifest.project_hash == hash_exact(request.project.model_dump(mode="json"))
    project = request.project.model_copy(update={"project_schema_version": _next_minor(SCHEMA_VERSION)})
    stamped = request.model_copy(update={"project": project})
    other = validate(stamped).manifest
    assert other.input_hash != manifest.input_hash
    if manifest.project_hash:
        assert other.project_hash != manifest.project_hash
    assert masked_input_hash(stamped) == masked_input_hash(request) and masked_project_hash(stamped) == masked_project_hash(request)
    for content in ("design_projection_hash", "result_hash", "trace_hash"):
        assert getattr(other, content) == getattr(manifest, content), content


def test_the_identity_mask_covers_the_stamp_and_nothing_else():
    request = FIXTURES["keplerian_expander"]()
    masked = mask_request_identity(request)
    assert masked["project"].pop("project_schema_version") == IDENTITY_MASK
    plain = request.model_dump(mode="json")
    plain["project"].pop("project_schema_version")
    assert masked == plain
    renamed = request.model_copy(update={"project": request.project.model_copy(update={"name": "renamed"})})
    assert masked_input_hash(renamed) != masked_input_hash(request) and masked_project_hash(renamed) != masked_project_hash(request)
    configured = request.model_copy(update={"configuration": request.configuration.model_copy(
        update={"numerical": request.configuration.numerical.model_copy(update={"beam_envelope_k": 3.0})})})
    assert masked_input_hash(configured) != masked_input_hash(request)


# The one subtree compared within the cross-runtime tolerance (``compare.py``) instead of exactly: the control optimizer's
# values come from LAPACK/BLAS, whose last bits differ between hosts. Measured: every fixture is bit-identical to its
# snapshot on Windows and on the Linux CI host except ``control_recoverability`` on Linux, which differs in the last bit of
# control.recoverability_cases[0].run.final_state.M/yaw and control.recoverability_cases[1].run.final_state.M/pitch.
TOLERANT_SUBTREE = "control"
# The snapshots are recorded on the Python 3.14 line. Python 3.11 (the CI leg of the oldest supported version) sums floats
# with the built-in ``sum`` uncompensated (3.12 made it compensated), so its results differ in the last bits: measured on
# the Linux CI host, trace_interaction_limit, trace_path_budget and trace_return_chain (power sums). There the whole
# result is compared within the tolerance.
EXACT_RUNTIME = sys.version_info >= (3, 12)


def _exact_differences(fresh, stored, path: str = "") -> list[str]:
    """Every leaf where the parsed JSON values differ (Python equality, types included: no tolerance)."""
    if type(fresh) is not type(stored):
        return [f"{path}: type {type(fresh).__name__} != {type(stored).__name__}"]
    if isinstance(fresh, dict):
        if set(fresh) != set(stored):
            return [f"{path}: keys {sorted(set(fresh) ^ set(stored))}"]
        return [d for k in sorted(fresh) for d in _exact_differences(fresh[k], stored[k], f"{path}.{k}" if path else k)]
    if isinstance(fresh, list):
        if len(fresh) != len(stored):
            return [f"{path}: length {len(fresh)} != {len(stored)}"]
        return [d for i, (a, b) in enumerate(zip(fresh, stored)) for d in _exact_differences(a, b, f"{path}[{i}]")]
    return [] if fresh == stored else [f"{path}: {fresh!r} != {stored!r}"]


@pytest.mark.parametrize("name", NAMES)
def test_result_matches_stored_snapshot_exactly(name):
    """A fresh run reproduces its stored snapshot exactly (every value but the Engine identity, which neither side holds:
    ``identity_mask.py``), so a change meant to keep the numbers cannot drift inside a tolerance unnoticed; only the
    ``control`` subtree is compared within tolerance (and, on Python 3.11, the whole result: ``EXACT_RUNTIME``)."""
    request = ValidationRequest.model_validate_json((REQ_DIR / f"{name}.json").read_text(encoding="utf-8"))
    fresh = json.loads(snapshot_text(request))
    stored = json.loads((SNAP_DIR / f"{name}.json").read_text(encoding="utf-8"))
    assert not set(stored["manifest"]) & set(IDENTITY_MANIFEST_FIELDS + IDENTITY_BOUND_MANIFEST_HASHES)
    if not EXACT_RUNTIME:
        tolerant = compare_results(fresh, stored)
        assert tolerant == [], "\n".join(str(d) for d in tolerant[:10])
        return
    fresh_control, stored_control = fresh.pop(TOLERANT_SUBTREE, None), stored.pop(TOLERANT_SUBTREE, None)
    if fresh_control is not None or stored_control is not None:
        # manifest.result_hash also hashes the control subtree (quantized), so a last-bit platform difference there
        # could flip it at a rounding boundary; every value it summarizes is compared above or below instead.
        for doc in (fresh, stored):
            doc["manifest"].pop("result_hash", None)
    diffs = _exact_differences(fresh, stored)
    assert diffs == [], "\n".join(diffs[:10])
    tolerant = compare_results({TOLERANT_SUBTREE: fresh_control}, {TOLERANT_SUBTREE: stored_control})
    assert tolerant == [], "\n".join(str(d) for d in tolerant[:10])


def test_exact_snapshot_comparison_sees_the_last_bit():
    stored = {"a": [1.0, {"b": 0.1 + 0.2}], "n": 1, "flag": True}
    assert _exact_differences(json.loads(json.dumps(stored)), stored) == []
    assert _exact_differences({**stored, "a": [1.0, {"b": 0.3}]}, stored) == ["a[1].b: 0.3 != 0.30000000000000004"]
    assert _exact_differences({**stored, "n": 1.0}, stored) == ["n: type float != int"]
    assert _exact_differences({**stored, "flag": 1}, stored) == ["flag: type int != bool"]


@pytest.mark.parametrize("name", NAMES)
def test_file_request_validate_json_and_object_api_agree(name):
    text = (REQ_DIR / f"{name}.json").read_text(encoding="utf-8")
    via_json = json.loads(validate_json(text))
    via_object = json.loads(validate(ValidationRequest.model_validate_json(text)).model_dump_json())
    assert via_json == via_object


def test_expected_fixture_outcomes():
    """Each fixture demonstrates one intended behaviour."""
    expected = {
        "interference_405_500": "PASS", "keplerian_expander": "PASS", "mzi_window": "PASS", "hwp_pbs": "PASS",
        "tilted_mirror_astigmatism": "PASS", "aperture_clipping": "WARNING", "multi_laser": "PASS",
        "out_of_scope_targets": "OUT_OF_MODEL_SCOPE", "layout_example": "PASS", "control_recoverability": "PASS",
        "invalid_zero_normal": "FAIL", "benchmark_standard": "PASS",
        # trace regression baselines (budget scheduling, partial beams, return light): test_trace_regression_fixtures.py
        # (trace_partial_beams, trace_return_chain, trace_probe_budget: PASS since findings carried by a quantity are notices
        # and an auxiliary budget leaves the calculation complete)
        "trace_path_budget": "WARNING", "trace_step_budget": "WARNING", "trace_interaction_limit": "UNKNOWN",
        "trace_partial_beams": "PASS", "trace_return_chain": "PASS", "trace_probe_budget": "PASS",
        "trace_fresnel_branch_budget": "WARNING",
        "trace_freeform_surfaces": "OUT_OF_MODEL_SCOPE",
        # The extended-emitter fixture explicitly assumes ideal N-BK7 transmission to isolate source sampling;
        # the birefringent fixture states no loss, so its transmitted-power/coherence quantities remain unknown.
        "trace_grating_orders": "UNKNOWN", "trace_faraday_rotator": "PASS", "trace_extended_emitter": "PASS", "trace_birefringent_crystals": "UNKNOWN",
    }
    assert set(expected) == set(NAMES)
    for name, status in expected.items():
        snap = json.loads((SNAP_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert snap["status"] == status, name


def _snapshot(name: str) -> dict:
    return json.loads((SNAP_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_the_grating_snapshot_holds_the_hand_values():
    """1 mW at 45 deg to the grooves: order 0 (0.6 + 0.2) / 2, order +1 (0.25 + 0.55) / 2, order -1 at most what they leave; behind a grating with no response all are bounds."""

    between, after = _snapshot("trace_grating_orders")["observations"]
    known = sorted(a["beam"]["power_W"] for a in between["arrivals"] if a["beam"]["power_W"] is not None)
    assert known == pytest.approx([4e-4, 4e-4], rel=1e-12)
    assert [a["power_upper_bound_W"] for a in between["arrivals"] if a["beam"]["power_W"] is None] == pytest.approx([2e-4], rel=1e-12)
    assert all(a["beam"]["polarization"] is None for a in between["arrivals"])
    assert sorted(a["power_upper_bound_W"] for a in after["arrivals"]) == pytest.approx([2e-4, 4e-4, 4e-4], rel=1e-12)


def test_the_faraday_snapshot_holds_the_hand_values():
    """1 mW, transmittance 0.9 per pass, a mirror of R = 0.99 behind it: 0.9, 0.891 and 0.8019 mW on the planes."""

    passing, back = _snapshot("trace_faraday_rotator")["observations"]
    assert [a["beam"]["power_W"] for a in passing["arrivals"]] == pytest.approx([9e-4, 8.91e-4], rel=1e-12)
    assert [a["beam"]["power_W"] for a in back["arrivals"]] == pytest.approx([1e-3, 8.019e-4], rel=1e-12)


def test_the_emitter_snapshot_accounts_for_every_watt():
    result = _snapshot("trace_extended_emitter")
    (light,) = result["observations"][0]["sampled_light"]
    assert light["sample_count"] == 256 and light["power_W"] == pytest.approx(result["power_budget"]["total_sinks"]["DETECTED"], rel=1e-12)
    assert sum(result["power_budget"]["total_sinks"].values()) == pytest.approx(1.0, rel=1e-12)


# ---- CLI --------------------------------------------------------------------------------------------------------------------------
def _cli(*args, stdin=None, env=None):
    return subprocess.run([sys.executable, "-m", "os_optics", *args], capture_output=True, text=True, input=stdin, env=env, timeout=120)


def test_cli_validate_exit_codes_and_output(tmp_path):
    ok = _cli("validate", str(REQ_DIR / "keplerian_expander.json"))
    assert ok.returncode == 0 and json.loads(ok.stdout)["status"] == "PASS"
    warn = _cli("validate", str(REQ_DIR / "aperture_clipping.json"))
    assert warn.returncode == 0 and json.loads(warn.stdout)["status"] == "WARNING"
    # the exit code follows the status contract (default requirements); the aggregate stays selectable
    fail = _cli("validate", str(REQ_DIR / "invalid_zero_normal.json"), "--status-contract", "aggregate")
    assert fail.returncode == 1 and json.loads(fail.stdout)["status"] == "FAIL"
    # layout findings are the layout's own outcome (no layout target declared): the aggregate and the default pass
    layout = _cli("validate", str(REQ_DIR / "layout_example.json"), "--status-contract", "aggregate")
    assert layout.returncode == 0 and json.loads(layout.stdout)["layout"]["status"] == "FAIL"
    assert _cli("validate", str(REQ_DIR / "layout_example.json")).returncode == 0
    oos = _cli("validate", str(REQ_DIR / "out_of_scope_targets.json"), "--status-contract", "aggregate")
    assert oos.returncode == 3
    assert _cli("validate", str(REQ_DIR / "out_of_scope_targets.json")).returncode == 2  # (the out-of-scope targets: undetermined)
    assert _cli("validate", str(REQ_DIR / "out_of_scope_targets.json"), "--status-contract", "execution").returncode == 0
    out_file = tmp_path / "r.json"
    piped = _cli("validate", "-", "-o", str(out_file), "--pretty", stdin=(REQ_DIR / "hwp_pbs.json").read_text(encoding="utf-8"))
    assert piped.returncode == 0 and json.loads(out_file.read_text(encoding="utf-8"))["status"] == "PASS"
    assert "\n  " in out_file.read_text(encoding="utf-8")  # pretty printed
    missing = _cli("validate", str(tmp_path / "nope.json"))
    assert missing.returncode == 64 and "cannot read" in missing.stderr
    garbage = _cli("validate", "-", stdin="{not json")
    assert garbage.returncode == 1 and json.loads(garbage.stdout)["issues"][0]["code"] == "INVALID_INPUT"


def test_cli_schema_and_compare(tmp_path):
    for which in ("request", "result"):
        schema = json.loads(_cli("schema", which).stdout)
        assert schema["title"] in ("ValidationRequest", "ValidationResult") and "$defs" in schema
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    result = _cli("validate", str(REQ_DIR / "hwp_pbs.json")).stdout
    a.write_text(result, encoding="utf-8")
    doc = json.loads(result)
    doc["observations"][0]["total_power_W"] *= 1.5
    b.write_text(json.dumps(doc), encoding="utf-8")
    same = _cli("compare", str(a), str(a))
    assert same.returncode == 0 and "equivalent" in same.stdout
    diff = _cli("compare", str(a), str(b))
    assert diff.returncode == 1 and "total_power_W" in diff.stdout
    assert _cli("--version").stdout.startswith("os-optics ")


def test_compare_results_rules():
    base = {"status": "PASS", "x": [1.0, 2.0], "n": 3, "manifest": {"runtime": {"python": "a"}, "result_hash": "aa", "input_hash": "h"}}
    same = {"status": "PASS", "x": [1.0 + 1e-13, 2.0], "n": 3, "manifest": {"runtime": {"python": "b"}, "result_hash": "bb", "input_hash": "h"}}
    assert compare_results(base, same) == []  # tolerance, ignored runtime + result_hash
    bad_status = {**same, "status": "WARNING"}
    assert [d.path for d in compare_results(base, bad_status)] == ["status"]
    bad_float = {**same, "x": [1.0, 2.001]}
    assert [d.kind for d in compare_results(base, bad_float)] == ["float"]
    bad_hash = {**same, "manifest": {**same["manifest"], "input_hash": "other"}}
    assert [d.path for d in compare_results(base, bad_hash)] == ["manifest.input_hash"]
    assert [d.kind for d in compare_results(base, {**same, "x": [1.0]})] == ["length"]
    assert compare_results({"a": 1}, {"b": 1})[0].kind == "keys"
    assert compare_results({"a": True}, {"a": 1})[0].kind == "type"  # bool is not an int here


# ---- FROZEN contract baseline -----------------------------------------------------------------------------------------------------
def test_frozen_contracts_are_unchanged_since_the_recorded_baseline():
    """FROZEN means: field names / semantics change only with a SCHEMA_VERSION bump + migration. If this fails on purpose, bump
    SCHEMA_VERSION in _version.py and run `python tests/generate_fixtures.py --freeze` (or, while the current SCHEMA_VERSION is
    in no published release, `--freeze --unreleased`: one bump per release)."""
    baseline = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    current = freeze_hashes()
    changed = sorted(k for k in current if current[k] != baseline["hashes"].get(k))
    assert baseline["schema_version"] == SCHEMA_VERSION, "SCHEMA_VERSION changed: regenerate the freeze baseline with --freeze"
    assert changed == [], f"frozen contract(s) {changed} changed without a SCHEMA_VERSION bump"
    assert set(current) == set(baseline["hashes"])


def test_unreleased_refuses_to_re_record_a_published_release(tmp_path, monkeypatch):
    """``--freeze --unreleased`` re-records the baseline only within a version and SCHEMA_VERSION past the last published
    release recorded in ``_version.py``; for a published one it refuses and writes nothing."""
    import generate_fixtures
    from os_optics import __version__
    from os_optics._version import LAST_PUBLISHED_SCHEMA_VERSION, LAST_PUBLISHED_VERSION

    def release(version):
        return tuple(int(part) for part in version.split("."))

    # the record never runs ahead of this source tree
    assert release(LAST_PUBLISHED_VERSION) <= release(__version__)
    assert release(LAST_PUBLISHED_SCHEMA_VERSION) <= release(SCHEMA_VERSION)
    problem = generate_fixtures.published_release_problem
    assert "is published" in problem(LAST_PUBLISHED_VERSION, _next_minor(LAST_PUBLISHED_SCHEMA_VERSION))
    assert "published release's" in problem(_next_minor(LAST_PUBLISHED_VERSION), LAST_PUBLISHED_SCHEMA_VERSION)
    assert problem(_next_minor(LAST_PUBLISHED_VERSION), _next_minor(LAST_PUBLISHED_SCHEMA_VERSION)) is None
    # a development or candidate version is refused with the reason, never a crash
    for version, schema in ((_next_minor(LAST_PUBLISHED_VERSION) + ".dev0", _next_minor(LAST_PUBLISHED_SCHEMA_VERSION)),
                            (_next_minor(LAST_PUBLISHED_VERSION) + "rc1", _next_minor(LAST_PUBLISHED_SCHEMA_VERSION)),
                            (_next_minor(LAST_PUBLISHED_VERSION), _next_minor(LAST_PUBLISHED_SCHEMA_VERSION) + ".dev2")):
        assert "is no release number MAJOR.MINOR.PATCH" in problem(version, schema)
    with monkeypatch.context() as dev:
        dev.setattr(generate_fixtures, "__version__", _next_minor(LAST_PUBLISHED_VERSION) + ".dev0", raising=False)
        dev.setattr(generate_fixtures.published_release_problem, "__defaults__",
                    (_next_minor(LAST_PUBLISHED_VERSION) + ".dev0", SCHEMA_VERSION))
        with pytest.raises(SystemExit, match="--unreleased refused: version .*dev0 is no release number"):
            generate_fixtures.write_freeze_baseline(unreleased=True)

    freeze = tmp_path / "contract_freeze.json"
    freeze.write_text(FREEZE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(generate_fixtures, "FREEZE_FILE", freeze)
    for name, value in (("LAST_PUBLISHED_VERSION", __version__), ("LAST_PUBLISHED_SCHEMA_VERSION", SCHEMA_VERSION)):
        with monkeypatch.context() as published:  # (as it reads once this version is published)
            published.setattr(generate_fixtures, name, value)
            with pytest.raises(SystemExit, match="--unreleased refused"):
                generate_fixtures.write_freeze_baseline(unreleased=True)
            assert freeze.read_text(encoding="utf-8") == FREEZE_FILE.read_text(encoding="utf-8")
    # past the last published release: re-recorded within the current SCHEMA_VERSION
    monkeypatch.setattr(generate_fixtures, "LAST_PUBLISHED_VERSION", "0.0.0")
    monkeypatch.setattr(generate_fixtures, "LAST_PUBLISHED_SCHEMA_VERSION", "0.0.0")
    freeze.write_text("{}", encoding="utf-8")
    generate_fixtures.write_freeze_baseline(unreleased=True)
    assert json.loads(freeze.read_text(encoding="utf-8")) == {"schema_version": SCHEMA_VERSION, "hashes": freeze_hashes()}
