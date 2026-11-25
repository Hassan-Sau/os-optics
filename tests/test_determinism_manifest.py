"""Determinism, canonical hashing, manifest completeness and serialization round trips."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from os_optics import NumericalConfig, ValidationRequest, ValidationResult, validate, validate_json
from os_optics._version import ENGINE_VERSION, MANIFEST_VERSION, NUMERICAL_CONFIG_VERSION, SCHEMA_VERSION
from os_optics.manifest import engine_build_id, hash_quantized

from generate_fixtures import REQ_DIR

STABLE = ["interference_405_500", "keplerian_expander", "hwp_pbs", "control_recoverability", "multi_laser"]


def _load(name) -> str:
    return (REQ_DIR / f"{name}.json").read_text(encoding="utf-8")


def _strip_runtime(text: str) -> dict:
    doc = json.loads(text)
    doc["manifest"].pop("runtime", None)
    return doc


@pytest.mark.parametrize("name", STABLE)
def test_repeated_calls_are_identical(name):
    first = validate_json(_load(name))
    for _ in range(3):
        assert validate_json(_load(name)) == first  # byte-identical: no clocks, no global state, no ordering noise


@pytest.mark.parametrize("name", ["interference_405_500", "control_recoverability"])
def test_results_do_not_depend_on_python_hash_randomization(name):
    """Fresh interpreter processes with different PYTHONHASHSEED values produce identical output."""
    outputs = []
    for seed in ("0", "1", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run([sys.executable, "-m", "os_optics", "validate", str(REQ_DIR / f"{name}.json")],
                              capture_output=True, text=True, env=env, timeout=300)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    assert len(set(outputs)) == 1
    assert outputs[0].strip() == validate_json(_load(name)).strip()  # and equal to the in-process result


def test_input_hash_is_canonical_key_order_and_number_format_independent():
    text = _load("keplerian_expander")
    doc = json.loads(text)

    def reorder(x):
        if isinstance(x, dict):
            return {k: reorder(x[k]) for k in reversed(list(x))}
        if isinstance(x, list):
            return [reorder(v) for v in x]
        return x

    permuted = json.dumps(reorder(doc))
    # 20 -> 20.0 style differences must not matter either
    permuted = permuted.replace('"power_W": 0.001', '"power_W": 1e-3')
    a, b = _strip_runtime(validate_json(text)), _strip_runtime(validate_json(permuted))
    assert a["manifest"]["input_hash"] == b["manifest"]["input_hash"] and a == b


def test_changing_any_input_changes_the_input_hash_and_numerical_settings_are_recorded():
    request = ValidationRequest.model_validate_json(_load("keplerian_expander"))
    base = validate(request)
    changed_project = request.project.model_copy(update={"name": "renamed"})
    other = validate(request.model_copy(update={"project": changed_project}))
    assert other.manifest.input_hash != base.manifest.input_hash and other.manifest.project_hash != base.manifest.project_hash
    assert other.manifest.design_projection_hash == base.manifest.design_projection_hash  # name is not physics
    cfg = NumericalConfig(beam_envelope_k=3.0)
    tweaked = validate(request.model_copy(update={"configuration": request.configuration.model_copy(update={"numerical": cfg})}))
    assert tweaked.manifest.input_hash != base.manifest.input_hash
    assert tweaked.manifest.numerical_tolerances["beam_envelope_k"] == 3.0 and base.manifest.numerical_tolerances["beam_envelope_k"] == 2.0


def test_manifest_is_complete_and_versioned():
    res = validate(ValidationRequest.model_validate_json(_load("control_recoverability")))
    m = res.manifest
    # the module's constants, not their values: what must hold is that they reach every manifest (their form: below)
    assert m.engine_version == ENGINE_VERSION and m.engine_build_id.startswith(ENGINE_VERSION + "+") and m.schema_version == SCHEMA_VERSION
    assert m.numerical_config_version == NUMERICAL_CONFIG_VERSION and m.manifest_version == MANIFEST_VERSION and m.project_schema_version == SCHEMA_VERSION
    for h in (m.input_hash, m.project_hash, m.design_projection_hash, m.result_hash, m.trace_hash):
        assert len(h) == 64 and int(h, 16) >= 0
    assert m.random_seed == 2026 and m.prng_id == "PCG64+BoxMuller-v1"
    assert set(m.model_versions) >= {"ray_geometry", "gaussian_q", "polarization_jones", "interference_two_beam", "control_lm"}
    assert m.numerical_tolerances["ray_min_advance_mm"] == 1e-7 and "hash_float_sig_digits" in m.numerical_tolerances
    assert any(s.startswith("gaussian_beam") for s in m.model_scope) and "unsupported:fdtd" in m.model_scope
    assert m.runtime["python"].startswith("3.")


def test_version_constants_keep_their_form():
    """The form a version may take. An Engine version is a PEP 440 release or pre-release; the schema and manifest axes are
    MAJOR.MINOR.PATCH, because ``project_schema_version`` is judged by its major part; a build id is the version, ``+``, 16 hex
    digits. The release relations between ENGINE_VERSION and SCHEMA_VERSION are in test_hygiene.py."""
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc|\.dev)\d+)?", ENGINE_VERSION)
    for version in (SCHEMA_VERSION, MANIFEST_VERSION, NUMERICAL_CONFIG_VERSION):
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), version
    assert re.fullmatch(re.escape(ENGINE_VERSION) + r"\+[0-9a-f]{16}", engine_build_id())


def test_no_seed_means_no_seed_in_manifest():
    res = validate(ValidationRequest.model_validate_json(_load("interference_405_500")))
    assert res.manifest.random_seed is None and res.manifest.prng_id is None


def test_warnings_are_listed_in_the_manifest():
    res = validate(ValidationRequest.model_validate_json(_load("aperture_clipping")))
    assert any(w.startswith("GAUSSIAN_TRUNCATION") for w in res.manifest.warnings)


def test_result_hash_is_an_integrity_aid_over_the_quantized_result_body():
    res = validate(ValidationRequest.model_validate_json(_load("hwp_pbs")))
    cfg = NumericalConfig()
    body = res.model_copy(update={"manifest": None}).model_dump(mode="json")
    assert res.manifest.result_hash == hash_quantized(body, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor)
    body["observations"][0]["total_power_W"] *= 1.0001  # any semantic change alters the hash
    assert hash_quantized(body, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor) != res.manifest.result_hash
    body2 = res.model_copy(update={"manifest": None}).model_dump(mode="json")
    body2["observations"][0]["total_power_W"] += 1e-15  # below the quantization: not a change
    assert hash_quantized(body2, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor) == res.manifest.result_hash


@pytest.mark.parametrize("name", ["interference_405_500", "control_recoverability", "layout_example", "out_of_scope_targets"])
def test_serialization_round_trip(name):
    res = validate(ValidationRequest.model_validate_json(_load(name)))
    text = res.model_dump_json()
    again = ValidationResult.model_validate_json(text)
    assert again == res and again.model_dump_json() == text
    request = ValidationRequest.model_validate_json(_load(name))
    assert ValidationRequest.model_validate_json(request.model_dump_json()) == request


def test_results_are_finite_json_with_only_documented_statuses():
    text = validate_json(_load("benchmark_standard"))
    assert "NaN" not in text and "Infinity" not in text
    doc = json.loads(text, parse_constant=lambda c: pytest.fail(c))
    assert doc["status"] in {"PASS", "WARNING", "FAIL", "UNKNOWN", "NOT_APPLICABLE", "OUT_OF_MODEL_SCOPE"}


def test_empty_project_is_not_applicable_rather_than_pass():
    res = validate(ValidationRequest(project=ValidationRequest.model_validate_json(_load("keplerian_expander")).project.model_copy(
        update={"components": (), "observations": (), "targets": ()})))
    assert res.status.value == "NOT_APPLICABLE"
