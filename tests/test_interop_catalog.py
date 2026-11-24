"""Interop capability catalog (``os_optics.interop``): schema, accurate statuses, existing entry points and
the Python / json_api / CLI surfaces."""

from __future__ import annotations

import importlib
import io
import json
from contextlib import redirect_stdout

import pytest

from os_optics import interop
from os_optics.cli import main as cli_main
from os_optics.interop.model import DIRECTION_TITLES, STATUS_MEANINGS, InteropEntryV1
from os_optics.local import json_api

ENTRIES = interop.interop_catalog()["entries"]
BY_ID = {e["entry_id"]: e for e in ENTRIES}
#: The directions that get executed companion examples: their data contracts exist, the verified records come
#: from the kit repository's (os-optics-kit) case records.
DATA_CONTRACT_ENTRIES = {"optimization-tolerance", "scalar-diffraction-imaging", "quantum-photonics", "metasurface-virtual-element"}


def test_every_direction_of_the_issue_has_an_entry_and_statuses_are_accurate():
    assert {e["direction"] for e in ENTRIES} == set(DIRECTION_TITLES)
    # the directions with executed cases in the public kit repository (os-optics-kit) are VERIFIED_EXAMPLE, each record naming
    # the tool version, the example, its independent reference and the companion commit; every other entry claims none
    verified = {e["entry_id"]: e for e in ENTRIES if e["status"] == "VERIFIED_EXAMPLE"}
    assert set(verified) == DATA_CONTRACT_ENTRIES
    for entry in verified.values():
        assert entry["verified"]
        for record in entry["verified"]:
            assert record["engine_version"] == "0.8.0" and len(record["companion_commit"]) == 40
            assert record["example"].startswith("cases/interop-") and record["reference"]
            assert record["outcome"].startswith("WITHIN_TOLERANCE")
    assert all(e["verified"] == [] for e in ENTRIES if e["entry_id"] not in verified)
    others = {e["entry_id"]: e["status"] for e in ENTRIES if e["entry_id"] not in DATA_CONTRACT_ENTRIES}
    assert others.pop("quantum-photonics-strawberry-fields") == "DEPRECATED"
    assert set(others.values()) == {"GUIDANCE_ONLY"}
    for entry in ENTRIES:
        InteropEntryV1.model_validate(entry)  # the JSON form round-trips through the schema


def test_strawberry_fields_is_archived_and_points_to_the_replacement():
    entry = BY_ID["quantum-photonics-strawberry-fields"]
    (tool,) = entry["tools"]
    assert tool["maintenance"] == "ARCHIVED" and tool["role"] == "HISTORICAL"
    assert entry["replaced_by"] == "quantum-photonics" and entry["replaced_by"] in BY_ID
    assert all(t["maintenance"] != "ARCHIVED" for t in BY_ID["quantum-photonics"]["tools"])


def test_cad_entry_distinguishes_the_official_library_from_the_third_party_wrapper():
    tools = {t["name"]: t for t in BY_ID["cad-brep-geometry"]["tools"]}
    official, wrapper = tools["NVIDIA USD BRep / SMLib"], tools["smlibpy"]
    assert official["maintainer"] == "UPSTREAM_PROJECT" and official["url"] == "https://github.com/NVIDIA-Omniverse/usd-brep"
    assert official["license"] == "Apache-2.0"
    assert wrapper["maintainer"] == "THIRD_PARTY_WRAPPER" and "not an NVIDIA package" in " ".join(wrapper["notes"])
    limits = " ".join(BY_ID["cad-brep-geometry"]["limits"])
    assert "not an optical model" in limits and "not an optical or physical acceptance" in limits


def test_quantum_entry_states_user_inputs_and_the_verified_orientation():
    entry = BY_ID["quantum-photonics"]
    text = json.dumps(entry)
    assert "U[out, in]" in text and "Perceval 1.3.1" in text and "Piquasso 8.0.1" in text
    assert any("never inferred from the classical beam" in a for a in entry["assumptions"])
    assert "CC BY-NC-ND 4.0" in json.dumps(entry["tools"])  # the license fact of Perceval's required dependency


def test_scientific_conditions_are_stated():
    text = json.dumps(ENTRIES)
    for statement in ("Intensity or power alone does not determine a complex field", "Sparse pupil rays are samples of the geometric "
                      "wavefront, not a sampled pupil field", "M^2 > 1 alone does not determine the modal content",
                      "Quantum source and detector properties are user inputs", "never replace the run's hashes or its Local Validation",
                      "A mode-port S-matrix and a sampled free-space field are different interfaces",
                      "One external output", "retained_dofs state the design intent"):
        assert statement in text, statement


def test_tools_carry_https_links_and_licenses_with_their_source_and_date():
    for entry in ENTRIES:
        for tool in entry["tools"]:
            assert tool["url"].startswith("https://")
            if tool["license"] is not None:
                assert tool["license_source"] in ("PyPI metadata", "source repository") and tool["license_checked"] == "2026-10-15"
            if tool["distribution"] == "COMMERCIAL":
                assert tool["maintainer"] == "COMMERCIAL_VENDOR"


def _resolve_python(name: str) -> object:
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[split:]:
            target = getattr(target, attribute)
        return target
    raise AssertionError(name)


def _cli_exists(name: str) -> bool:
    words = name.split()[1:]
    out = io.StringIO()
    with redirect_stdout(out), pytest.raises(SystemExit) as info:
        cli_main([*words, "--help"])
    return info.value.code == 0


def test_every_entry_point_exists():
    seen = set()
    for entry in ENTRIES:
        for point in entry["entry_points"]:
            key = (point["kind"], point["name"])
            if key in seen:
                continue
            seen.add(key)
            if point["kind"] == "PYTHON":
                assert callable(_resolve_python(point["name"])), point
            elif point["kind"] == "JSON_API":
                assert callable(getattr(json_api, point["name"])), point
            else:
                assert point["name"].startswith("os-optics ") and _cli_exists(point["name"]), point
    assert ("CLI", "os-optics results handoff") in seen and ("CLI", "os-optics project paraxial") in seen


def test_python_json_api_and_cli_give_the_same_catalog():
    catalog = interop.interop_catalog()
    assert catalog["statuses"] == STATUS_MEANINGS and catalog["schema"] == "opensci.interop_catalog"
    reply = json_api.interop_catalog()
    assert reply.pop("ok") is True and reply == json.loads(json.dumps(catalog))
    one = json_api.interop_catalog("quantum-photonics")
    assert one == {"ok": True, "entry": BY_ID["quantum-photonics"]}
    unknown = json_api.interop_catalog("nope")
    assert unknown["ok"] is False and unknown["error"]["code"] == "INTEROP_ENTRY_UNKNOWN"
    caps = json_api.capabilities()["interop_catalog"]
    assert caps["entries"] == interop.entry_ids() and caps["schema_version"] == "1.0"
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli_main(["interop", "list", "--json"]) == 0
    assert json.loads(out.getvalue()) == json.loads(json.dumps(catalog))
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli_main(["interop", "list", "--status", "DEPRECATED"]) == 0
    assert "quantum-photonics-strawberry-fields" in out.getvalue() and "quantum-photonics " not in out.getvalue()
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli_main(["interop", "show", "cad-brep-geometry"]) == 0
    text = out.getvalue()
    assert "third party wrapper" in text and "no executed check recorded" in text
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli_main(["interop", "show", "quantum-photonics", "--json"]) == 0
    assert json.loads(out.getvalue()) == BY_ID["quantum-photonics"]
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli_main(["interop", "show", "nope"]) == 64
    assert json.loads(out.getvalue())["error"]["code"] == "INTEROP_ENTRY_UNKNOWN"


def test_entry_schema_refuses_inconsistent_statuses():
    entry = dict(BY_ID["cavity-interferometer"])
    record = {"tool": "Finesse 3", "version": "3.0.2", "python": "3.12", "date": "2026-10-15", "example": "cases/interop-x/x.py",
              "companion_commit": "abcdef1", "engine_version": "0.8.0", "reference": "closed form", "outcome": "PASS"}
    with pytest.raises(ValueError):
        InteropEntryV1.model_validate({**entry, "verified": [record]})  # GUIDANCE_ONLY never carries verified records
    with pytest.raises(ValueError):
        InteropEntryV1.model_validate({**entry, "status": "VERIFIED_EXAMPLE"})  # a verified example names its record
    assert InteropEntryV1.model_validate({**entry, "status": "VERIFIED_EXAMPLE", "verified": [record]}).verified[0].tool == "Finesse 3"
    with pytest.raises(ValueError):
        InteropEntryV1.model_validate({**entry, "status": "DEPRECATED"})  # needs replaced_by and an archived tool
