"""The headless JSON API method registry, capabilities, binary transport and dispatch."""

from __future__ import annotations

import inspect
import json

from os_optics.local import json_api
from os_optics.local.api_registry import CATEGORIES, INPUT_INVALID, METHOD_NOT_IN_RUNTIME

#: Module-level functions of json_api that are not methods of the API (helpers, the dispatch itself).
NOT_METHODS = {"call_method", "catalog_summary", "composition_capability", "product_summary", "run_summary", "stopped_reply"}


def test_every_public_json_api_function_is_a_registered_method_or_a_named_helper():
    public = {name for name, value in vars(json_api).items()
              if inspect.isfunction(value) and not name.startswith("_") and value.__module__ in {json_api.__name__, "os_optics.composition.json_wrappers"}}
    assert public - NOT_METHODS == set(json_api.API_METHODS)
    assert not NOT_METHODS & set(json_api.API_METHODS)
    assert {m.category for m in json_api.API_METHODS.values()} <= set(CATEGORIES)
    assert json_api.API_METHODS.of_category("workspace") == sorted(json_api.WORKSPACE_FUNCTIONS)


def test_capabilities_list_the_registered_methods():
    caps = json_api.capabilities()
    methods = caps["json_api_methods"]
    assert list(methods) == sorted(json_api.API_METHODS)
    assert methods["install_catalog"] == {"category": "runtime", "parameters": ["data", "expected_sha256"], "required": ["data"],
                                          "data": "data", "runtime_state": True}
    assert methods["generate_assembly_drawing"] == {"category": "assembly", "parameters": ["package", "detail", "options"],
                                                    "required": ["package", "detail"], "file_map": "package", "data": "detail"}
    assert methods["run_results"]["file_map"] == "files" and methods["commercial_submission"]["file_map"] == "run_files"
    # Python-host keyword-only parameters (a catalog object) are not part of the JSON surface
    assert methods["check_project"]["parameters"] == ["document"] and methods["paraxial_analysis"]["parameters"] == ["document", "request"]
    assert caps["composition"]["functions"] == json_api.API_METHODS.of_category("workspace")
    assert caps["project_ops"]["call"] == "update_project(document, ops, options=None)"
    assert caps["run_freshness"]["call"] == "run_freshness(document, run, verify=False)"


def test_call_method_dispatches_registered_methods_only():
    reply, files = json_api.call_method("example_project_document", {"kind": "focus"})
    assert reply["ok"] is True and reply["kind"] == "focus" and files is None
    json.dumps(reply, allow_nan=False)
    for name in ("catalog_summary", "run_summary", "load_project", "call_method", "__doc__", 3):
        reply, files = json_api.call_method(name, {})
        assert reply["error"]["code"] == METHOD_NOT_IN_RUNTIME and files is None
    for params, message in (({"kind": "project", "catalog": {}}, "not ['catalog']"), ([], "a JSON object"), ({}, "needs ['kind']")):
        reply, _ = json_api.call_method("json_schema", params)
        assert reply["error"]["code"] == INPUT_INVALID and message in reply["error"]["problems"][0]["message"]
    reply, _ = json_api.call_method("json_schema", {"kind": "project"}, file_map={"a": b""})
    assert reply["error"]["code"] == INPUT_INVALID and "takes no files" in reply["error"]["problems"][0]["message"]
    reply, _ = json_api.call_method("run_results", {"files": {}}, file_map={"a": b""})
    assert reply["error"]["code"] == INPUT_INVALID and "given twice" in reply["error"]["problems"][0]["message"]


def test_call_method_binds_bytes_and_returns_the_reply_bytes_apart():
    # run_project's run directory comes back as bytes beside a JSON-safe reply; run_results binds those bytes
    example = json_api.example_project_document()["project"]
    reply, files = json_api.call_method("run_project", {"document": example})
    assert reply["ok"] is True and "files" not in reply and set(files) >= {"manifest.json", "project.json", "local_validation.json"}
    json.dumps(reply, allow_nan=False)
    reply, exported = json_api.call_method("run_results", {"query": {"kind": "export", "export_kind": "monitors", "format": "csv"}}, file_map=files)
    assert reply["ok"] is True, reply
    assert "data" not in reply["result"] and set(exported) == {reply["result"]["file_name"]}
    json.dumps(reply, allow_nan=False)
    reply, submission = json_api.call_method("commercial_submission", {}, file_map=files)
    assert reply["ok"] is True and set(submission) == {"submission.json"}
    reply, none = json_api.call_method("run_freshness", {"document": example}, file_map=files)
    assert reply["ok"] is True and reply["freshness"] == "CURRENT" and none is None


def test_capability_methods_are_json_safe_and_match_the_runtime_registry():
    methods = json_api.capabilities()["json_api_methods"]
    assert list(methods) == sorted(json_api.API_METHODS)
    for name, method in json_api.API_METHODS.items():
        advertised = methods[name]
        assert advertised["category"] == method.category
        assert advertised["parameters"] == [parameter.name for parameter in method.parameters]
        assert advertised["required"] == [parameter.name for parameter in method.parameters if parameter.required]
    json.dumps(methods, allow_nan=False)
