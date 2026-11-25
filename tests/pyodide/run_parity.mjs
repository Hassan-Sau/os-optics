// Real Pyodide integration harness for os-optics (no mocking):
//   load Pyodide 314.0.x -> load numpy/scipy/pydantic/shapely -> micropip-install the built engine wheel ->
//   import os_optics -> run every fixture request through validate_json -> install the public optical catalog release
//   given with --catalog through json_api.install_catalog (as the browser Designer does: the wheel bundles no catalog and
//   Pyodide never fetches one itself) -> run the local design workflow
//   (os_optics.local) -> run the browser Designer's JSON entry points (os_optics.local.json_api) on the given
//   Project cases -> write results + an environment/timing report.
//
//   Optionally, run the local assembly layout / collision check (json_api.validate_assembly_package) on an Assembly
//   Preview Package archive, and the local Detailed Assembly Drawing (json_api.generate_assembly_drawing) of an Assembly
//   Preview Package archive + its Assembly Detail Package.
//
// usage: node run_parity.mjs --wheel <path.whl> --requests <dir> --out <dir> --catalog <catalog.json> [--warm-iterations N]
//                            [--json-api-cases <file.json>]
//                            [--assembly-package <archive.zip>] [--drawing-package <archive.zip> --drawing-detail <detail.json>]
//                            [--drawing-package-2 <archive.zip> --drawing-detail-2 <detail.json>]   (a second pair: _drawing_2.json)
//                            [--drawing-package-3 <archive.zip> --drawing-detail-3 <detail.json>]   (a third pair, drawn in
//                                                                                                     zh-CN: _drawing_3.json)
//                            [--layout-package <optical-layout.json>]   (the Optical Layout Drawing in zh-CN: _layout.json)
//                            [--network-cases <file.json>]   (json_api.linear_optical_network of Engine requests: _network.json)
//                            [--json-api-calls <file.json>]  ({name: [json_api function, args]} -> {name: reply}: _json_api_calls.json;
//                                                             prescription import, native components, sections, catalog_model, ...)
//                            [--prepared-sequence <file.json>]  ({"script": Python source defining sequence(json_api, documents),
//                                                             "documents": {...}}: calls on the prepared evaluation handles of one
//                                                             runtime, in order -> its JSON-safe result: _prepared.json)
//                            [--skip fixtures,local_workflow]  (leave out the fixture requests -- and with them the warm
//                                                             benchmark -- or the local design workflow; the optional sections above
//                                                             run only when their arguments are given. tests/
//                                                             test_packaging_and_pyodide.py asks for the sections its selected
//                                                             tests read, so a CI shard runs only its own.)
import { loadPyodide, version as pyodideVersion } from "pyodide";
import fs from "node:fs";
import path from "node:path";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

const wheel = arg("wheel");
const catalogFile = arg("catalog");
const requestsDir = arg("requests");
const outDir = arg("out");
const warmIterations = Number(arg("warm-iterations", "5"));
const jsonApiCases = arg("json-api-cases");
const assemblyPackage = arg("assembly-package");
const drawingPackage = arg("drawing-package");
const drawingDetail = arg("drawing-detail");
const drawingPackage2 = arg("drawing-package-2");
const drawingDetail2 = arg("drawing-detail-2");
const drawingPackage3 = arg("drawing-package-3");
const drawingDetail3 = arg("drawing-detail-3");
const layoutPackage = arg("layout-package");
const networkCases = arg("network-cases");
const jsonApiCalls = arg("json-api-calls");
const preparedSequence = arg("prepared-sequence");
const SKIPPABLE = ["fixtures", "local_workflow"];
const skip = new Set((arg("skip", "") || "").split(",").filter(Boolean));
for (const name of skip) {
  if (!SKIPPABLE.includes(name)) {
    console.error(`--skip takes a comma-separated subset of ${SKIPPABLE.join(",")}, not ${name}`);
    process.exit(64);
  }
}
if (!wheel || !requestsDir || !outDir || !catalogFile) {
  console.error("usage: node run_parity.mjs --wheel W --requests DIR --out DIR --catalog CATALOG [--warm-iterations N]");
  process.exit(64);
}
fs.mkdirSync(outDir, { recursive: true });

const report = { pyodide_version: pyodideVersion, timings_ms: {}, fixtures: {} };
let t = performance.now();
const lap = (label) => {
  const now = performance.now();
  report.timings_ms[label] = Math.round(now - t);
  t = now;
};

const pyodide = await loadPyodide({});
lap("pyodide_core_load");
await pyodide.loadPackage(["numpy", "scipy", "pydantic", "shapely", "micropip"]);
lap("dependency_packages_load");

const wheelName = path.basename(wheel);
pyodide.FS.writeFile(`/tmp/${wheelName}`, fs.readFileSync(wheel));
await pyodide.runPythonAsync(`import micropip\nawait micropip.install("emfs:/tmp/${wheelName}")`);
lap("engine_wheel_install");

pyodide.runPython(`
import sys, json, time, importlib.metadata as md
import numpy, scipy, pydantic, shapely
import os_optics
from os_optics import validate_json
from os_optics.manifest import engine_build_id

def environment():
    return json.dumps({
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pydantic": pydantic.VERSION,
        "shapely": shapely.__version__,
        "os_optics": os_optics.__version__,
        "engine_build_id": engine_build_id(),
        "engine_dist_version": md.version("os-optics"),
    })

def run_fixture(text):
    t0 = time.perf_counter()
    out = validate_json(text)
    return out, (time.perf_counter() - t0) * 1000.0
`);
lap("engine_import");
report.environment = JSON.parse(pyodide.globals.get("environment")());

const runFixture = pyodide.globals.get("run_fixture");
const files = skip.has("fixtures") ? [] : fs.readdirSync(requestsDir).filter((f) => f.endsWith(".json")).sort();
for (const file of files) {
  const name = file.replace(/\.json$/, "");
  const text = fs.readFileSync(path.join(requestsDir, file), "utf-8");
  const res = runFixture(text).toJs();
  fs.writeFileSync(path.join(outDir, file), res[0], "utf-8");
  const warm = [];
  for (let i = 0; i < warmIterations; i++) warm.push(runFixture(text).toJs()[1]);
  warm.sort((a, b) => a - b);
  report.fixtures[name] = {
    cold_ms: Math.round(res[1] * 10) / 10,
    warm_median_ms: Math.round(warm[Math.floor(warm.length / 2)] * 10) / 10,
    warm_max_ms: Math.round(warm[warm.length - 1] * 10) / 10,
  };
}
if (!skip.has("fixtures")) lap("fixtures_total");

// The public optical catalog, installed in memory from its exact bytes (never fetched by the runtime itself).
pyodide.FS.writeFile("/tmp/catalog.json", fs.readFileSync(catalogFile));
pyodide.runPython(`
def install_catalog():
    import hashlib
    from os_optics.local import json_api
    data = open("/tmp/catalog.json", "rb").read()
    before = json_api.catalog_status()
    reply = json_api.install_catalog(data, hashlib.sha256(data).hexdigest())
    return json.dumps({"before": before, "install": reply, "after": json_api.catalog_status()})
`);
report.catalog = JSON.parse(pyodide.globals.get("install_catalog")());
if (!report.catalog.install.ok) {
  console.error(`json_api.install_catalog refused the catalog: ${JSON.stringify(report.catalog.install)}`);
  process.exit(1);
}
lap("catalog_install");

// The local design workflow in the browser (Pyodide) runtime: one run -> record + Optical Simulation View + run directory.
if (!skip.has("local_workflow")) {
pyodide.runPython(`
def local_workflow():
    from os_optics.local import example_project, run_optical_project, verify_run_artifacts
    run = run_optical_project(example_project(), output_dir="/tmp/opensci_run")
    return json.dumps({
        "status": run.status.value,
        "record": run.local_validation.payload(),
        "scene": run.optical_view.payload(),
        "verify": list(verify_run_artifacts("/tmp/opensci_run")),
        "files": sorted(p.name for p in run.artifacts.directory.iterdir()),
    })
`);
fs.writeFileSync(path.join(outDir, "_local_workflow.json"), pyodide.globals.get("local_workflow")(), "utf-8");
lap("local_workflow");
}
// The browser Designer's transport (os_optics.local.json_api): each case -> the run envelope, files as SHA-256 + bytes.
if (jsonApiCases) {
  pyodide.runPython(`
RESULT_QUERIES = ("monitors", "paths", "power_budget", "exposures")


def json_api_cases(cases_json):
    import base64, hashlib
    from os_optics.local import json_api
    out = {"capabilities": json_api.capabilities(), "cases": {}}
    for name, document in json.loads(cases_json).items():
        reply = json_api.run_project(document, output_dir=f"/tmp/json_api/{name}")
        files = reply.pop("files", None) or {}
        if files:  # the scientific results of the run, read back from its in-memory files (json_api.run_results)
            reply["results"] = {q: json_api.run_results(files, {"kind": q}) for q in RESULT_QUERIES}
            reply["results"]["rays"] = json_api.run_results(files, {"kind": "rays", "limit": 4})
            monitors = reply["results"]["monitors"]
            if monitors["ok"] and monitors["result"]:
                reply["results"]["handoff"] = json_api.run_results(files, {"kind": "handoff",
                                                                           "monitor": monitors["result"][0]["observation_id"]})
            if name.startswith("ghost_"):  # the ghost-path analysis of the image plane (json_api.run_results kind "ghosts")
                reply["results"]["ghosts"] = json_api.run_results(files, {"kind": "ghosts", "monitor": "image"})
            if name.startswith("field_"):  # the field propagation of a sampled field (json_api.run_results kind "field")
                reply["results"]["field"] = json_api.run_results(files, {"kind": "field", "monitor": "P"})
        reply["files"] = {k: {"sha256": hashlib.sha256(v).hexdigest(), "base64": base64.b64encode(v).decode()} for k, v in files.items()}
        reply["check"] = json_api.check_project(document)
        reply["network"] = json_api.linear_optical_network(document)
        out["cases"][name] = reply
    return json.dumps(out)
`);
  const cases = fs.readFileSync(jsonApiCases, "utf-8");
  fs.writeFileSync(path.join(outDir, "_json_api.json"), pyodide.globals.get("json_api_cases")(cases), "utf-8");
  lap("json_api_cases");
}
// The linear optical network export (json_api.linear_optical_network) of Engine requests: the same contract hash as native.
if (networkCases) {
  pyodide.runPython(`
def network_cases(cases_json):
    from os_optics.local import json_api
    return json.dumps({name: json_api.linear_optical_network(document) for name, document in json.loads(cases_json).items()})
`);
  const cases = fs.readFileSync(networkCases, "utf-8");
  fs.writeFileSync(path.join(outDir, "_network.json"), pyodide.globals.get("network_cases")(cases), "utf-8");
  lap("network_cases");
}
// Other json_api entry points the browser Designer / Agent calls (prescription import, native components, exact sections,
// catalog_model, optical-solid inspection): each named call's reply, compared with native Python byte for byte.
if (jsonApiCalls) {
  pyodide.runPython(`
def json_api_calls(calls_json):
    import hashlib
    from os_optics.local import json_api
    # (a reply's bytes -- a layout's GLB -- are compared by their SHA-256)
    def digest(value):
        if isinstance(value, (bytes, bytearray)):
            return {"sha256": hashlib.sha256(bytes(value)).hexdigest(), "bytes": len(value)}
        raise TypeError(type(value).__name__)
    return json.dumps({name: getattr(json_api, function)(*args) for name, (function, args) in json.loads(calls_json).items()}, default=digest)
`);
  const calls = fs.readFileSync(jsonApiCalls, "utf-8");
  fs.writeFileSync(path.join(outDir, "_json_api_calls.json"), pyodide.globals.get("json_api_calls")(calls), "utf-8");
  lap("json_api_calls");
}
// The prepared evaluation handles (json_api.prepare_evaluation / evaluate_prepared / release_evaluation /
// evaluation_handles) as the browser Designer's worker keeps them, in its module state between calls: the sequence of
// calls the test gives (Python source, run here and natively), its result compared with native Python.
if (preparedSequence) {
  pyodide.runPython(`
def prepared_sequence(text):
    from os_optics.local import json_api
    given = json.loads(text)
    namespace = {}
    exec(given["script"], namespace)
    return json.dumps(namespace["sequence"](json_api, given["documents"]))
`);
  fs.writeFileSync(path.join(outDir, "_prepared.json"), pyodide.globals.get("prepared_sequence")(fs.readFileSync(preparedSequence, "utf-8")), "utf-8");
  lap("prepared_sequence");
}
// The local assembly layout / collision check in the browser (Pyodide) runtime: the Assembly Preview Package
// archive bytes -> json_api.validate_assembly_package -> the AssemblyLocalValidationV1 envelope. The same archive is
// passed three ways a delivery layer may hand it over: Python bytes, a JS Uint8Array (a memoryview in Python) and a
// {path: bytes} file map.
if (assemblyPackage) {
  pyodide.runPython(`
def assembly_validation(archive_proxy):
    import base64, hashlib, io, zipfile
    from os_optics.local import json_api
    view = archive_proxy.to_py()  # memoryview over a copy of the JS Uint8Array
    archive = view.tobytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        file_map = {name: z.read(name) for name in z.namelist()}
    out = {}
    for label, source in (("bytes", archive), ("memoryview", view), ("file_map", file_map)):
        t0 = time.perf_counter()
        reply = json_api.validate_assembly_package(source)
        ms = (time.perf_counter() - t0) * 1000.0
        data = reply.pop("result_file", None)
        if data is not None:
            reply["result_file"] = {"sha256": hashlib.sha256(data).hexdigest(), "base64": base64.b64encode(data).decode()}
        reply["ms"] = ms
        out[label] = reply
    return json.dumps(out)
`);
  const archive = new Uint8Array(fs.readFileSync(assemblyPackage));
  fs.writeFileSync(path.join(outDir, "_assembly.json"), pyodide.globals.get("assembly_validation")(archive), "utf-8");
  lap("assembly_validation");
}
// The local Detailed Assembly Drawing in the browser (Pyodide) runtime: archive + detail bytes (as a Designer
// worker passes them, JS Uint8Arrays) -> json_api.generate_assembly_drawing -> the four files (SHA-256 + base64).
if (drawingPackage && drawingDetail) {
  pyodide.runPython(`
def assembly_drawing(archive_proxy, detail_proxy, locale="en"):
    import base64, hashlib
    from os_optics.local import json_api
    archive, detail = archive_proxy.to_py(), detail_proxy.to_py()  # memoryviews
    runs = []
    for _ in range(2):
        t0 = time.perf_counter()
        reply = json_api.generate_assembly_drawing(archive, detail, {"locale": locale})
        runs.append((time.perf_counter() - t0) * 1000.0)
    files = reply.pop("files", None) or {}
    reply["files"] = {k: {"sha256": hashlib.sha256(v).hexdigest(), "base64": base64.b64encode(v).decode()} for k, v in files.items()}
    reply["ms"] = runs
    return json.dumps(reply)
`);
  const archive = new Uint8Array(fs.readFileSync(drawingPackage));
  const detail = new Uint8Array(fs.readFileSync(drawingDetail));
  fs.writeFileSync(path.join(outDir, "_drawing.json"), pyodide.globals.get("assembly_drawing")(archive, detail), "utf-8");
  lap("assembly_drawing");
  if (drawingPackage2 && drawingDetail2) {
    const archive2 = new Uint8Array(fs.readFileSync(drawingPackage2));
    const detail2 = new Uint8Array(fs.readFileSync(drawingDetail2));
    fs.writeFileSync(path.join(outDir, "_drawing_2.json"), pyodide.globals.get("assembly_drawing")(archive2, detail2), "utf-8");
    lap("assembly_drawing_2");
  }
  if (drawingPackage3 && drawingDetail3) {
    const archive3 = new Uint8Array(fs.readFileSync(drawingPackage3));
    const detail3 = new Uint8Array(fs.readFileSync(drawingDetail3));
    fs.writeFileSync(path.join(outDir, "_drawing_3.json"), pyodide.globals.get("assembly_drawing")(archive3, detail3, "zh-CN"), "utf-8");
    lap("assembly_drawing_3");
  }
}
// The Optical Layout Drawing of an optics-only purchase (json_api.generate_optical_layout) of a package's JSON text, in zh-CN.
if (layoutPackage) {
  pyodide.runPython(`
def optical_layout(text):
    import base64, hashlib
    from os_optics.local import json_api
    reply = json_api.generate_optical_layout(text, "zh-CN")
    files = reply.pop("files", None) or {}
    reply["files"] = {k: {"sha256": hashlib.sha256(v).hexdigest(), "base64": base64.b64encode(v).decode()} for k, v in files.items()}
    return json.dumps(reply)
`);
  fs.writeFileSync(path.join(outDir, "_layout.json"), pyodide.globals.get("optical_layout")(fs.readFileSync(layoutPackage, "utf-8")), "utf-8");
  lap("optical_layout");
}
fs.writeFileSync(path.join(outDir, "_report.json"), JSON.stringify(report, null, 2), "utf-8");
console.log(JSON.stringify(report, null, 2));
