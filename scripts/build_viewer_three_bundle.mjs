// Builds the vendored three.js bundle of the local viewer (os_optics/viewer/assets/three.bundle.js).
//
// The bundle is the pinned `three` release of scripts/viewer-bundle/package-lock.json, reduced to the names the
// viewer uses, minified, as one classic script that defines `globalThis.OPENSCI_THREE` (so the viewer page works both when
// served from 127.0.0.1 and when written as a single self-contained HTML file). three.js is MIT licensed; its LICENSE is
// copied next to the bundle as THREE_LICENSE.txt.
//
// Usage (after `npm ci` in scripts/viewer-bundle/):
//   node scripts/build_viewer_three_bundle.mjs [path/to/viewer-bundle/node_modules]
//
// Re-run after changing EXPORTS below or the pinned three version; tests/test_viewer.py pins the bundle's version header.
import { createRequire } from "node:module";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const bundleInputs = join(here, "viewer-bundle");
const nodeModules = resolve(process.argv[2] ?? join(bundleInputs, "node_modules"));
const assets = resolve(here, "..", "src", "os_optics", "viewer", "assets");
const require = createRequire(join(nodeModules, "noop.js"));

const threePackage = JSON.parse(readFileSync(join(nodeModules, "three", "package.json"), "utf8"));
const lock = JSON.parse(readFileSync(join(bundleInputs, "package-lock.json"), "utf8"));
const pinned = lock.packages["node_modules/three"].version;
if (threePackage.version !== pinned) {
  throw new Error(`three ${threePackage.version} in ${nodeModules} is not the pinned ${pinned}`);
}

const EXPORTS = [
  "AmbientLight", "BufferAttribute", "BufferGeometry", "Box3", "Color", "CylinderGeometry", "ConeGeometry", "DataTexture", "DirectionalLight",
  "DoubleSide", "EdgesGeometry", "Float32BufferAttribute", "GreaterDepth", "Group", "HemisphereLight", "LessEqualDepth", "Line", "LinearFilter",
  "LineBasicMaterial",
  "LineDashedMaterial", "LineSegments", "Matrix4", "Mesh", "MeshBasicMaterial", "MeshLambertMaterial", "MeshStandardMaterial",
  "NoToneMapping", "OrthographicCamera", "PerspectiveCamera", "Quaternion", "Raycaster", "SRGBColorSpace", "Scene", "ShaderMaterial",
  "SphereGeometry", "Sphere", "Uint32BufferAttribute", "Vector2", "Vector3", "WebGLRenderer",
];

const work = mkdtempSync(join(tmpdir(), "opensci-three-"));
try {
  const entry = join(work, "entry.mjs");
  writeFileSync(entry, [
    `export { ${EXPORTS.join(", ")} } from "three";`,
    `export { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";`,
    `export { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";`,
    `export { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";`,
    `export { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";`,
    "",
  ].join("\n"));
  const { rolldown } = await import(pathToFileURL(require.resolve("rolldown")).href);
  const bundle = await rolldown({ input: entry, resolve: { modules: [nodeModules] }, treeshake: true });
  const { output } = await bundle.generate({ format: "iife", name: "OPENSCI_THREE", minify: true });
  const header = `/*! three.js r${threePackage.version.split(".")[1]} (npm three@${threePackage.version}), MIT License, `
    + `see THREE_LICENSE.txt. Reduced to the local viewer's imports. */\n`;
  writeFileSync(join(assets, "three.bundle.js"), header + output[0].code);
  copyFileSync(join(nodeModules, "three", "LICENSE"), join(assets, "THREE_LICENSE.txt"));
  console.log(`wrote three.bundle.js (three ${threePackage.version}, ${output[0].code.length} bytes)`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
