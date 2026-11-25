# Assembly Preview Package reference fixture

`reference.assembly-preview.zip` is the single-file archive form (`package_archive_bytes`) of the Assembly Preview
Package (scene schema 2.0) the OpenSci service projects from the commercial resolution of the reference project
(`example_project()`). It is what a client device receives: OpenSci-facing identities only (`OS-OPT-*` / `OS-MECH-*`
SKUs, B01..B05 BOM lines), `NOMINAL_SHAPE` part geometry (`geometry/*.glb`), `scene.json` and `manifest.json`.

The archive reads back and verifies (`verify_assembly_preview_package` reports no problems): it is a self-consistent
snapshot of one run, its manifest digests match its own files. The service checks on its side that a fresh projection
still produces every byte of the two JSON files and of every `geometry/*.glb`, with the values that name the run and
the catalog release masked (the scene's `beam_reference.engine_input_hash`, `engine_result_hash` and each
`design_binding.catalog_version`; the manifest's `local_validation_hash`, `commercial_submission_hash`,
`resolution.resolution_hash`, the Project hash, `scene.json` entry `sha256` and `size_bytes` and
`package_content_hash`). These values move with any Engine version, schema, result-text or catalog release change though
no geometry moved, so a new Engine build or catalog release needs no refresh: the snapshot simply names the run that
produced it. A deliberate change of the archive's content replaces it with a new projection.

These are synthetic sample packages of the reference design, made as test material and published with this repository under its MIT License. The `usage` block inside each package (permitted uses, "not for redistribution") is part of the sample: it is the text a real delivery carries, and the tests check that it is read and kept; it does not restrict these files.

This is a test fixture. It is not part of the `os-optics` wheel or sdist.
