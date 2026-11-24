# Assembly drawing reference fixtures

A matched pair: the input of the local Detailed Assembly Drawing generator for the reference design
(`example_project()`: one lens on an XY translation mount, resolved by the v0.13 generalized resolution).

| File | What it is |
| --- | --- |
| `reference.assembly-preview.zip` | The Assembly Preview Package (scene schema 2.0) of the reference resolution (five parts, B01..B05, on a 150 x 150 board), bound to the fixed reference resolution id `00000000-0000-4000-8000-0000000005a0`. Scene and geometry are those of `../assembly_preview/reference.assembly-preview.zip`; only the manifest binding (resolution id) and so its digests differ. |
| `reference.assembly-detail.json` | The Assembly Detail Package (`opensci.assembly_detail.package` 3.1) that supplements exactly that preview package: eight dimension references with their basis, released under the direct release (`opensci.commercial.direct_release` 1.0.0, basis `DIRECT_RELEASE`). |

v0.13 multi-chain pairs (several support chains on one breadboard; scene schema 2.0), produced by the service's
generalized commercial resolution -> Assembly Preview projection -> Assembly Detail projection (released under the
confirmed order policy), for `tests/test_assembly_drawing_multichain.py` and the Pyodide parity test:

| File | What it is |
| --- | --- |
| `two_lenses.assembly-preview.zip` / `.assembly-detail.json` | Two lenses, each on its own beam from its own customer source, on a 200 x 200 board (B01..B09, 18 dimensions). |
| `four_lenses.assembly-preview.zip` / `.assembly-detail.json` | Four lenses in a row, the design turned 90 deg (beams along +Y), on a 450 x 150 board (B01..B17, 44 dimensions). |

These two multi-chain pairs are not drift-tested against the current projection; they were re-encoded field by field
into the 2.0 contracts (same poses, parts, geometry bytes and dimensions; digests recomputed by the contract builders). They were projected before
Detail was changed to dimension each customer source only to the optic its own beam meets first, so they still carry
one `EXTERNAL_SOURCE_STANDOFF` per source and optic; the drawing draws what it is given. 8- and 12-chain inputs are synthesized from the four-lens pair by
`drawing_helpers.replicated_chains` / `projected_detail` (a copy of the server's fixed rule table), not stored.
Their customer-supplied source anchors were re-encoded with `customer_supply` (reason
`CUSTOMER_LIGHT_SOURCE`, role `source`, the orientation that takes local +Z along the source's beam with local +Y up);
nothing else changed, digests recomputed by the contract builders.

| File | What it is |
| --- | --- |
| `customer_supplied.assembly-preview.zip` / `.assembly-detail.json` | The reference pair with a customer-supplied (design-only, `NOT_ORDERABLE`) polarizer `P1` on the source's beam, with its reserved envelope and `CUSTOMER_SUPPLIED_COMPONENT` note (built by `tests/test_assembly_drawing_customer_supplied.add_customer_polarizer` + `drawing_helpers.rebound_detail(..., models=(MODEL,))`: the detail names the catalog model of the polarizer, `customer_supplied_models`), for the Pyodide byte-identity test of the customer-supplied drawing (drawn in zh-CN). |

These are synthetic sample packages of the reference design, made as test material and published with this repository under its MIT License. The `usage` block inside each package (permitted uses, "not for redistribution") is part of the sample: it is the text a real delivery carries, and the tests check that it is read and kept; it does not restrict these files.

Customer-facing data only: OpenSci identities, Customer Purchase BOM line references, package-local part refs and
part geometry. In the product the detail is delivered only encrypted (`shared_contracts.protected_assembly_delivery`,
detail envelope + short-lived grant) and only after its release; these plaintext files exist for tests.

`verify_assembly_detail(detail_bytes, preview)` returns no problem for each pair.

## Origin and refresh

The reference pair is the OpenSci service's projection of the reference resolution (commercial resolution -> Assembly
Preview Package -> Assembly Detail Package); the service checks on its side that this projection still produces these
bytes, with the values that name the local run and the catalog release masked (the scene's
`beam_reference.engine_input_hash`, `engine_result_hash` and each `design_binding.catalog_version`, the preview manifest's
`local_validation_hash`, `commercial_submission_hash`, `resolution.resolution_hash`, Project hash, `scene.json` entry
`sha256` and `size_bytes` and `package_content_hash`, and correspondingly the detail's `binding.local_validation_hash`,
`binding.project`'s hash, `binding.preview_package_content_hash` and `detail_content_hash`).

So after a new Engine build or catalog release the committed pair stays valid and self-consistent and needs no refresh:
it names the run that produced it. A deliberate change of its content replaces both files with a new projection, and
refreshes the drawing generator's own expected outputs that bind to their digests (their `binding`) in the same change.

Detail schema 3.1: the four detail documents were re-encoded from 3.0 (schema version, an empty or -- for
`customer_supplied` -- a one-row `customer_supplied_models`, digests recomputed by the contract builders); the
reference pair was then projected again.

`line_display_names_zh_cn` (detail 3.1): the reference detail carries the Simplified Chinese name of each
Customer Purchase BOM line from the resolution's own customer BOM; the `customer_supplied` pair was rebuilt from it
(`add_customer_polarizer()` + `rebound_detail(..., models=(MODEL,))`, which keeps the reference's line names); the
multi-chain pairs carry none (their zh-CN legend falls back to the English names).
