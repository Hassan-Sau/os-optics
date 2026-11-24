# Protected delivery test vectors

`vector_v1.json` and `detail_vector_v1.json` are cross-language test vectors of the protected Assembly Preview and
Assembly Detail envelopes (`os_optics.protected_delivery`). The Engine tests open them through the wrapper and compare
the result with the stated plaintext byte for byte.

They hold content only. The packages inside are small synthetic ones built from the contracts in
`src/os_optics/_vendor/shared_contracts` (a board and one lens, fixed hashes), protected with a key and nonce that are
SHA-256 of fixed public labels, and fixed times. No Engine run produced them and nothing names an Engine build, so a new
Engine build or schema does not move them. The OpenSci service produces the same vectors from its side of the envelope
format; a deliberate change of the envelope format replaces both files.
