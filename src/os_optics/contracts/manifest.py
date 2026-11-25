"""ValidationManifest: the reproducibility and integrity record of one validation (versions, configuration, hashes)."""

from __future__ import annotations

from .base import StrictModel
from .._version import MANIFEST_VERSION


class ValidationManifest(StrictModel):
    manifest_version: str = MANIFEST_VERSION
    engine_version: str
    engine_build_id: str
    schema_version: str
    project_schema_version: str
    numerical_config_version: str
    input_hash: str
    project_hash: str
    design_projection_hash: str
    result_hash: str
    trace_hash: str | None = None
    random_seed: int | None = None
    prng_id: str | None = None
    model_versions: dict[str, str]
    numerical_tolerances: dict[str, float]
    warnings: tuple[str, ...] = ()
    model_scope: tuple[str, ...] = ()
    runtime: dict[str, str] = {}
