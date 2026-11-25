"""Public request contracts: EngineConfiguration and ValidationRequest."""

from __future__ import annotations

from pydantic import Field

from ..config import NumericalConfig
from .base import StrictModel
from .project import ProjectSnapshot


class EngineConfiguration(StrictModel):
    """Numerical settings plus which stages to run."""

    numerical: NumericalConfig = NumericalConfig()
    run_physics: bool = True
    run_layout: bool = True
    run_control: bool = True
    include_paths: bool = Field(True, description="Include OpticalPath segment lists in the result.")
    random_seed: int | None = Field(None, ge=0, description="Seed for Monte Carlo (required if Monte Carlo is requested).")


class ValidationRequest(StrictModel):
    project: ProjectSnapshot
    configuration: EngineConfiguration = EngineConfiguration()
