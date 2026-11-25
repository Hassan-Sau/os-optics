"""A ghost-path analysis request (``ghost-request``): the ghost images of a run's image plane, with a bound on what is not listed.

The run's light is traced again with the reflection of every transmitting interface (a lens or window face: its bare Fresnel
reflection, or its coating's stated reflectance) followed as a ghost branch, up to ``max_bounce_order`` such reflections per
path. An even number of bounces sends the light on toward the image side, an odd number back toward the source. Ghost
reflections at or below ``relative_power_floor`` of their line's input are not traced; they, the reflections beyond the bounce
order, whatever the trace budget leaves out and the light the trace does not follow (scattered light, light ended outside the
model) are summed into an upper bound on the power of every ghost not listed, proven where every ghost surface met a valid
Gaussian beam.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import StrictModel

GHOST_REQUEST_KIND = "ghost-request"
MAX_BOUNCE_ORDER = 4
MAX_FIELDS = 32
MAX_LISTED = 4096
MAX_GHOST_PATHS = 65536
MAX_GHOST_TRACE_STEPS = 1048576
DEFAULT_GHOST_PATHS = 4096
DEFAULT_GHOST_TRACE_STEPS = 16384


class GhostRequestV1(StrictModel):
    """The fields, bounce order, floor and budgets of a ghost-path analysis (module docstring)."""

    fields: tuple[str, ...] | None = Field(None, min_length=1, max_length=MAX_FIELDS,
                                           description="Source ids of the fields. Omitted: every beam source of the run.")
    wavelengths_nm: tuple[float, ...] | None = Field(None, min_length=1, description="Only these spectral lines. Omitted: every line of each field.")
    max_bounce_order: int = Field(2, ge=1, le=MAX_BOUNCE_ORDER,
                                  description="Ghost reflections followed per path. Even orders reach the image side; 2 gives every two-surface ghost.")
    relative_power_floor: float = Field(1e-9, gt=0.0, lt=1.0,
                                        description="A ghost reflection at or below this share of its line's input power is not traced; its power enters the bound.")
    max_listed: int = Field(64, ge=1, le=MAX_LISTED,
                            description="Ghost paths listed per line, the strongest at the image plane first; the others enter the bound with their power.")
    max_paths: int = Field(DEFAULT_GHOST_PATHS, ge=1, le=MAX_GHOST_PATHS, description="Trace budget of the ghost trace: beam paths.")
    max_trace_steps: int = Field(DEFAULT_GHOST_TRACE_STEPS, ge=1, le=MAX_GHOST_TRACE_STEPS,
                                 description="Trace budget of the ghost trace: chief-ray steps over all paths.")
    convergence: bool = Field(
        True, description="Trace again with twice the pupil rings and twice the points per ring: the values are the finer trace's, "
        "each with how far the two traces differ (its convergence). False: one trace with the sources' own pupil sampling, the "
        "values that trace's and each value's convergence NONE (not checked).")

    @model_validator(mode="after")
    def _values(self) -> GhostRequestV1:
        if self.wavelengths_nm is not None and any(w <= 0.0 for w in self.wavelengths_nm):
            raise ValueError("wavelengths_nm must be positive")
        return self


__all__ = ["GHOST_REQUEST_KIND", "GhostRequestV1", "MAX_BOUNCE_ORDER"]
