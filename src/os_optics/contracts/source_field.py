"""A field-propagation request (``field-request``): the coherent field of a run's sampled-field or speckle source at one of its
observation planes (``os-optics results field``, ``os_optics.results.field``).

The source's field is built again from the run's request (``SampledFieldSourceV1``, ``SpeckleSourceV1``) and propagated as a
scalar field to the plane: ``ANGULAR_SPECTRUM`` (free space between the source and a plane parallel to it: the exact scalar
transfer function, band-limited) or ``PARAXIAL_TRANSFER`` (the generalized Collins integral of the first-order ray-transfer
matrix of the source axis to the plane, traced through the system's elements; paraxial, without the elements' apertures and
aberrations). ``AUTO`` takes the angular spectrum where the source sees the plane through free space and its window holds
the field there, else the paraxial transfer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import StrictModel

FIELD_REQUEST_KIND = "field-request"
MAX_PADDING = 8
MAX_MAP_POINTS = 1024
#: The largest change of a plane's values the window refinement and the rays' aberration may make before the values are
#: withheld (``value_tolerance``): the default, and the largest a request may state (the aberration check's estimate is
#: evidenced against an exact diffraction reference up to changes of that size).
DEFAULT_VALUE_TOLERANCE = 0.01
MAX_VALUE_TOLERANCE = 0.1


class FieldRegionV1(StrictModel):
    """A rectangle of the plane over which the statistics (mean irradiance, contrast, power) are taken."""

    center_uv_mm: tuple[float, float] = Field((0.0, 0.0), description="Centre [u, v] of the region on the plane, in mm.")
    half_width_mm: tuple[float, float] = Field(description="Half widths [along u, along v] of the region, in mm.")

    @model_validator(mode="after")
    def _positive(self) -> FieldRegionV1:
        if not all(w > 0 for w in self.half_width_mm):
            raise ValueError("half_width_mm must be two positive numbers")
        return self


class FieldRequestV1(StrictModel):
    """What a field propagation computes (module docstring)."""

    source_id: str | None = Field(None, description="The sampled-field or speckle source. Omitted: the run's only one.")
    method: Literal["AUTO", "ANGULAR_SPECTRUM", "PARAXIAL_TRANSFER"] = Field(
        "AUTO", description="ANGULAR_SPECTRUM (free space to a parallel plane), PARAXIAL_TRANSFER (Collins integral of the source "
        "axis's first-order matrix) or AUTO (module docstring).")
    padding: int = Field(2, ge=1, le=MAX_PADDING,
                         description="The transform window in units of the source grid's extent, per axis (zero-padded).")
    region: FieldRegionV1 | None = Field(None, description="The statistics region on the plane. Omitted: the whole output window.")
    realizations: int | None = Field(None, ge=1, description="Use the first this many realizations of a speckle source. Omitted: all.")
    crossing_index: int = Field(0, ge=0, description="Which crossing of the plane by the source axis (0: the first).")
    refinement: bool = Field(True, description="Attach the window refinement (realization 0 on a window twice as wide) to the "
                             "quantities as their convergence records. The window conditions are checked either way.")
    include_field: bool = Field(False, description="Also return the complex field of realization 0 at the plane (re, im) in the arrays.")
    value_tolerance: float = Field(
        DEFAULT_VALUE_TOLERANCE, gt=0.0, le=MAX_VALUE_TOLERANCE,
        description="The accuracy asked of the plane's values: the largest change of them that the window refinement "
        "(window_refinement_change) and the rays' aberration beyond first order (aberration_value_change) may make; above it "
        "the values are OUT_OF_MODEL_SCOPE. Relative for powers, irradiances, rms widths and contrast; relative to the rms "
        "width for the centroid.")
    map_points: int = Field(256, ge=8, le=MAX_MAP_POINTS,
                            description="The irradiance maps in the document hold at most this many points per axis (block sums of the "
                            "computed grid, so their power is the grid's).")


__all__ = ["DEFAULT_VALUE_TOLERANCE", "FIELD_REQUEST_KIND", "FieldRegionV1", "FieldRequestV1", "MAX_VALUE_TOLERANCE"]
