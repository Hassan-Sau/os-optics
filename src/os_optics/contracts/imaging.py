"""An imaging-analysis request (``imaging-request``): point spread function, MTF and encircled energy of a run's image plane.

Each field is a source of the run (its spectral lines): its light, traced again with a dense regular grid of rays that fills
the system's aperture stop with uniform irradiance (a point object), reaches the image plane, the observation plane the
analysis names. The geometric wavefront there, on a reference sphere about the image point, gives the point spread function
by the Debye integral over the traced rays; its Fourier transform gives the MTF.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import StrictModel

IMAGING_REQUEST_KIND = "imaging-request"
MIN_LAUNCH_GRID = 16
MAX_LAUNCH_GRID = 256
MIN_PSF_SAMPLES = 16
MAX_PSF_SAMPLES = 512
MAX_FIELDS = 32
MAX_FREQUENCIES = 128
#: The search for the system's aperture grows the launch disc up to this many beam radii by default; a request states at least
#: the disc the search starts from and at most ``LAUNCH_SEARCH_MAX_RADII``.
LAUNCH_SEARCH_DEFAULT_RADII = 16.0
LAUNCH_SEARCH_MIN_RADII = 2.0
LAUNCH_SEARCH_MAX_RADII = 1024.0


class ImagingRequestV1(StrictModel):
    """The fields, sampling and output of an imaging analysis (module docstring)."""

    fields: tuple[str, ...] | None = Field(None, min_length=1, max_length=MAX_FIELDS,
                                           description="Source ids of the fields. Omitted: every beam source whose light reaches the image plane.")
    wavelengths_nm: tuple[float, ...] | None = Field(None, min_length=1, description="Only these spectral lines. Omitted: every line of each field.")
    crossing_index: int | None = Field(None, ge=0, description="Which crossing of the image plane by the light (0 = first). Omitted: the last one.")
    launch_grid: int = Field(64, ge=MIN_LAUNCH_GRID, le=MAX_LAUNCH_GRID,
                             description="Rays across the launch disc; the convergence check also runs twice as many per axis.")
    launch_radius_beam_radii: float | None = Field(
        None, gt=0, description="Radius of the launch disc in the source's beam radii: the pupil, when stated. Omitted: grown from 2 beam radii until the system's "
        "apertures cut the rays (up to max_launch_radius_beam_radii), then fitted to the light that passes; where no aperture cuts it, the analysis is that of a disc "
        "of 2 beam radii, stated as a stand-in pupil on every quantity.")
    max_launch_radius_beam_radii: float = Field(
        LAUNCH_SEARCH_DEFAULT_RADII, ge=LAUNCH_SEARCH_MIN_RADII, le=LAUNCH_SEARCH_MAX_RADII,
        description="How far the search for the system's aperture grows the launch disc, in the source's beam radii (not read when launch_radius_beam_radii is stated).")
    psf_samples: int = Field(64, ge=MIN_PSF_SAMPLES, le=MAX_PSF_SAMPLES, description="Points per axis of the point spread function map.")
    psf_half_width_um: float | None = Field(
        None, gt=0, description="Half width of the point spread function map in micrometres. Omitted: sixteen Airy radii, or twice the geometric spot when that is larger.")
    frequencies_cyc_per_mm: tuple[float, ...] | None = Field(
        None, min_length=1, max_length=MAX_FREQUENCIES, description="Spatial frequencies of the MTF in cycles/mm. Omitted: 33 steps from 0 to the cut-off 2 NA / lambda.")
    reference: Literal["CENTROID", "CHIEF"] = Field(
        "CENTROID", description="Centre of the reference sphere and of the maps: the centroid of the traced rays on the image plane, or the source axis's ray (where that ray fell beside a clear aperture, the passing pupil ray that carries the beam's reference from there).")
    include_psf_map: bool = Field(False, description="Also return the point spread function map of every field.")
    convergence: bool = Field(
        True, description="Repeat the analysis with twice the rays per axis: each value is the finer grid's, with how far the two "
        "grids differ (its convergence). False: the analysis runs once, on launch_grid rays: the values are that grid's, each "
        "value's convergence is NONE (not checked), and on that coarser pupil a diffraction quantity can be OUT_OF_MODEL_SCOPE "
        "(PUPIL_UNDERSAMPLED) where the default computes it; raise launch_grid for a finer pupil.")
    diffraction: bool = Field(
        True, description="Compute the diffraction quantities: the point spread function, Strehl ratio, MTF, diffraction-limited MTF "
        "and encircled energy. False: the geometric and wavefront quantities alone; the diffraction quantities are NOT_COMPUTED "
        "(reason NOT_REQUESTED), and a field without a wavefront states it as the full analysis does.")

    @model_validator(mode="after")
    def _values(self) -> ImagingRequestV1:
        if self.psf_samples % 2:
            raise ValueError("psf_samples must be even")
        if self.include_psf_map and not self.diffraction:
            raise ValueError("include_psf_map needs the diffraction quantities (diffraction true)")
        if self.frequencies_cyc_per_mm is not None and any(f < 0.0 for f in self.frequencies_cyc_per_mm):
            raise ValueError("frequencies_cyc_per_mm must not be negative")
        if self.wavelengths_nm is not None and any(w <= 0.0 for w in self.wavelengths_nm):
            raise ValueError("wavelengths_nm must be positive")
        return self


__all__ = ["IMAGING_REQUEST_KIND", "ImagingRequestV1"]
