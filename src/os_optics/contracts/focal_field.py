"""A focal-field request (``focal-field-request``): the vector focus of an ideal aplanatic objective near its focal point.

Frames and signs. The entrance pupil has coordinates ``(x, y)`` in mm, centred on the objective's axis; the focal region has
coordinates ``(x, y, z)`` in micrometres, parallel to them, with the origin at the objective's focal point and ``+z`` the
direction of propagation (away from the objective). A collimated input whose direction cosines are ``(l_x, l_y)`` focuses at
``x = F l_x``, ``y = F l_y``, with ``F`` the effective focal length (no inversion). The objective obeys the sine condition: a
pupil point at height ``h`` sends its ray to the focus at the angle ``theta`` with ``h = F n sin(theta)``, ``n`` the index of
the focal medium, so the pupil radius is ``F NA``. The light on the pupil plane is in air.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .._vendor.shared_contracts.optical_design.pupil_field import PupilBeamV1
from .base import Complex, StrictModel

FOCAL_FIELD_REQUEST_KIND = "focal-field-request"
#: The largest sine of the marginal ray in the focal medium (``NA / n``) of the model.
MAX_MARGINAL_SINE = 0.999
MAX_BEAMS = 64
MAX_ZERNIKE_ORDER = 12
MAX_ZERNIKE_TERMS = 45
MAX_OUTPUT_SAMPLES = 201
MAX_Z_PLANES = 16
MAX_ENCIRCLED_RADII = 32
MAX_QUADRATURE_NODES = 512
MAX_GRID_CELLS = 1024


class WindowV1(StrictModel):
    """A plane-parallel plate between the objective and the focus, normal to the axis (a vacuum window), in the focal medium on both sides."""

    thickness_mm: float = Field(gt=0, description="Thickness of the plate in mm.")
    refractive_index: float = Field(gt=0, description="Refractive index of the plate at the wavelength.")
    surfaces: Literal["UNCOATED", "IDEAL_ANTIREFLECTION"] = Field(
        "UNCOATED", description="UNCOATED: Fresnel transmission of both surfaces for p and s light. IDEAL_ANTIREFLECTION: full transmission at every angle.")


class IdealAplanaticObjectiveV1(StrictModel):
    """An ideal aplanatic objective: free of aberration at every field point used, obeying the sine condition."""

    model: Literal["IDEAL_APLANATIC"]
    numerical_aperture: float = Field(gt=0, description="Numerical aperture n sin(theta_max).")
    focal_length_mm: float = Field(gt=0, description="Effective focal length F in mm (1 / power); the pupil radius is F times the numerical aperture.")
    focal_medium_index: float = Field(1.0, ge=1.0, description="Refractive index of the medium at the focus (1 for air or vacuum, an immersion liquid's index otherwise).")
    window: WindowV1 | None = Field(None, description="A plate between the objective and the focus. Omitted: none.")

    @model_validator(mode="after")
    def _below_grazing(self) -> IdealAplanaticObjectiveV1:
        if self.numerical_aperture / self.focal_medium_index > MAX_MARGINAL_SINE:
            raise ValueError(f"numerical_aperture / focal_medium_index must be at most {MAX_MARGINAL_SINE}")
        if self.window is not None and self.window.refractive_index < self.focal_medium_index:
            raise ValueError("the window's refractive index must be at least the focal medium's: light at the rim would not pass it")
        return self


class LinearPolarizationV1(StrictModel):
    kind: Literal["LINEAR"]
    angle_deg: float = Field(0.0, description="Direction of the electric field on the pupil, from +x toward +y, in degrees.")


class CircularPolarizationV1(StrictModel):
    kind: Literal["CIRCULAR"]
    handedness: Literal["RIGHT", "LEFT"] = Field(
        description="RIGHT: Jones vector (1, -i) / sqrt(2); LEFT: (1, i) / sqrt(2), for fields exp(i (k z - w t)).")


class JonesPolarizationV1(StrictModel):
    kind: Literal["JONES"]
    jones: tuple[Complex, Complex] = Field(description="Jones vector (E_x, E_y) on the pupil as [real, imag] pairs; normalized here.")

    @model_validator(mode="after")
    def _nonzero(self) -> JonesPolarizationV1:
        if all(part == 0.0 for component in self.jones for part in component):
            raise ValueError("the Jones vector must not be zero")
        return self


class UnpolarizedV1(StrictModel):
    kind: Literal["UNPOLARIZED"]


PolarizationV1 = Annotated[LinearPolarizationV1 | CircularPolarizationV1 | JonesPolarizationV1 | UnpolarizedV1, Field(discriminator="kind")]


class UniformPupilV1(StrictModel):
    """Uniform irradiance filling the pupil; ``power_w`` is the power inside it."""

    kind: Literal["UNIFORM"]
    power_w: float = Field(gt=0, description="Power inside the pupil in W.")
    direction_cosines: tuple[float, float] = Field((0.0, 0.0), description="Direction cosines [l_x, l_y] of the collimated input: it focuses at x = F l_x, y = F l_y.")


class AnnularPupilV1(StrictModel):
    """Uniform irradiance over an annulus of the pupil (a central obstruction); ``power_w`` is the power inside the annulus."""

    kind: Literal["ANNULAR"]
    power_w: float = Field(gt=0, description="Power inside the annulus in W.")
    inner_radius_fraction: float = Field(gt=0, lt=1, description="Inner radius of the annulus as a share of the pupil radius.")
    direction_cosines: tuple[float, float] = Field((0.0, 0.0), description="Direction cosines [l_x, l_y] of the collimated input.")


class GaussianPupilV1(StrictModel):
    """A collimated round Gaussian beam; ``power_w`` is its whole power, of which the pupil passes a part."""

    kind: Literal["GAUSSIAN"]
    power_w: float = Field(gt=0, description="Whole power of the beam in W.")
    radius_mm: float = Field(gt=0, description="1/e^2 intensity radius on the pupil in mm.")
    center_mm: tuple[float, float] = Field((0.0, 0.0), description="Beam centre [x, y] on the pupil in mm.")
    direction_cosines: tuple[float, float] = Field((0.0, 0.0), description="Direction cosines [l_x, l_y] of the beam: it focuses at x = F l_x, y = F l_y.")


class BeamsPupilV1(StrictModel):
    """Several TEM00 beams on the pupil (a set of field points, several foci): each focuses at x = F l_x, y = F l_y."""

    kind: Literal["BEAMS"]
    beams: tuple[PupilBeamV1, ...] = Field(min_length=1, max_length=MAX_BEAMS, description="The beams.")
    composition: Literal["INCOHERENT", "COHERENT"] = Field(
        description="INCOHERENT: irradiances add (beams of different frequencies over a long exposure). COHERENT: fields add with their stated phases.")


class HandoffBeamsPupilV1(StrictModel):
    """The beams a run delivers to the objective's pupil plane, with their Jones vectors, read from the run's handoff at that plane."""

    kind: Literal["HANDOFF_BEAMS"]
    composition: Literal["INCOHERENT", "COHERENT"] = Field(
        description="INCOHERENT: irradiances add. COHERENT: fields add with the phases of the handoff, which every beam must define.")
    path_ids: tuple[str, ...] | None = Field(None, min_length=1, description="Only these beams, by path id. Omitted: every beam of this wavelength.")


FocalIlluminationV1 = Annotated[UniformPupilV1 | AnnularPupilV1 | GaussianPupilV1 | BeamsPupilV1 | HandoffBeamsPupilV1, Field(discriminator="kind")]


class ZernikeTermV1(StrictModel):
    """One Zernike term of the pupil phase: ``coefficient_waves * N R_n^|m|(rho) cos(m phi)`` (``sin(|m| phi)`` for negative
    ``m``), ``N`` the normalization to unit RMS over the pupil, ``rho`` the pupil radius fraction."""

    n: int = Field(ge=0, le=MAX_ZERNIKE_ORDER, description="Radial order.")
    m: int = Field(description="Azimuthal frequency, |m| <= n, n - |m| even.")
    coefficient_waves: float = Field(description="RMS of the term in waves at the wavelength.")

    @model_validator(mode="after")
    def _indices(self) -> ZernikeTermV1:
        if abs(self.m) > self.n or (self.n - abs(self.m)) % 2:
            raise ValueError("a Zernike term needs |m| <= n and n - |m| even")
        return self


class ZernikePhaseV1(StrictModel):
    """A pupil phase stated as Zernike terms over the pupil (an objective's measured or specified aberration)."""

    kind: Literal["ZERNIKE"]
    terms: tuple[ZernikeTermV1, ...] = Field(min_length=1, max_length=MAX_ZERNIKE_TERMS, description="The terms.")


class FocalSamplingV1(StrictModel):
    """Discretization of the integrals; every quantity is also computed with each count doubled (its convergence)."""

    theta_nodes: int = Field(64, ge=8, le=MAX_QUADRATURE_NODES, description="Gauss-Legendre nodes in the polar angle.")
    phi_nodes: int = Field(64, ge=8, le=MAX_QUADRATURE_NODES, description="Azimuthal nodes of the direct sphere sum (pupils without rotational symmetry).")
    grid_cells: int = Field(128, ge=16, le=MAX_GRID_CELLS, description="Cells per axis of the Cartesian pupil grid of the focal maps and the energy check.")


class FocalOutputV1(StrictModel):
    """The maps of the reply."""

    half_extent_um: tuple[float, float, float] | None = Field(
        None, description="Half widths [x, y, z] of the maps around the focus in micrometres. Omitted: three times lambda / NA laterally and three depths of focus axially.")
    samples: int = Field(41, ge=3, le=MAX_OUTPUT_SAMPLES, description="Points per axis of every map.")
    z_planes_um: tuple[float, ...] | None = Field(
        None, max_length=MAX_Z_PLANES, description="Axial positions of the transverse maps in micrometres from the focal point. Omitted: the plane of the peak.")
    include_components: bool = Field(False, description="Also return E_x, E_y, E_z of each transverse map as real and imaginary parts.")
    encircled_energy_radii_um: tuple[float, ...] | None = Field(
        None, max_length=MAX_ENCIRCLED_RADII, description="Radii of the encircled-energy values in micrometres. Omitted: a curve out to three times lambda / NA.")


class FocalFieldRequestV1(StrictModel):
    """The vector focal field of an ideal aplanatic objective (module docstring)."""

    wavelength_nm: float = Field(gt=0, description="Vacuum wavelength in nm.")
    objective: IdealAplanaticObjectiveV1 = Field(description="The objective.")
    illumination: FocalIlluminationV1 = Field(description="The light on the entrance pupil.")
    polarization: PolarizationV1 | None = Field(
        None, description="Polarization of the light on the pupil. Required for a stated illumination; omitted for HANDOFF_BEAMS, whose Jones vectors come from the run.")
    pupil_phase: ZernikePhaseV1 | None = Field(None, description="A phase over the pupil. Omitted: none (the ideal objective).")
    sampling: FocalSamplingV1 = Field(FocalSamplingV1(), description="Discretization.")
    output: FocalOutputV1 = Field(FocalOutputV1(), description="The maps of the reply.")

    @model_validator(mode="after")
    def _polarization_source(self) -> FocalFieldRequestV1:
        handoff = isinstance(self.illumination, HandoffBeamsPupilV1)
        if handoff and self.polarization is not None:
            raise ValueError("HANDOFF_BEAMS take each beam's Jones vector from the run: omit polarization")
        if not handoff and self.polarization is None:
            raise ValueError("polarization is required for a stated illumination")
        return self


__all__ = [
    "AnnularPupilV1", "BeamsPupilV1", "CircularPolarizationV1", "FOCAL_FIELD_REQUEST_KIND", "FocalFieldRequestV1", "FocalOutputV1",
    "FocalSamplingV1", "GaussianPupilV1", "HandoffBeamsPupilV1", "IdealAplanaticObjectiveV1", "JonesPolarizationV1",
    "LinearPolarizationV1", "UniformPupilV1", "UnpolarizedV1", "WindowV1", "ZernikePhaseV1", "ZernikeTermV1",
]
