"""`NumericalSettingsV1`: the limits and levels that decide whether or how a value is reported, as Project state.

The local computation budgets (``compute_budget``) decide how much work a run may do. These settings decide something
else: the validity limits of a model (how far the paraxial Gaussian model, a second-harmonic model or a published
design point of a coating is taken), the levels below which light is left out of the trace, and the windows within which
a power rating's test conditions count as matched. Each has an Engine default; a Project states only the ones it
changes. A number computed under a setting other than the default says so on the quantity it qualifies (the setting's
name in its value basis) and the run records the value in force (``manifest.numerical_tolerances``).

Every field is optional and ``None`` means "the Engine's default", never a copied literal: the defaults stay in the
Engine's ``NumericalConfig``, which its ``NUMERICAL_CONFIG_VERSION`` versions. Absent fields are omitted from the payload,
so a Project that states no setting keeps its exact bytes and osc1 hash (the hash evolution rule in ``project.py``).
The settings are part of the Engine input, so a run made with one value is not current for a Project that states another.

Each field is named as the Engine's ``NumericalConfig`` field it sets and accepts the same range (``SETTING_NAMES`` maps
the fields of ``design_point`` to theirs). Not settable: the trace's reach beyond the scene (a declared numerical choice
recorded with the run), the result-hash and cross-runtime quantization, the reported resolutions and the defaults'
version.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

from pydantic import BeforeValidator, Field

from ..base import OmitAbsentFields, StrictContractModel
from .compute_budget import MAX_EXACT_JSON_INTEGER


def _not_bool(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("a number is required, not a boolean")
    return value


#: A finite real number written as a JSON number (an integer is accepted, a boolean is not).
Number = Annotated[float, BeforeValidator(_not_bool)]


class SingleAngleTolerancesV1(OmitAbsentFields, StrictContractModel):
    """Per coating kind: how far (deg) from a coating's single stated design angle of incidence the published
    coefficients are used as they are."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("metallic_reflector", "dielectric_high_reflector", "anti_reflection",
                                                  "beamsplitter")

    metallic_reflector: Number | None = Field(
        None, ge=0, le=90, description="Metallic reflectors (Engine default 5 deg).")
    dielectric_high_reflector: Number | None = Field(
        None, ge=0, le=90, description="Dielectric high reflectors (Engine default 3 deg).")
    anti_reflection: Number | None = Field(
        None, ge=0, le=90, description="Anti-reflection coatings (Engine default 3 deg).")
    beamsplitter: Number | None = Field(
        None, ge=0, le=90, description="Beamsplitter coatings (Engine default 2 deg).")


class DesignPointTolerancesV1(OmitAbsentFields, StrictContractModel):
    """How far from a single stated design point of published model data (one wavelength, one angle of incidence) the
    published values are used as they are. Beyond it the run reports the use as outside the model's stated range."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("single_wavelength_rel", "multi_order_retarder_rel", "single_angle_deg",
                                                  "interference_shift_fraction", "interference_angle_max_deg")

    single_wavelength_rel: Number | None = Field(
        None, ge=0, lt=1, description="Relative distance from a single stated design wavelength of a coating or a zero-order "
        "retarder within which its published values are used (Engine default 0.005).")
    multi_order_retarder_rel: Number | None = Field(
        None, ge=0, lt=1, description="The same for a multi-order retarder, whose retardance changes with wavelength m + 1 "
        "times faster (Engine default 0.0005).")
    single_angle_deg: SingleAngleTolerancesV1 | None = Field(
        None, description="Per coating kind, the distance in degrees from a single stated design angle of incidence within "
        "which the coating's published values are used.")
    interference_shift_fraction: Number | None = Field(
        None, gt=0, le=1, description="For an interference filter (dichroic, bandpass, edge-pass, notch) with a single "
        "design angle: the share of the narrowest spectral width its bands leave that the angle-shift of its spectrum may "
        "reach (Engine default 0.25).")
    interference_angle_max_deg: Number | None = Field(
        None, ge=0, le=90, description="The largest angle tolerance an interference filter gets from that rule (Engine "
        "default 3 deg).")


#: The Engine ``NumericalConfig`` field each ``design_point`` field sets (a dotted path below ``design_point``).
DESIGN_POINT_SETTINGS: dict[str, str] = {
    "single_wavelength_rel": "design_point_wavelength_rel",
    "multi_order_retarder_rel": "design_point_retarder_wavelength_rel",
    "single_angle_deg.metallic_reflector": "design_point_angle_metallic_reflector_deg",
    "single_angle_deg.dielectric_high_reflector": "design_point_angle_dielectric_high_reflector_deg",
    "single_angle_deg.anti_reflection": "design_point_angle_anti_reflection_deg",
    "single_angle_deg.beamsplitter": "design_point_angle_beamsplitter_deg",
    "interference_shift_fraction": "design_point_interference_shift_fraction",
    "interference_angle_max_deg": "design_point_interference_angle_max_deg",
}


class NumericalSettingsV1(OmitAbsentFields, StrictContractModel):
    """The validity limits, levels and match windows this Project states (module docstring). Absent: the Engine's
    defaults."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("power_epsilon_W", "negligible_power_fraction",
        "gaussian_warn_divergence_rad", "gaussian_scope_divergence_rad", "gaussian_max_w_over_radius",
        "gaussian_max_aoi_deg", "gaussian_max_ray_spread_ratio", "gaussian_clip_warn_fraction",
        "gaussian_clip_scope_fraction", "aperture_quad_sector_nodes", "capture_min_power_fraction",
        "interference_min_crossing_deg", "interference_wavelength_tol_nm", "interference_min_overlap",
        "nonlinear_depletion_rel_tol", "nonlinear_focusing_rel_tol", "nonlinear_quadrature_rel_tol",
        "nonlinear_harmonic_gaussian_rel_tol", "nonlinear_beam_quality_tol", "nonlinear_linewidth_rel_tol",
        "rating_wavelength_tol_rel", "rating_pulse_duration_tol_rel", "rating_repetition_rate_tol_rel",
        "rating_beam_diameter_tol_rel", "rating_polarization_fraction_tol", "rating_unlocated_light_share",
        "design_point")

    # ---- what the trace follows -----------------------------------------------------------------------------------
    power_epsilon_W: Number | None = Field(
        None, ge=0, description="Absolute power floor (W): a path whose power is at or below it ends there and its power is "
        "booked as ended at the floor. Lower it to follow a source weaker than the default floor (1e-15 W); 0 sets no "
        "absolute floor.")
    negligible_power_fraction: Number | None = Field(
        None, ge=0, lt=1, description="Share of a source line's power below which light is not followed, probed or listed "
        "(Engine default 1e-9). Results that read 'nothing left out' mean nothing above this level.")

    # ---- the paraxial Gaussian model ---------------------------------------------------------------------------------
    gaussian_warn_divergence_rad: Number | None = Field(
        None, gt=0, description="Far-field half angle (rad) above which a Gaussian state is reported with a warning (Engine "
        "default 0.10).")
    gaussian_scope_divergence_rad: Number | None = Field(
        None, gt=0, description="Far-field half angle (rad) above which the paraxial Gaussian model is out of its validity "
        "range (Engine default 0.30).")
    gaussian_max_w_over_radius: Number | None = Field(
        None, gt=0, description="Beam radius over the surface's radius of curvature above which the Gaussian transfer is "
        "out of its validity range (Engine default 0.2).")
    gaussian_max_aoi_deg: Number | None = Field(
        None, gt=0, lt=90, description="Angle of incidence (deg) on a tilted surface above which the Gaussian transfer is out "
        "of its validity range (Engine default 60).")
    gaussian_max_ray_spread_ratio: Number | None = Field(
        None, gt=1, description="Pupil-ray spread over the aberration-free spread of the arrival's Gaussian envelope above "
        "which the beam is aberration dominated (Engine default 2.0).")
    gaussian_clip_warn_fraction: Number | None = Field(
        None, ge=0, description="Share of a Gaussian beam's power cut by an aperture above which the cut is reported with "
        "a warning (Engine default 1e-3).")
    gaussian_clip_scope_fraction: Number | None = Field(
        None, gt=0, description="Share of a Gaussian beam's power cut by an aperture above which its Gaussian state is out "
        "of its validity range (Engine default 0.10).")
    aperture_quad_sector_nodes: int | None = Field(
        None, strict=True, ge=4, le=MAX_EXACT_JSON_INTEGER,
        description="Gauss-Legendre nodes per sector of the off-axis aperture truncation integral (Engine default 48); more "
        "nodes refine the cut power and cost time: building the rule grows with the square of the count (about 0.5 s at "
        "1024 and 4 s at 4096 nodes on a desktop CPU). A run's monitor stops a long build (cancel, memory limit), and a count "
        "whose rule the machine cannot hold is refused by name. The maximum is the largest integer JSON preserves exactly.")

    # ---- capture and two-beam interference ---------------------------------------------------------------------------
    capture_min_power_fraction: Number | None = Field(
        None, gt=0, description="Detected over available power below which a beam counts as lost (Engine default 1e-3).")
    interference_min_crossing_deg: Number | None = Field(
        None, gt=0, description="Crossing angle (deg) below which two beams have no defined fringe period (Engine default "
        "1e-6).")
    interference_wavelength_tol_nm: Number | None = Field(
        None, gt=0, description="Wavelength difference (nm) within which two arrivals interfere (Engine default 1e-9).")
    interference_min_overlap: Number | None = Field(
        None, gt=0, description="Spatial overlap ratio below which two beams count as not overlapping (Engine default 1e-3).")

    # ---- second-harmonic generation ----------------------------------------------------------------------------------
    nonlinear_depletion_rel_tol: Number | None = Field(
        None, gt=0, lt=1, description="Relative overstatement of the harmonic power by the undepleted-pump model above which "
        "the conversion is out of its validity range (Engine default 0.02).")
    nonlinear_focusing_rel_tol: Number | None = Field(
        None, gt=0, lt=1, description="Relative difference between the near-field and the focused conversion above which the "
        "near-field result is out of its validity range (Engine default 0.02).")
    nonlinear_quadrature_rel_tol: Number | None = Field(
        None, gt=0, lt=1e-2, description="Relative agreement at which the overlap integral of the focused conversion stops "
        "refining (Engine default 1e-7).")
    nonlinear_harmonic_gaussian_rel_tol: Number | None = Field(
        None, gt=0, lt=1, description="Relative growth of the harmonic's width by walk-off above which the harmonic is traced "
        "without a Gaussian state (Engine default 0.02).")
    nonlinear_beam_quality_tol: Number | None = Field(
        None, gt=0, lt=1, description="Excess of the fundamental's M^2 over 1 above which the conversion is out of its "
        "validity range (Engine default 0.05).")
    nonlinear_linewidth_rel_tol: Number | None = Field(
        None, gt=0, lt=1, description="Relative difference between the line-averaged and the line-centre phase-matching "
        "factor above which the conversion is out of its validity range (Engine default 0.02).")

    # ---- power ratings -----------------------------------------------------------------------------------------------
    rating_wavelength_tol_rel: Number | None = Field(
        None, ge=0, le=1, description="Relative distance from a rating's test wavelength within which the light's wavelength "
        "matches it (Engine default 0.02).")
    rating_pulse_duration_tol_rel: Number | None = Field(
        None, ge=0, le=1, description="Relative distance from a rating's test pulse duration within which the light's "
        "duration matches it (Engine default 0.10).")
    rating_repetition_rate_tol_rel: Number | None = Field(
        None, ge=0, le=1, description="Relative distance from a rating's test repetition rate within which the light's rate "
        "matches it (Engine default 0.10).")
    rating_beam_diameter_tol_rel: Number | None = Field(
        None, ge=0, le=1, description="Relative distance from a rating's test beam diameter within which the beam's diameter "
        "matches it (Engine default 0.05).")
    rating_polarization_fraction_tol: Number | None = Field(
        None, ge=0, le=1, description="Largest share of a beam's power outside a rating's stated polarization for the "
        "polarization to match (Engine default 0.01).")
    rating_unlocated_light_share: Number | None = Field(
        None, ge=0, le=1, description="Largest share of the light a surface can receive that light of unknown size, place or "
        "state may carry for a rating condition to be verified (Engine default 0.01).")

    # ---- published design points ---------------------------------------------------------------------------------------
    design_point: DesignPointTolerancesV1 | None = Field(
        None, description="How far from a single stated design wavelength or angle of published model data its values are "
        "used as they are.")

    def overrides(self) -> dict[str, float | int]:
        """The settings this block states, keyed by the Engine ``NumericalConfig`` field each sets (``design_point``
        flattened through ``DESIGN_POINT_SETTINGS``), in field order. Empty: the Engine defaults throughout."""

        stated = self.model_dump(mode="python", exclude_none=True)
        point = stated.pop("design_point", {})
        values: dict[str, float | int] = {name: value for name, value in stated.items()}
        angles = point.pop("single_angle_deg", {})
        for name, value in point.items():
            values[DESIGN_POINT_SETTINGS[name]] = value
        for name, value in angles.items():
            values[DESIGN_POINT_SETTINGS[f"single_angle_deg.{name}"]] = value
        return values


#: Every Engine ``NumericalConfig`` field a Project may set, in contract order (``design_point`` flattened).
SETTING_NAMES: tuple[str, ...] = (
    *(name for name in NumericalSettingsV1.model_fields if name != "design_point"),
    *DESIGN_POINT_SETTINGS.values(),
)

#: The Project path of each setting (``numerical_settings.<field>``, ``numerical_settings.design_point.<path>``).
SETTING_PATHS: dict[str, str] = {
    **{name: f"numerical_settings.{name}" for name in NumericalSettingsV1.model_fields if name != "design_point"},
    **{engine: f"numerical_settings.design_point.{path}" for path, engine in DESIGN_POINT_SETTINGS.items()},
}


def merge_numerical_settings(base: Any, stated: Any) -> dict[str, Any] | None:
    """The JSON of ``base`` with every field ``stated`` gives in place of the base's, nested blocks merged field by field
    (a configuration's settings over its Project's). ``None`` when neither states anything."""

    def plain(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, NumericalSettingsV1):
            return value.model_dump(mode="json", exclude_none=True)
        return NumericalSettingsV1.model_validate(value).model_dump(mode="json", exclude_none=True)

    def merged(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        out = dict(a)
        for key, value in b.items():
            out[key] = merged(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
        return out

    result = merged(plain(base), plain(stated))
    return result or None


def check_numerical_settings(value: Any) -> dict[str, float | int]:
    """``NumericalSettingsV1.overrides`` of a document field, tolerating ``None`` (no setting stated)."""

    if value is None:
        return {}
    settings = value if isinstance(value, NumericalSettingsV1) else NumericalSettingsV1.model_validate(value)
    return settings.overrides()


__all__ = ["DESIGN_POINT_SETTINGS", "SETTING_NAMES", "SETTING_PATHS", "DesignPointTolerancesV1", "NumericalSettingsV1",
           "SingleAngleTolerancesV1", "check_numerical_settings", "merge_numerical_settings"]
