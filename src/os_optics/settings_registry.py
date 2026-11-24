"""The registry of caller settings: every ``NumericalConfig`` field a Project may set, where it is set and what it means.

A Project sets two kinds of value. Its ``compute_budget`` holds the work limits of a run (how much the trace, the return
analysis or the scatter trace may do); its ``numerical_settings`` holds the limits and levels that decide whether or how
a value is reported (validity limits of a model, the levels below which light is left out, the match windows of a power
rating, the tolerances of published design points). Each is a ``NumericalConfig`` field of the same name (a
``design_point`` field maps to its ``design_point_*`` field), its default is the ``NumericalConfig`` default, and the
value in force is recorded with every run in ``manifest.numerical_tolerances``.

``SETTINGS`` is the one place that lists them: per name its ``home`` (the Project block), its Project ``path``, its
default, unit, ``direction`` (which way widens what the Engine accepts or reports) and a one-line ``meaning``.
The headless capabilities and settings reference derive from it. ``NOT_SETTABLE`` names the fields a Project may never set,
with the reason.

Recording rule: a quantity computed under a setting names it -- ``ValidityCheck.setting`` for a validity limit (written
when the case is outside the limit, or when the limit in force is not the default), ``Omitted.setting`` for a level that
decides what is left out -- and every name a result carries is a key of its run's ``manifest.numerical_tolerances``
(``unrecorded_settings`` checks it).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._vendor.shared_contracts.optical_design.compute_budget import BUDGET_RANGES
from ._vendor.shared_contracts.optical_design.numerical_settings import SETTING_NAMES, SETTING_PATHS
from .config import DEFAULT_NUMERICAL_CONFIG


class SettingHome(StrEnum):
    """The Project block a setting is stated in."""

    #: ``ProjectDocumentV2.compute_budget`` (and per configuration): a work limit of the run.
    COMPUTE_BUDGET = "compute_budget"
    #: ``ProjectDocumentV2.numerical_settings`` (and per configuration): a limit or level a reported value depends on.
    NUMERICAL_SETTINGS = "numerical_settings"


class SettingDirection(StrEnum):
    """Which way of changing a setting widens what the Engine accepts or reports."""

    #: A larger value accepts more: a validity limit or match window that admits more cases, a level that leaves more out.
    RAISE_ACCEPTS_MORE = "RAISE_ACCEPTS_MORE"
    #: A smaller value accepts more: a floor or minimum below which a value is not computed or a case not admitted.
    LOWER_ACCEPTS_MORE = "LOWER_ACCEPTS_MORE"
    #: A work limit: a larger value lets the run compute more of the result and costs time and memory.
    WORK_LIMIT = "WORK_LIMIT"
    #: A resolution of a computation: a finer value costs time and narrows the numerical error, and admits nothing more.
    ACCURACY_COST = "ACCURACY_COST"


@dataclass(frozen=True)
class Setting:
    """One caller setting (module docstring)."""

    name: str
    home: SettingHome
    path: str
    unit: str
    direction: SettingDirection
    meaning: str

    @property
    def default(self) -> float | int:
        """The Engine default (``NumericalConfig``)."""

        return getattr(DEFAULT_NUMERICAL_CONFIG, self.name)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "home": self.home.value, "path": self.path, "default": self.default, "unit": self.unit,
                "direction": self.direction.value, "meaning": self.meaning}


_R, _L, _W, _A = (SettingDirection.RAISE_ACCEPTS_MORE, SettingDirection.LOWER_ACCEPTS_MORE, SettingDirection.WORK_LIMIT,
                  SettingDirection.ACCURACY_COST)

#: ``name -> (unit, direction, meaning)`` of every Project-settable ``NumericalConfig`` field, budgets first.
_TABLE: dict[str, tuple[str, SettingDirection, str]] = {
    # ---- compute_budget -------------------------------------------------------------------------------------------
    "max_interactions_per_path": ("1", _W, "physical surface interactions one beam path may meet before it stops"),
    "max_paths": ("1", _W, "beam paths one run traces, the strongest first"),
    "max_trace_steps": ("1", _W, "chief-ray steps over all paths of one run"),
    "max_interference_pairs": ("1", _W, "beam pairs analysed per observation plane"),
    "max_return_analysis_work": ("1", _W, "work units of the return-light analysis over the whole run"),
    "pupil_ring_count": ("1", _A, "equal-power rings of pupil rays per Gaussian beam"),
    "pupil_points_per_ring": ("1", _A, "points on each pupil ring"),
    "max_scatter_origins": ("1", _W, "scattering origins whose scattered light is traced as rays, the strongest first"),
    "max_mesh_patches": ("1", _W, "traced planar patches one faceted surface may have"),
    # ---- numerical_settings ---------------------------------------------------------------------------------------
    "power_epsilon_W": ("W", _L, "absolute power at or below which a path ends"),
    "negligible_power_fraction": ("1", _R, "share of a source line's power below which light is not followed, probed or listed"),
    "gaussian_warn_divergence_rad": ("rad", _R, "far-field half angle above which a Gaussian state warns"),
    "gaussian_scope_divergence_rad": ("rad", _R, "far-field half angle above which the paraxial Gaussian model is out of scope"),
    "gaussian_max_w_over_radius": ("1", _R, "beam radius over surface radius of curvature above which Gaussian transfer is out of scope"),
    "gaussian_max_aoi_deg": ("deg", _R, "angle of incidence above which Gaussian transfer on a tilted surface is out of scope"),
    "gaussian_max_ray_spread_ratio": ("1", _R, "pupil-ray spread over the Gaussian envelope's above which a beam is aberration dominated"),
    "gaussian_clip_warn_fraction": ("1", _R, "share of a Gaussian beam's power cut by an aperture above which the cut warns"),
    "gaussian_clip_scope_fraction": ("1", _R, "share of a Gaussian beam's power cut by an aperture above which its Gaussian state is out of scope"),
    "aperture_quad_sector_nodes": ("1", _A, "quadrature nodes per sector of the off-axis aperture truncation integral"),
    "capture_min_power_fraction": ("1", _L, "detected over available power below which a beam counts as lost"),
    "interference_min_crossing_deg": ("deg", _L, "crossing angle below which two beams have no defined fringe period"),
    "interference_wavelength_tol_nm": ("nm", _R, "wavelength difference within which two arrivals interfere"),
    "interference_min_overlap": ("1", _L, "spatial overlap ratio below which two beams count as not overlapping"),
    "nonlinear_depletion_rel_tol": ("1", _R, "overstatement of harmonic power by the undepleted pump above which conversion is out of scope"),
    "nonlinear_focusing_rel_tol": ("1", _R, "difference of near-field and focused conversion above which the near-field result is out of scope"),
    "nonlinear_quadrature_rel_tol": ("1", _A, "relative agreement at which the focused-conversion overlap integral stops refining"),
    "nonlinear_harmonic_gaussian_rel_tol": ("1", _R, "walk-off growth of the harmonic's width above which it has no Gaussian state"),
    "nonlinear_beam_quality_tol": ("1", _R, "excess of the fundamental's M^2 over 1 above which conversion is out of scope"),
    "nonlinear_linewidth_rel_tol": ("1", _R, "line-averaged against line-centre phase matching difference above which conversion is out of scope"),
    "rating_wavelength_tol_rel": ("1", _R, "relative window around a rating's test wavelength that counts as matched"),
    "rating_pulse_duration_tol_rel": ("1", _R, "relative window around a rating's test pulse duration that counts as matched"),
    "rating_repetition_rate_tol_rel": ("1", _R, "relative window around a rating's test repetition rate that counts as matched"),
    "rating_beam_diameter_tol_rel": ("1", _R, "relative window around a rating's test beam diameter that counts as matched"),
    "rating_polarization_fraction_tol": ("1", _R, "share of a beam's power outside a rating's polarization that still matches"),
    "rating_unlocated_light_share": ("1", _R, "share of a surface's light that unlocated light may carry for a rating condition to be verified"),
    "design_point_wavelength_rel": ("1", _R, "relative distance from a single design wavelength within which published values are used"),
    "design_point_retarder_wavelength_rel": ("1", _R, "the same for a multi-order retarder"),
    "design_point_angle_metallic_reflector_deg": ("deg", _R, "distance from a metallic reflector's single design angle within which its values are used"),
    "design_point_angle_dielectric_high_reflector_deg": ("deg", _R, "distance from a dielectric high reflector's single design angle within which its values are used"),
    "design_point_angle_anti_reflection_deg": ("deg", _R, "distance from an anti-reflection coating's single design angle within which its values are used"),
    "design_point_angle_beamsplitter_deg": ("deg", _R, "distance from a beamsplitter coating's single design angle within which its values are used"),
    "design_point_interference_shift_fraction": ("1", _R, "share of an interference filter's narrowest band width its angle shift may reach"),
    "design_point_interference_angle_max_deg": ("deg", _R, "largest single-design-angle tolerance of an interference filter"),
}


def _home(name: str) -> SettingHome:
    return SettingHome.COMPUTE_BUDGET if name in BUDGET_RANGES else SettingHome.NUMERICAL_SETTINGS


#: Every caller setting, budgets first, then ``numerical_settings`` in contract order.
SETTINGS: dict[str, Setting] = {
    name: Setting(name=name, home=_home(name),
                  path=f"compute_budget.{name}" if name in BUDGET_RANGES else SETTING_PATHS[name],
                  unit=_TABLE[name][0], direction=_TABLE[name][1], meaning=_TABLE[name][2])
    for name in (*BUDGET_RANGES, *SETTING_NAMES)
}

#: ``NumericalConfig`` fields a Project never sets, with the reason.
NOT_SETTABLE: dict[str, str] = {
    "config_version": "the version of the defaults set, not a setting",
    "laboratory_reach_mm": "a declared numerical choice of the run, fixed and recorded with it",
    "hash_float_sig_digits": "the result-hash quantization: identity, not physics",
    "hash_float_abs_floor": "the result-hash quantization: identity, not physics",
    "cross_runtime_rel_tol": "the cross-runtime comparison tolerance: identity, not physics",
    "cross_runtime_abs_tol": "the cross-runtime comparison tolerance: identity, not physics",
    "first_order_unitless_resolution": "a reported resolution, chosen so a result is identical on every runtime",
    "first_order_b_resolution_mm": "a reported resolution, chosen so a result is identical on every runtime",
    "first_order_c_resolution_per_mm": "a reported resolution, chosen so a result is identical on every runtime",
    "power_ledger_fraction_resolution": "a reported resolution, chosen so a result is identical on every runtime",
}


def setting(name: str) -> Setting:
    """The registry entry of ``name`` (``KeyError`` for a name a Project cannot set)."""

    return SETTINGS[name]


def named_settings(document: Any) -> set[str]:
    """Every setting name a result document (a model or its JSON) carries: the ``setting`` of each ``Omitted`` and
    ``ValidityCheck`` anywhere in it."""

    if hasattr(document, "model_dump"):
        document = document.model_dump(mode="json")
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("setting")
            if isinstance(name, str) and ("part" in node or "parameter" in node):
                found.add(name)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    return found


def unrecorded_settings(result: Any) -> tuple[str, ...]:
    """The setting names a ``ValidationResult`` (or its JSON) carries that are not keys of its run-level settings record,
    ``manifest.numerical_tolerances`` (sorted; empty when the recording rule holds). A result without a manifest records
    nothing, so every name it carries is unrecorded."""

    document = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    recorded = set(((document.get("manifest") or {}).get("numerical_tolerances") or {}))
    return tuple(sorted(named_settings(document) - recorded))


__all__ = ["NOT_SETTABLE", "SETTINGS", "Setting", "SettingDirection", "SettingHome", "named_settings", "setting",
           "unrecorded_settings"]
