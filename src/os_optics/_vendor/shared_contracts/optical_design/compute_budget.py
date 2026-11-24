"""`ComputeBudgetV1`: the customer's local computation budget.

A run's tracing limits are resource and termination controls, not physical limits: a design that stops at
`max_interactions_per_path` is not mis-designed, its trace was simply not allowed to follow more round trips.
They are therefore the *customer's* settings, chosen for the machine the run happens on, and they live with
the Project so that every path -- Python, CLI, workspace, browser Designer -- carries the same numbers and
the Engine input hash (and with it a run's freshness) binds them.

Every field is optional and `None` means "the Engine's default", never a copied literal: the defaults stay in
`opensci_engine.config.NumericalConfig`, which `NUMERICAL_CONFIG_VERSION` versions. `payload()` omits `None`
fields, so a Project that states no budget keeps its exact bytes and osc1 hash (the hash evolution rule in
`project.py`).

The limits and levels that decide whether or how a value is reported (the power floor, the level below which light is
left out, the validity limits of a model, the match windows of a power rating) are the Project's ``numerical_settings``
(``NumericalSettingsV1``), a sibling of this block: a number computed under a setting other than the Engine's default
names that setting on the quantity it qualifies. The result-hash quantization and the reported resolutions are settings
of neither block.

Every budget's upper bound is the largest integer JSON preserves exactly, never a capacity cap: the size of a run is
bounded by the caller's run monitor (its memory and artifact limits and its preflight estimate), not by this contract.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from ..base import OmitAbsentFields, StrictContractModel

#: JSON's interoperable integer range: browser Number and Python must preserve every budget exactly in
#: Project identity and server re-lowering. This is a representation ceiling, not a resource recommendation.
MAX_EXACT_JSON_INTEGER = 2**53 - 1
BUDGET_RANGES: dict[str, tuple[int, int]] = {
    name: (1, MAX_EXACT_JSON_INTEGER)
    for name in ("max_interactions_per_path", "max_paths", "max_trace_steps", "max_interference_pairs",
                 "max_return_analysis_work", "pupil_ring_count", "pupil_points_per_ring", "max_scatter_origins",
                 "max_mesh_patches")
}


class ComputeBudgetV1(OmitAbsentFields, StrictContractModel):
    """Which local computation budgets this Project overrides, and with what."""

    # Serialized only when given: a document without them keeps its earlier payload and hash.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("pupil_ring_count", "pupil_points_per_ring", "max_scatter_origins",
                                                  "max_mesh_patches")

    max_interactions_per_path: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["max_interactions_per_path"][0], le=BUDGET_RANGES["max_interactions_per_path"][1],
        description="Physical surface interactions one beam path may meet before it stops (observation planes are "
        "virtual and never count). Raise it to follow more round trips of a cavity or a long surface chain to a "
        "downstream monitor. The Engine default applies when absent. Values are finite positive integers; the maximum "
        "is the largest integer JSON preserves exactly across Python and browser runtimes.",
    )
    max_paths: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["max_paths"][0], le=BUDGET_RANGES["max_paths"][1],
        description="Beam paths one run may trace (source spectral lines and every branch a splitting surface "
        "creates), the strongest first. Beyond it the weakest source lines and branches are not traced and the run "
        "reports MAX_PATHS with their count and power.",
    )
    max_trace_steps: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["max_trace_steps"][0], le=BUDGET_RANGES["max_trace_steps"][1],
        description="Chief-ray steps over all paths of one run (a path takes one step to each surface it reaches). "
        "It bounds a run's time, memory and run files; a path stopped by it reports TRACE_BUDGET in the MAX_PATHS "
        "issue.",
    )
    max_interference_pairs: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["max_interference_pairs"][0], le=BUDGET_RANGES["max_interference_pairs"][1],
        description="Beam pairs analysed per observation plane. A plane with more reports MAX_BEAM_PAIRS and the "
        "pairs beyond the cap are not analysed.",
    )
    max_return_analysis_work: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["max_return_analysis_work"][0], le=BUDGET_RANGES["max_return_analysis_work"][1],
        description="Work of the return-light analysis over the whole run, in work units (about 4 microseconds each on "
        "a desktop CPU): the screen of the surfaces that may return light left out of the trace, pairs, internal folds of "
        "plane-faced solids and closed returns of three or more surfaces. It runs only for a wavelength whose left-out "
        "light a power or exposure upper bound reads; the wavelengths share the budget in the order the run reads them. "
        "An analysis that reaches it stops; the upper bounds that depend on it are withheld and the run reports "
        "RETURN_ANALYSIS_BUDGET. Lower it to keep a large scene quick; raise it to obtain those bounds. The Engine "
        "default applies when absent.",
    )
    pupil_ring_count: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["pupil_ring_count"][0], le=BUDGET_RANGES["pupil_ring_count"][1],
        description="Equal-power rings of pupil rays each Gaussian beam source launches (Engine default 8). The pupil rays "
        "give a beam's spot, angles of incidence, aperture use and per-ray paths, and the power of a beam without a valid "
        "Gaussian state that an edge cuts (geometric pupil-sample basis, power_sampling_resolution_W). Rays per beam: "
        "1 + rings x points per ring + 8; the trace's time, memory and full ray data grow in proportion. The sampling "
        "resolution of a cut power shrinks about in proportion to rings (with points per ring raised in step): doubling "
        "both halves it and quadruples the rays. The Engine default applies when absent.",
    )
    pupil_points_per_ring: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["pupil_points_per_ring"][0], le=BUDGET_RANGES["pupil_points_per_ring"][1],
        description="Points on each pupil ring (Engine default 12); see pupil_ring_count for what the pupil rays give and what "
        "they cost. Raise it with pupil_ring_count (in proportion, e.g. 16 x 24 or 32 x 48) to narrow the sampling "
        "resolution of a cut power. The Engine default applies when absent.",
    )
    max_scatter_origins: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["max_scatter_origins"][0], le=BUDGET_RANGES["max_scatter_origins"][1],
        description="Scattering origins whose scattered light the run traces as rays, the strongest first (Engine default "
        "64). The light of the origins beyond it stays booked as scattered at its surface and the planes it may reach "
        "report SCATTERED_LIGHT_NOT_FOLLOWED. The scattered rays share the trace's step budget, so time and memory grow "
        "about in proportion. The Engine default applies when absent.",
    )
    max_mesh_patches: int | None = Field(
        None, strict=True, ge=BUDGET_RANGES["max_mesh_patches"][0], le=BUDGET_RANGES["max_mesh_patches"][1],
        description="Traced patches one faceted surface may have (Engine default 1024); a surface with more is refused. "
        "Each patch is intersected on every crossing, so the trace's time grows about in proportion. The Engine default "
        "applies when absent.",
    )

    def overrides(self) -> dict[str, int]:
        """The fields this budget states, in `BUDGET_RANGES` order (empty: the Engine defaults throughout)."""

        return {name: int(getattr(self, name)) for name in BUDGET_RANGES if getattr(self, name) is not None}


def budget_field_range(name: str) -> tuple[int, int]:
    """`(lowest, highest)` a Project may state for budget field `name`."""

    return BUDGET_RANGES[name]


def check_compute_budget(value: Any) -> dict[str, int]:
    """`ComputeBudgetV1.overrides` of a document field, tolerating `None` (no budget stated)."""

    if value is None:
        return {}
    budget = value if isinstance(value, ComputeBudgetV1) else ComputeBudgetV1.model_validate(value)
    return budget.overrides()


__all__ = ["BUDGET_RANGES", "ComputeBudgetV1", "budget_field_range", "check_compute_budget"]

