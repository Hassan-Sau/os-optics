"""Checker identity and the default thresholds of the local assembly layout check.

Every value is recorded in each result (``AssemblyLocalValidationV1.tolerances``), so a result states the
thresholds it was produced with: well above the float32 rounding of the part meshes (~1e-5 mm) and the chord error
of their 32-segment circles, and well below any interference a person would call a collision.

Every threshold is a caller setting (``tolerances_with``: the CLI's ``--tolerance NAME=VALUE``, the JSON API's
``options.tolerances``, ``CommercialClient.validate_assembly(tolerances=)``); ``max_total_triangles`` is the checker's
work limit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.assembly_preview import CheckerTolerancesV1

#: Version of the checker's semantics (checks, rules, thresholds). Bump when a result could change.
ASSEMBLY_CHECKER_VERSION = "2.2.0"  # 2.0.0: result schema 2.0 (summary, PART_COLLISION, collision / detachment tolerances); 2.1.0: reserved spaces of customer-supplied components (RESERVED_SPACE_INTRUSION); 2.2.0: the work limit is UNKNOWN, a pair measured with skipped columns keeps its metrics (COLUMNS_CAST / COLUMNS_SKIPPED) and a proven collision, an open or inward-facing mesh is checked wherever its vertices suffice

DEFAULT_TOLERANCES = CheckerTolerancesV1(
    overlap_noise_mm=0.1,  # thinner overlap is contact: coplanar faces, board hole markers (0.05 mm proud)
    overlap_noise_volume_mm3=0.1,
    collision_penetration_mm=1.0,
    collision_volume_mm3=1.0,
    buried_fraction=0.5,  # an overlap burying half the smaller part is a collision, however thin that part is
    engagement_radial_allowance_mm=0.5,  # nominal interface radius + this; covers 32-gon meshes and fits
    engagement_axial_allowance_mm=0.5,
    fastener_engagement_diameters=3.0,  # a threaded stud may reach 3 nominal diameters into its mate
    mate_position_tolerance_mm=0.05,
    mate_angle_tolerance_deg=0.05,
    detachment_distance_mm=2.0,
    detachment_angle_deg=2.0,
    board_outline_tolerance_mm=0.5,
    quaternion_norm_tolerance=1e-6,
    sample_spacing_mm=0.25,
    max_columns_per_axis=64,
    max_total_triangles=100_000,  # pure-Python budget
)


#: The tolerance names a caller may set (every ``CheckerTolerancesV1`` field).
TOLERANCE_NAMES: tuple[str, ...] = tuple(CheckerTolerancesV1.model_fields)


def tolerances_with(values: Mapping[str, Any] | CheckerTolerancesV1 | None) -> CheckerTolerancesV1:
    """``DEFAULT_TOLERANCES`` with the caller's value for each named field, re-validated by the model (its ordering rule
    included). Raises ``ValueError`` naming an unknown field or the first value the model refuses."""

    if values is None:
        return DEFAULT_TOLERANCES
    if isinstance(values, CheckerTolerancesV1):
        return values
    if not isinstance(values, Mapping):
        raise ValueError("tolerances are a {name: value} object")
    unknown = sorted(str(k) for k in values if k not in CheckerTolerancesV1.model_fields)
    if unknown:
        raise ValueError(f"unknown tolerance {', '.join(unknown)}; the tolerances are {', '.join(TOLERANCE_NAMES)}")
    not_numbers = sorted(str(k) for k, v in values.items() if isinstance(v, bool) or not isinstance(v, (int, float)))
    if not_numbers:
        raise ValueError(f"tolerance {', '.join(not_numbers)}: a number is required")
    try:
        return CheckerTolerancesV1.model_validate({**DEFAULT_TOLERANCES.model_dump(), **values})
    except ValidationError as exc:
        error = exc.errors(include_input=False, include_url=False)[0]
        where = ".".join(str(p) for p in error.get("loc", ()))
        raise ValueError(f"tolerance {where}: {error.get('msg', 'invalid')}" if where else str(error.get("msg", "invalid"))) from None


def tolerance_argument(text: str) -> tuple[str, int | float]:
    """One ``NAME=VALUE`` argument (a count for an integer field, a number otherwise); ``ValueError`` if malformed."""

    name, sep, raw = text.partition("=")
    name = name.strip()
    field = CheckerTolerancesV1.model_fields.get(name)
    if not sep or field is None:
        raise ValueError(f"takes NAME=VALUE with NAME one of {', '.join(TOLERANCE_NAMES)}, got {text!r}")
    try:
        return name, (int(raw) if field.annotation is int else float(raw))
    except ValueError:
        raise ValueError(f"{name} takes {'a whole number' if field.annotation is int else 'a number'}, got {raw!r}") from None
