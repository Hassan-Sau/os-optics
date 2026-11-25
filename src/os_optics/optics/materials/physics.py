"""Material physics of a published or declared material at one vacuum wavelength: the Engine's own evaluation, for selection
and reporting (``os-optics catalog material``), with nothing re-implemented.

A ``PublicMaterialV1`` (catalog) or ``ProjectModelMaterialV1`` (a Project's copy) is lowered exactly as a run lowers it
(``local.compile.engine_material``, at the stated temperature) and evaluated by the evaluator the tracer uses
(``index.build_evaluator``: the one material-evaluation rule) and the group-dispersion of ``group.py``:

* ``material_point(material, wavelength_nm, temperature_c=20)``: the absolute index, group index and group-delay dispersion,
  the law's wavelength, the convention, and why something is not available (outside the vacuum band or the temperature
  range: ``reason``; a measured table's derivatives: ``derivative_limitation``);
* ``bulk_alpha_per_mm`` / ``internal_transmittance``: the bulk absorption coefficient and the internal transmittance
  ``exp(-alpha L)`` of a thickness, from the material's ``bulk_loss`` (or its scalar ``absorption_coefficient_per_mm``) --
  ``None`` with a reason where the loss is not stated or not established, never 1 for missing data.

Wavelengths are vacuum wavelengths (nm); temperatures in C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ...config import DEFAULT_NUMERICAL_CONFIG
from ...contracts.materials import law_wavelength_nm
from .group import group_dispersion
from .index import DEFAULT_TEMPERATURE_C, build_evaluator


@dataclass(frozen=True)
class MaterialPoint:
    """A material at one vacuum wavelength and temperature (module docstring). ``gdd_mm2_per_mm`` is ``k2 = lambda^3 / (2 pi)
    d^2n/dlambda^2`` (mm^2 per mm of path; divided by c^2 the group-velocity dispersion). ``convention``: ``"<WAVELENGTH>/<INDEX>"``
    as the source states its law (``ENGINE_DEFAULT``: no ``index_reference``, vacuum wavelength and absolute index by
    definition); ``convention_stated`` is false when a part is ``NOT_STATED`` (then evaluated as vacuum / absolute)."""

    material_id: str
    wavelength_nm: float
    temperature_c: float
    n_abs: float | None
    n_group: float | None
    gdd_mm2_per_mm: float | None
    law_wavelength_nm: float
    convention: str
    convention_stated: bool
    reason: str | None = None
    derivative_limitation: str | None = None
    temperature_note: str | None = None


def _engine(material: Any, temperature_c: float):
    from ...local.compile import engine_material

    return engine_material(material, "", temperature_c=temperature_c)


def _convention(material: Any) -> tuple[str, bool]:
    reference = getattr(material, "index_reference", None)
    if reference is None:
        return "ENGINE_DEFAULT", True
    stated = reference.wavelength != "NOT_STATED" and reference.index != "NOT_STATED"
    return f"{reference.wavelength}/{reference.index}", stated


def _temperature_note(material: Any, temperature_c: float) -> str | None:
    """A material without thermo data at a temperature more than 1 K from the one its source states."""

    if getattr(material, "thermo_optic", None) is not None or getattr(material.dispersion, "kind", "") == "TEMPERATURE_SELLMEIER":
        return None
    measurement = getattr(material.dispersion, "measurement", None)
    source = None if measurement is None or measurement.temperature_c is None else float(measurement.temperature_c)
    if source is None:
        return (f"no thermo-optic data: the index is the source's, at its own temperature (not stated), not corrected to "
                f"{temperature_c:g} C") if temperature_c != DEFAULT_TEMPERATURE_C else None
    if abs(temperature_c - source) > 1.0:
        return f"no thermo-optic data: the index is the source's at {source:g} C, not corrected to {temperature_c:g} C"
    return None


def material_point(material: Any, wavelength_nm: float, *, temperature_c: float = DEFAULT_TEMPERATURE_C) -> MaterialPoint:
    """The material at a vacuum wavelength and temperature (module docstring)."""

    model = _engine(material, temperature_c)
    evaluator = build_evaluator(model, DEFAULT_NUMERICAL_CONFIG, temperature_c)
    convention, stated = _convention(material)
    law_nm = law_wavelength_nm(getattr(model, "index_reference", None), wavelength_nm)
    n, issue = evaluator.index(wavelength_nm)
    common = dict(material_id=material.material_id, wavelength_nm=wavelength_nm, temperature_c=temperature_c, law_wavelength_nm=law_nm,
                  convention=convention, convention_stated=stated, temperature_note=_temperature_note(material, temperature_c))
    if n is None:
        reason = f"{issue.code.value}: {issue.message}" if issue is not None else "no index"
        return MaterialPoint(n_abs=None, n_group=None, gdd_mm2_per_mm=None, reason=reason, **common)
    group = group_dispersion(model, wavelength_nm, DEFAULT_NUMERICAL_CONFIG, temperature_c)
    return MaterialPoint(n_abs=n, n_group=None if group is None else group.n_group, gdd_mm2_per_mm=None if group is None else group.k2,
                         derivative_limitation=None if group is None else group.limitation, **common)


def bulk_alpha_per_mm(material: Any, wavelength_nm: float) -> tuple[float | None, str | None]:
    """``(alpha per mm, None)`` at a vacuum wavelength from the material's ``bulk_loss`` (linear in alpha between its rows, on
    the law's wavelength) or its scalar ``absorption_coefficient_per_mm``; ``(None, reason)`` where no loss is stated or the
    wavelength lies outside the loss rows (not established: never a zero)."""

    if getattr(material, "bulk_loss", None) is None:
        scalar = getattr(material, "absorption_coefficient_per_mm", None)
        if scalar is None:
            return None, "NO_STATED_BULK_LOSS: the material states no bulk loss (unknown, not zero)"
        return float(scalar), None
    model = _engine(material, DEFAULT_TEMPERATURE_C)
    alpha = build_evaluator(model, DEFAULT_NUMERICAL_CONFIG).absorption_at(wavelength_nm)
    if alpha is None:
        rows = material.bulk_loss.wavelength_nm
        return None, (f"BULK_LOSS_NOT_ESTABLISHED: {wavelength_nm:g} nm (vacuum) lies outside the loss data "
                      f"({rows[0]}-{rows[-1]} nm on the law's wavelength)")
    return alpha, None


def internal_transmittance(material: Any, wavelength_nm: float, thickness_mm: float, *,
                           temperature_c: float = DEFAULT_TEMPERATURE_C) -> tuple[float | None, str | None]:
    """``(T_i, None)``: the internal (bulk) transmittance ``exp(-alpha L)`` of ``thickness_mm`` at a vacuum wavelength (no
    surface reflection, no coating); ``(None, reason)`` where the loss is not stated or not established
    (``bulk_alpha_per_mm``). ``math.exp`` here: a transmittance is not bit-pinned. ``temperature_c`` is accepted for symmetry
    (the loss data carry no temperature dependence)."""

    del temperature_c
    if not thickness_mm >= 0.0:
        raise ValueError("thickness_mm must be non-negative")
    alpha, reason = bulk_alpha_per_mm(material, wavelength_nm)
    if alpha is None:
        return None, reason
    return math.exp(-alpha * thickness_mm), None


__all__ = ["MaterialPoint", "bulk_alpha_per_mm", "internal_transmittance", "material_point"]
