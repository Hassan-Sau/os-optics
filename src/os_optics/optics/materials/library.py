"""Material library: built-in materials + project overrides (per validation call, no global state)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from ...config import NumericalConfig
from ...contracts.materials import (
    GRADIENT_MEDIA,
    AxialGradientIndexMaterial,
    AirMaterial,
    ConstantIndexMaterial,
    GradientIndexMaterial,
    MaterialOpticalModel,
    SellmeierMaterial,
    TabulatedGradient,
    UniaxialMaterial,
)
from ...errors import EngineInputError, ErrorCode
from .group import NM3_PER_NM2_TO_MM, group_delay_dispersion, group_index
from .index import DEFAULT_TEMPERATURE_C, MaterialEvaluator, build_evaluator, law_material, offset_evaluator
from .thermo import offsets as thermo_offsets

TWO_PI = 2.0 * math.pi

# Built-in models. "air" is n = 1 exactly (vacuum-wavelength convention, dispersion ignored): the unit-index medium of the
# Engine layer, which a project replaces by supplying its own material with id "air" -- the native lowering always does,
# with the air of the Project's environment (an ``AirMaterial``: air_index at its temperature and pressure). "vacuum" is
# n = 1 exactly in every request (a declared model's VACUUM gap).
BUILTIN_MATERIALS: tuple[MaterialOpticalModel, ...] = (
    ConstantIndexMaterial(material_id="vacuum", valid_wavelength_nm=(100.0, 100000.0), refractive_index=1.0),
    ConstantIndexMaterial(material_id="air", valid_wavelength_nm=(100.0, 100000.0), refractive_index=1.0),
    # Schott N-BK7 (catalogue Sellmeier coefficients), valid 300-2500 nm
    SellmeierMaterial(
        material_id="N-BK7",
        valid_wavelength_nm=(300.0, 2500.0),
        b=(1.03961212, 0.231792344, 1.01046945),
        c_um2=(0.00600069867, 0.0200179144, 103.560653),
    ),
    # Fused silica, Malitson 1965, valid 210-3710 nm
    SellmeierMaterial(
        material_id="fused_silica",
        valid_wavelength_nm=(210.0, 3710.0),
        b=(0.6961663, 0.4079426, 0.8974794),
        c_um2=(0.0684043**2, 0.1162414**2, 9.896161**2),
    ),
)
# Historical Engine models declare ideal lossless propagation by construction;
# project/catalog copies must supply their own loss evidence or explicit assumption.
BUILTIN_LOSSLESS_IDS = frozenset(m.material_id for m in BUILTIN_MATERIALS)


@dataclass(frozen=True)
class UniaxialLaws:
    """A uniaxial crystal's two indices as isotropic models (``law_material``) and their evaluators, with the stated optic axis
    (``UniaxialMaterial.optic_axis`` in the frame of ``axis_component``; ``scene/lower.py`` turns it into the scene frame)."""

    material: UniaxialMaterial
    ordinary: MaterialOpticalModel
    extraordinary: MaterialOpticalModel
    ordinary_index: MaterialEvaluator
    extraordinary_index: MaterialEvaluator


@dataclass(frozen=True)
class GradientLaws:
    """A gradient-index medium: its index on the axis (a radial medium) or at its axis point (an axial one) as an isotropic
    model (``law_material``) and its evaluator, and the gradient's law (``GradientIndexMaterial``: the radial constant ``g``;
    ``AxialGradientIndexMaterial``: the axial terms; ``scene/lower.py`` turns its axis into the scene frame)."""

    material: GradientIndexMaterial | AxialGradientIndexMaterial
    base: MaterialOpticalModel
    base_index: MaterialEvaluator

    @property
    def axial(self) -> tuple[float, ...] | None:
        """The axial terms ``(a1, a2, a3)`` of an axial medium (``n(z) = n0 + a1 z + ...``), else ``None``."""

        return self.material.coefficients_per_mm if isinstance(self.material, AxialGradientIndexMaterial) else None

    def gradient(self, wavelength_nm: float) -> tuple[float, float]:
        """``(g, dg/dlambda)`` at a vacuum wavelength inside the band (1/mm, 1/mm per nm): a constant, or linear between rows
        (``(0, 0)`` for an axial medium, which has no radial gradient)."""

        if isinstance(self.material, AxialGradientIndexMaterial):
            return 0.0, 0.0
        law = self.material.gradient
        if not isinstance(law, TabulatedGradient):
            return law.per_mm, 0.0
        rows = list(zip(law.wavelength_nm, law.per_mm))
        for (w0, g0), (w1, g1) in zip(rows, rows[1:]):
            if w0 <= wavelength_nm <= w1:
                slope = (g1 - g0) / (w1 - w0)
                return g0 + slope * (wavelength_nm - w0), slope
        w, g = rows[0] if wavelength_nm < rows[0][0] else rows[-1]  # (within the band's slack of an end)
        return g, 0.0


class MaterialLibrary:
    def __init__(self, project_materials: Iterable[MaterialOpticalModel], cfg: NumericalConfig,
                 temperature_c: float = DEFAULT_TEMPERATURE_C) -> None:
        """``temperature_c``: the temperature of every material that states none of its own (the environment's; a material
        with thermo-optic data or a temperature law is evaluated there, one without is evaluated as its source states it)."""
        models: dict[str, MaterialOpticalModel] = {m.material_id: m for m in BUILTIN_MATERIALS}
        ideal_builtin_ids = set(BUILTIN_LOSSLESS_IDS)
        for m in project_materials:
            models[m.material_id] = m  # project definitions override built-ins
            ideal_builtin_ids.discard(m.material_id)
        self._ideal_builtin_ids = ideal_builtin_ids
        self._cfg = cfg
        self.temperature_c = temperature_c
        t = temperature_c
        # a uniaxial crystal: its ordinary index stands for it wherever one isotropic model is read (its band, its bulk
        # absorption); the tracer reads both indices through ``uniaxial`` (``optics/polarization/uniaxial.py``)
        self._uniaxial: dict[str, UniaxialLaws] = {}
        for k, m in sorted(models.items()):
            if isinstance(m, UniaxialMaterial):
                o, e = law_material(m, m.ordinary), law_material(m, m.extraordinary)
                o_eval, e_eval = build_evaluator(o, cfg, t), build_evaluator(e, cfg, t)
                if m.temperature is not None:  # (each index at the crystal's temperature: optics/materials/thermo.py)
                    o_eval = offset_evaluator(o_eval, lambda wl, t=m.temperature: _offset(t, wl, 0))
                    e_eval = offset_evaluator(e_eval, lambda wl, t=m.temperature: _offset(t, wl, 1))
                self._uniaxial[k] = UniaxialLaws(m, o, e, o_eval, e_eval)
        # a gradient-index medium: its index on the axis (radial) or at its axis point (axial) stands for it wherever one
        # isotropic model is read; the tracer reads the profile through ``gradient`` (``optics/rays/gradient.py``)
        self._gradient: dict[str, GradientLaws] = {}
        for k, m in sorted(models.items()):
            if isinstance(m, GRADIENT_MEDIA):
                base = law_material(m, m.base)
                self._gradient[k] = GradientLaws(m, base, build_evaluator(base, cfg, t))
        self._source_models = models
        self._models = {k: (self._uniaxial[k].ordinary if k in self._uniaxial else self._gradient[k].base if k in self._gradient else m)
                        for k, m in models.items()}
        self._evaluators: dict[str, MaterialEvaluator] = {
            k: (self._uniaxial[k].ordinary_index if k in self._uniaxial else self._gradient[k].base_index if k in self._gradient
                else build_evaluator(v, cfg, t)) for k, v in sorted(models.items())}
        self._group: dict[tuple[str, float], tuple[float, float] | None] = {}

    def uniaxial(self, material_id: str) -> UniaxialLaws | None:
        """The crystal's two indices when ``material_id`` is a uniaxial crystal, else ``None``."""
        return self._uniaxial.get(material_id)

    def gradient(self, material_id: str) -> GradientLaws | None:
        """The medium's profile when ``material_id`` is a gradient-index medium (radial or axial), else ``None``."""
        return self._gradient.get(material_id)

    def gradient_derivatives(self, laws: GradientLaws, wavelength_nm: float, n0: float) -> tuple[float, float] | None:
        """``(dn0/dlambda, d2n0/dlambda2)`` of the index on the axis (per nm, per nm^2) from its group index and group-delay
        dispersion (``optics/materials/group.py``); ``None`` when they are not available there."""
        group = self.law_group(laws.base, wavelength_nm)
        if group is None:
            return None
        ng, k2 = group
        return (n0 - ng) / wavelength_nm, k2 * TWO_PI / (wavelength_nm * wavelength_nm * wavelength_nm * NM3_PER_NM2_TO_MM)

    def law_group(self, model: MaterialOpticalModel, wavelength_nm: float) -> tuple[float, float] | None:
        """``(n_g, k2)`` of one isotropic model (a crystal's index, ``UniaxialLaws``) at a vacuum wavelength; ``None`` when
        either is not available there."""
        ng = group_index(model, wavelength_nm, self._cfg, self.temperature_c)
        k2 = group_delay_dispersion(model, wavelength_nm, self._cfg, self.temperature_c)
        return None if ng is None or k2 is None else (ng, k2)

    def crystal_group(self, laws: UniaxialLaws, extraordinary: bool, wavelength_nm: float) -> tuple[float, float] | None:
        """``(n_g, k2)`` of one index of a crystal at a vacuum wavelength, at the crystal's temperature (``law_group`` of its
        law plus the temperature offset's ``delta - lambda d delta / d lambda``)."""
        got = self.law_group(laws.extraordinary if extraordinary else laws.ordinary, wavelength_nm)
        temperature = laws.material.temperature
        if got is None or temperature is None:
            return got
        shift = thermo_offsets(temperature, wavelength_nm)
        if shift is None:
            return None
        delta, slope = (shift[1], shift[3]) if extraordinary else (shift[0], shift[2])
        return got[0] + delta - wavelength_nm * slope, got[1]

    def crystal_indices(self, material_id: str, wavelength_nm: float, temperature_c: float | None = None
                        ) -> tuple[float, float] | None:
        """``(n_o, n_e)`` of the crystal ``material_id`` at a vacuum wavelength and a temperature (default: its own); ``None``
        where an index is not available (outside its band, or outside its thermo-optic rows)."""
        laws = self._uniaxial.get(material_id)
        if laws is None:
            return None
        temperature = laws.material.temperature
        if temperature is None or temperature_c is None:
            n_o, issue_o = laws.ordinary_index.index(wavelength_nm)
            n_e, issue_e = laws.extraordinary_index.index(wavelength_nm)
            return None if n_o is None or n_e is None else (n_o, n_e)
        base_o, _ = build_evaluator(laws.ordinary, self._cfg, self.temperature_c).index(wavelength_nm)
        base_e, _ = build_evaluator(laws.extraordinary, self._cfg, self.temperature_c).index(wavelength_nm)
        shift = thermo_offsets(temperature, wavelength_nm, temperature_c)
        if base_o is None or base_e is None or shift is None:
            return None
        return base_o + shift[0], base_e + shift[1]

    def group(self, material_id: str, wavelength_nm: float) -> tuple[float, float] | None:
        """``(n_g, k2)`` of a medium at a vacuum wavelength (``optics/materials/group.py``: the group index and the group-delay
        dispersion per mm of path); ``None`` when either is not available there (outside the valid band, unknown medium)."""
        key = (material_id, wavelength_nm)
        if key not in self._group:
            # (a crystal has one group index per wave and direction: the tracer reads them per beam, never here)
            # (a gradient-index medium's group index varies across it: the tracer reads it along each beam's ray)
            model = None if material_id in self._uniaxial or material_id in self._gradient else self._models.get(material_id)
            ng = None if model is None else group_index(model, wavelength_nm, self._cfg, self.temperature_c)
            k2 = None if model is None else group_delay_dispersion(model, wavelength_nm, self._cfg, self.temperature_c)
            self._group[key] = None if ng is None or k2 is None else (ng, k2)
        return self._group[key]

    def loss_unstated(self, material_id: str) -> bool:
        """Whether a material states neither measured loss nor a scalar absorption assumption.

        A defaulted zero is not evidence of zero loss. The Engine's legacy built-in
        materials remain its expressly ideal lossless models for compatibility; a
        declared scalar zero is a modelling assumption, not a measured zero.
        """
        model = self._source_models[material_id]
        return (material_id not in self._ideal_builtin_ids and material_id not in ("air", "vacuum") and not isinstance(model, AirMaterial)
                and model.bulk_loss is None and "absorption_coefficient_per_mm" not in model.model_fields_set)

    def evaluator(self, material_id: str, *, location: str | None = None) -> MaterialEvaluator:
        try:
            return self._evaluators[material_id]
        except KeyError:
            raise EngineInputError(
                ErrorCode.UNKNOWN_MATERIAL, f"unknown material {material_id!r}", location=location,
                known=sorted(self._evaluators),
            ) from None

    def ids(self) -> list[str]:
        return sorted(self._evaluators)


def _offset(temperature, wavelength_nm: float, which: int) -> float | None:
    shift = thermo_offsets(temperature, wavelength_nm)
    return None if shift is None else shift[which]
