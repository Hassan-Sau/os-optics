"""Refractive-index evaluation of a material at a vacuum wavelength: the one material-evaluation rule of the Engine.

Laws (``l`` the law's wavelength in micrometres): constant; Sellmeier ``n^2 = 1 + sum B_i l^2 / (l^2 - C_i)`` (1 to 10 terms);
power series ``n^2 = sum a_i l^e_i`` (integer exponents, by repeated multiplication); Herzberger ``n = A + B L + C L^2 + D l^2
+ E l^4 + F l^6``, ``L = 1 / (l^2 - 0.028)``; a temperature Sellmeier law (CHARMS form, at the material's temperature, refused
outside its temperature range); a tabulated index (PCHIP / linear; ``measured`` tables flagged); air at a temperature and
pressure (``air_index``, exactly 1 at zero pressure). No extrapolation outside the vacuum band ``valid_wavelength_nm``.

The rule (``contracts.materials.IndexReference``), at a vacuum wavelength ``l``:

* ``l_law = l / n_ref(l)`` for a law on air wavelengths, else ``l``;
* ``n_law = law(l_law)``;
* ``n_abs = n_law n_ref(l)`` for a law relative to air, else ``n_law``;
* ``SCHOTT_DN_DT``: ``n_abs(T) = n_abs + dn_abs(n_law, l_law, T)`` (Schott formula on the law's index and wavelength at its
  reference temperature).

``n_ref`` is the source's reference air (``air_index``), never the operating medium. A material without ``index_reference``
(or with ``not_stated`` parts) is evaluated as vacuum wavelength / absolute index (``MaterialEvaluator.convention_stated``
tells which). Arithmetic is ``+ - * / sqrt`` only, so every runtime gives the same bits; a material with neither an index
reference nor thermo data evaluates exactly as before (the same operations in the same order).

Spectral bulk loss (``bulk_loss``): ``alpha(l)`` per mm from ``k`` (``alpha = 4 pi k / l``, ``l`` the queried vacuum wavelength,
``k`` linear between measured rows on ``l_law``) or internal transmittance (``alpha = -ln(T_i) / d``; linear in ``alpha``
between rows on ``l_law``), unknown (``None``)
outside them (``MaterialEvaluator.absorption_at``). ``math.log`` is used there only: a loss is not bit-pinned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Context, Decimal, localcontext
from typing import Any, Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..._vendor.shared_contracts.design_catalog.material_physics import schott_e_term
from ...config import NumericalConfig
from ...contracts.materials import (
    GRADIENT_MEDIA,
    HERZBERGER_CONSTANT_UM2,
    AirMaterial,
    AxialGradientIndexMaterial,
    ConstantIndexLaw,
    ConstantIndexMaterial,
    ExtinctionCoefficientLoss,
    GradientIndexMaterial,
    HerzbergerMaterial,
    IndexLaw,
    IndexReference,
    MaterialOpticalModel,
    PowerSeriesMaterial,
    SchottThermoOptic,
    SellmeierIndexLaw,
    SellmeierMaterial,
    TabulatedIndexMaterial,
    TemperatureSellmeierMaterial,
    UniaxialMaterial,
    law_wavelength_nm,
    reference_air_index,
)
from ...errors import EngineInputError, ErrorCode, Status, ValidationIssue, make_issue
from ...units import NM_TO_UM

#: The material temperature where nothing states another (the glass catalogues' reference, a Project's default environment).
DEFAULT_TEMPERATURE_C = 20.0
#: Exact offset from Celsius to kelvin.
KELVIN_OFFSET = Decimal("273.15")
_TEMPERATURE_CONTEXT = Context(prec=34)
#: Fixed-point steps from a law (air) wavelength to its vacuum wavelength (n_air varies slowly: 4 steps reach float precision).
VACUUM_FIXED_POINT_STEPS = 4
#: Vacuum wavelength of a loss row in nm -> mm (alpha = 4 pi k / l per mm).
NM_TO_MM = 1e-6

# Frey et al., Table 5, claim average fit residual 1e-4 in n. Near 21.384 K
# the rounded fourth-order third pole passes through the measured Si band;
# its enormous contribution is inconsistent with the nearby 20/22 K source
# law. Two such residuals permit one residual at the queried temperature and
# one in the interpolation of the two ordinary-temperature anchors.
_SI_FREY_FIT_RESIDUAL_N = 1e-4
_SI_FREY_ANCHORS_K = (20.0, 22.0)
# Identify the *published law*, not the label of a record that contains it.
# All coefficients of the fourth-order Si fit are transcribed verbatim from
# NASA NTRS 20070021411, Table 5, in measured_sources.v1.json. Requiring the
# complete fit ensures its *published* residual is never applied to an
# independently specified temperature-Sellmeier law with a coincidental pole.
_SI_FREY_S = ((10.4907, -0.00020802, 0.00000421694, -5.82298e-9, 3.44688e-12),
              (-1346.61, 29.1664, -0.278724, 0.00105939, -1.35089e-6),
              (44282700.0, -1762130.0, -76157.5, 678.414, 103.243))
_SI_FREY_POLES_UM = ((0.299713, -1.14234e-5, 1.67134e-7, -2.51049e-10, 2.32484e-14),
                     (-3517.1, 42.3892, -0.357957, 0.00117504, -1.13212e-6),
                     (1714000.0, -144984.0, -6907.44, -39.3699, 23.577))


def _si_frey_fit(model: TemperatureSellmeierMaterial) -> bool:
    """The complete Table 5 fit, irrespective of material id."""

    return model.s_coefficients == _SI_FREY_S and model.pole_coefficients_um == _SI_FREY_POLES_UM


def temperature_kelvin(temperature_c: float) -> Decimal:
    """Convert the Celsius value's shortest decimal spelling before testing the measured Kelvin range."""

    with localcontext(_TEMPERATURE_CONTEXT):
        return Decimal(str(temperature_c)) + KELVIN_OFFSET


@dataclass(frozen=True)
class MaterialEvaluator:
    material_id: str
    band_nm: tuple[float, float]
    absorption_per_mm: float
    _eval: Callable[[float], float] = field(repr=False)
    cfg: NumericalConfig = field(repr=False)
    # bulk scattering coefficient mu_s (1/mm) and its stated anisotropy g (reported only: the scattered light is not followed)
    scattering_per_mm: float = 0.0
    anisotropy_g: float | None = None
    # spectral bulk absorption alpha(l_vacuum) per mm, None outside its rows (``bulk_loss``); None: the scalar coefficient
    _loss: Callable[[float], float | None] | None = field(default=None, repr=False)
    # why the material is not evaluated at all (a temperature outside its law's or its coefficients' range)
    _refusal: tuple[ErrorCode, str, dict[str, Any]] | None = field(default=None, repr=False)
    #: whether the source states its wavelength and index convention (else it is evaluated as vacuum / absolute)
    convention_stated: bool = True
    #: the material's temperature as evaluated (C)
    temperature_c: float = DEFAULT_TEMPERATURE_C

    @property
    def extinction_per_mm(self) -> float:
        """The scalar coefficient a traced beam's power decays with along its path: absorption plus bulk scattering (exactly the
        absorption coefficient where nothing scatters). A spectral loss is read per wavelength (``extinction_at``)."""
        return self.absorption_per_mm if self.scattering_per_mm == 0.0 else self.absorption_per_mm + self.scattering_per_mm

    @property
    def spectral_loss(self) -> bool:
        return self._loss is not None

    def absorption_at(self, wavelength_nm: float) -> float | None:
        """The bulk absorption coefficient (1/mm) at a vacuum wavelength: the spectral loss's (``None`` outside its rows: not
        established) or the scalar coefficient."""
        if self._loss is None:
            return self.absorption_per_mm
        return self._loss(wavelength_nm)

    def loss_established(self, wavelength_nm: float) -> bool:
        return self._loss is None or self._loss(wavelength_nm) is not None

    def extinction_at(self, wavelength_nm: float) -> float:
        """What the tracer attenuates with at a vacuum wavelength: the absorption there (0 where a spectral loss is not
        established: the transmission is then an upper bound and the run says so) plus bulk scattering."""
        if self._loss is None:
            return self.extinction_per_mm
        alpha = self._loss(wavelength_nm)
        alpha = 0.0 if alpha is None else alpha
        return alpha if self.scattering_per_mm == 0.0 else alpha + self.scattering_per_mm

    def index(self, wavelength_nm: float, *, location: str | None = None) -> tuple[float | None, ValidationIssue | None]:
        """Return ``(n, None)`` or ``(None, issue)``. Outside the validity band the index is not stated (never extrapolated):
        a data-coverage limit, ``OUT_OF_MODEL_SCOPE`` for the quantities behind it."""
        lo, hi = self.band_nm
        slack = self.cfg.tabulated_range_tol_nm
        if not (lo - slack <= wavelength_nm <= hi + slack):
            name = self.material_id.rsplit(":", 1)[-1]
            return None, make_issue(
                ErrorCode.MATERIAL_OUT_OF_RANGE,
                Status.OUT_OF_MODEL_SCOPE,
                f"wavelength {wavelength_nm} nm is outside the validity band [{lo}, {hi}] nm of material {name}",
                location=location or f"material:{self.material_id}",
                material=name,  # the material's name; its full id is in the location
                wavelength_nm=wavelength_nm,
                valid_min_nm=lo,
                valid_max_nm=hi,
                suggested_action="use a material valid at this wavelength or supply a model covering it",
            )
        if self._refusal is not None:
            code, message, context = self._refusal
            return None, make_issue(code, Status.FAIL, message, location=location or f"material:{self.material_id}",
                                    wavelength_nm=wavelength_nm, **context)
        try:
            n = self._eval(wavelength_nm)
        except _MaterialEvalError as exc:
            return None, make_issue(
                ErrorCode.MATERIAL_INVALID, Status.FAIL, str(exc), location=location or f"material:{self.material_id}",
                wavelength_nm=wavelength_nm,
            )
        if not math.isfinite(n) or n <= 0.0:
            return None, make_issue(
                ErrorCode.MATERIAL_INVALID,
                Status.FAIL,
                f"material {self.material_id!r} produced a non-physical index at {wavelength_nm} nm",
                location=location or f"material:{self.material_id}",
                wavelength_nm=wavelength_nm,
            )
        return n, None


class _MaterialEvalError(ArithmeticError):
    pass


def _constant(n0: float) -> Callable[[float], float]:
    return lambda _wl: n0


def _sellmeier(b: tuple[float, ...], c: tuple[float, ...], pole_rel_tol: float) -> Callable[[float], float]:
    def evaluate(wavelength_nm: float) -> float:
        lam = wavelength_nm * NM_TO_UM
        lam2 = lam * lam  # um^2
        total = 0.0
        for bi, ci in zip(b, c):
            denom = lam2 - ci
            if abs(denom) <= pole_rel_tol * lam2:
                raise _MaterialEvalError(f"Sellmeier pole reached at {wavelength_nm} nm (lambda^2 = C_i)")
            total += bi * lam2 / denom
        n2 = 1.0 + total
        if n2 <= 0.0:
            raise _MaterialEvalError(f"Sellmeier n^2 = {n2} is not positive at {wavelength_nm} nm")
        return math.sqrt(n2)

    return evaluate


def integer_power(x: float, e: int) -> float:
    """``x^e`` for an integer ``e`` by repeated multiplication (``1 / x^-e`` for a negative one): the same bits everywhere."""

    if e < 0:
        return 1.0 / integer_power(x, -e)
    out = 1.0
    for _ in range(e):
        out = out * x
    return out


def _power_series(model: PowerSeriesMaterial) -> Callable[[float], float]:
    def evaluate(wavelength_nm: float) -> float:
        lam = wavelength_nm * NM_TO_UM
        n2 = 0.0
        for a, e in zip(model.coefficients, model.exponents):
            n2 += a * integer_power(lam, e)
        if not n2 > 0.0:
            raise _MaterialEvalError(f"power series n^2 = {n2} is not positive at {wavelength_nm} nm")
        return math.sqrt(n2)

    return evaluate


def _herzberger(model: HerzbergerMaterial) -> Callable[[float], float]:
    a, b, c, d, e, f = model.coefficients

    def evaluate(wavelength_nm: float) -> float:
        lam = wavelength_nm * NM_TO_UM
        s = lam * lam
        denom = s - HERZBERGER_CONSTANT_UM2
        if denom == 0.0:
            raise _MaterialEvalError(f"Herzberger pole reached at {wavelength_nm} nm (lambda^2 = 0.028 um^2)")
        big_l = 1.0 / denom
        return a + b * big_l + c * big_l * big_l + d * s + e * s * s + f * s * s * s

    return evaluate


def _polynomial(coefficients: tuple[float, ...], x: float) -> float:
    acc = 0.0
    for c in reversed(coefficients):
        acc = acc * x + c
    return acc


def temperature_sellmeier_terms(model: TemperatureSellmeierMaterial, temperature_k: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(S_i, l_i^2)`` of a temperature Sellmeier law at a temperature (kelvin): its Sellmeier ``B`` and ``C`` (um^2) there."""

    s_values = tuple(_polynomial(row, temperature_k) for row in model.s_coefficients)
    poles = tuple(_polynomial(row, temperature_k) for row in model.pole_coefficients_um)
    return s_values, tuple(p * p for p in poles)


def _tabulated(model: TabulatedIndexMaterial) -> Callable[[float], float]:
    wl = np.array(model.wavelength_nm, dtype=np.float64)
    n = np.array(model.refractive_index, dtype=np.float64)
    if model.interpolation == "pchip":
        interp = PchipInterpolator(wl, n, extrapolate=False)
        return lambda x: float(interp(x))
    return lambda x: float(np.interp(x, wl, n))


def schott_delta(thermo: SchottThermoOptic, n: float, law_nm: float, temperature_c: float) -> float:
    """``dn_abs`` of the Schott formula at the law's wavelength in nm.

    The pinned OHARA/CDGM laws have 0.311/0.312-um poles *inside* their
    index bands. Outside the coefficient's half-last-digit neighbourhood
    the original law is unchanged, even when the record has been renamed.
    At the reference temperature its correction is exactly zero everywhere.
    When the E numerator is zero the pole is removable and D-only correction
    remains defined, including at lambda_tk_um.
    """

    dt = temperature_c - thermo.reference_temperature_c
    if dt == 0.0:
        return 0.0
    numerator = thermo.e0 * dt + thermo.e1 * dt * dt
    try:
        e_term = schott_e_term(law_nm, thermo.lambda_tk_um, numerator, wavelength_is_nm=True)
    except ValueError as exc:
        raise _MaterialEvalError(str(exc)) from exc
    return (n * n - 1.0) / (2.0 * n) * (thermo.d0 * dt + thermo.d1 * dt * dt + thermo.d2 * dt * dt * dt + e_term)


def material_temperature(model: Any, temperature_c: float | None) -> float:
    """The temperature a material is evaluated at: its own ``temperature_c``, else the library's (default 20 C)."""

    own = getattr(model, "temperature_c", None) if not isinstance(model, AirMaterial) else None
    if own is not None:
        return own
    return DEFAULT_TEMPERATURE_C if temperature_c is None else temperature_c


def law_function(model: MaterialOpticalModel, cfg: NumericalConfig, temperature_c: float = DEFAULT_TEMPERATURE_C
                 ) -> Callable[[float], float]:
    """The law of an isotropic material as a function of the law's wavelength (nm): ``n_law``."""

    if isinstance(model, ConstantIndexMaterial):
        return _constant(model.refractive_index)
    if isinstance(model, SellmeierMaterial):
        return _sellmeier(model.b, model.c_um2, cfg.sellmeier_pole_rel_tol)
    if isinstance(model, TabulatedIndexMaterial):
        return _tabulated(model)
    if isinstance(model, PowerSeriesMaterial):
        return _power_series(model)
    if isinstance(model, HerzbergerMaterial):
        return _herzberger(model)
    if isinstance(model, TemperatureSellmeierMaterial):
        temperature_k = float(temperature_kelvin(temperature_c))
        b, c = temperature_sellmeier_terms(model, temperature_k)
        law = _sellmeier(b, c, cfg.sellmeier_pole_rel_tol)
        # The Si Table 5 fit is singular at an interior temperature even though
        # both the tabulated 30 K point and its own 20/22 K evaluations are
        # ordinary. At that narrow crossing, compare to the straight line
        # between 20 and 22 K; its tolerance is twice the *published* 1e-4
        # fit residual, not an invented global n threshold or a clipped index.
        if not _si_frey_fit(model) or not _SI_FREY_ANCHORS_K[0] < temperature_k < _SI_FREY_ANCHORS_K[1]:
            return law
        start, end = _SI_FREY_ANCHORS_K
        first = _sellmeier(*temperature_sellmeier_terms(model, start), cfg.sellmeier_pole_rel_tol)
        last = _sellmeier(*temperature_sellmeier_terms(model, end), cfg.sellmeier_pole_rel_tol)

        def verified(wavelength_nm: float) -> float:
            n = law(wavelength_nm)
            baseline = first(wavelength_nm) + (last(wavelength_nm) - first(wavelength_nm)) * (temperature_k - start) / (end - start)
            if abs(n - baseline) > 2.0 * _SI_FREY_FIT_RESIDUAL_N:
                raise _MaterialEvalError(f"silicon Frey 2006 temperature-Sellmeier pole neighbourhood near "
                                         f"{temperature_k:.8g} K at {wavelength_nm:.9g} nm: "
                                         "the fit differs from its 20/22 K continuation beyond the source's fit residual")
            return n

        return verified
    if isinstance(model, AirMaterial):
        from ..._vendor.shared_contracts.design_catalog.material_physics import air_index

        t, p = model.temperature_c, model.pressure_pa
        return lambda wl: air_index(wl, t, p)
    raise EngineInputError(ErrorCode.INVALID_INPUT, f"unknown material model {type(model).__name__}")  # pragma: no cover


def _composed(law: Callable[[float], float], reference: IndexReference | None, thermo: SchottThermoOptic | None,
              temperature_c: float) -> Callable[[float], float]:
    """The material-evaluation rule (module docstring) around a law of the law's wavelength."""

    air_wavelength = reference is not None and reference.air_wavelength
    relative = reference is not None and reference.relative_to_air
    if not air_wavelength and not relative and thermo is None:
        return law

    def evaluate(wavelength_nm: float) -> float:
        lam_law = wavelength_nm
        n_ref = 1.0
        if air_wavelength or relative:
            n_ref = reference_air_index(reference, wavelength_nm)  # type: ignore[arg-type]
            if air_wavelength:
                lam_law = wavelength_nm / n_ref
        n_law = law(lam_law)
        n_abs = n_law * n_ref if relative else n_law
        if thermo is not None:
            n_abs = n_abs + schott_delta(thermo, n_law, lam_law, temperature_c)
        return n_abs

    return evaluate


def vacuum_wavelength_nm(reference: IndexReference | None, law_nm: float) -> float:
    """The vacuum wavelength whose ``l_law`` is ``law_nm`` (fixed point for a law on air wavelengths)."""

    if reference is None or not reference.air_wavelength:
        return law_nm
    vacuum = law_nm
    for _ in range(VACUUM_FIXED_POINT_STEPS):
        vacuum = law_nm * reference_air_index(reference, vacuum)
    return vacuum


def loss_rows(model: Any) -> list[tuple[float, float]]:
    """``(l_law nm, alpha per mm)`` of each row of a material's ``bulk_loss`` (``alpha = 4 pi k / l_vacuum`` or
    ``-ln(T_i) / d``)."""

    loss = model.bulk_loss
    reference = getattr(model, "index_reference", None)
    if isinstance(loss, ExtinctionCoefficientLoss):
        return [(w, 4.0 * math.pi * k / (vacuum_wavelength_nm(reference, w) * NM_TO_MM)) for w, k in zip(loss.wavelength_nm, loss.k)]
    return [(w, -math.log(t) / d) for w, t, d in zip(loss.wavelength_nm, loss.internal_transmittance, loss.thickness_mm)]


def _loss_function(model: Any, cfg: NumericalConfig) -> Callable[[float], float | None] | None:
    if getattr(model, "bulk_loss", None) is None:
        return None
    loss = model.bulk_loss
    reference = getattr(model, "index_reference", None)
    # k is the measured imaginary index: interpolate *k* on the law's wavelength, then use the requested vacuum
    # wavelength in alpha = 4 pi k / lambda. Internal transmittance is measured at thickness d: interpolate
    # alpha = -ln(T_i) / d instead, so Beer-Lambert scaling is consistent between its rows.
    rows = (list(zip((float(w) for w in loss.wavelength_nm), loss.k))
            if isinstance(loss, ExtinctionCoefficientLoss) else loss_rows(model))
    slack = cfg.tabulated_range_tol_nm

    def alpha(wavelength_nm: float) -> float | None:
        lam = law_wavelength_nm(reference, wavelength_nm)
        if lam < rows[0][0] - slack or lam > rows[-1][0] + slack:
            return None
        if lam <= rows[0][0]:
            value = rows[0][1]
        elif lam >= rows[-1][0]:
            value = rows[-1][1]
        else:
            value = None
            for (w0, a0), (w1, a1) in zip(rows, rows[1:]):
                if w0 <= lam <= w1:
                    value = a0 + (a1 - a0) * (lam - w0) / (w1 - w0)
                    break
            if value is None:  # pragma: no cover - the rows cover [first, last]
                return None
        return (4.0 * math.pi * value / (wavelength_nm * NM_TO_MM)
                if isinstance(loss, ExtinctionCoefficientLoss) else value)

    return alpha


def _refusal(model: Any, temperature_c: float) -> tuple[ErrorCode, str, dict[str, Any]] | None:
    name = model.material_id.rsplit(":", 1)[-1]
    if isinstance(model, TemperatureSellmeierMaterial):
        low, high = model.temperature_k_range
        kelvin = temperature_kelvin(temperature_c)
        if not kelvin.is_finite() or not Decimal(str(low)) <= kelvin <= Decimal(str(high)):
            return (ErrorCode.MATERIAL_OUT_OF_RANGE,
                    f"material {name}: its temperature law holds for {low}-{high} K; the material is at {kelvin} K "
                    f"({temperature_c} C)", {"material": name, "temperature_k": float(kelvin), "valid_min_k": low, "valid_max_k": high})
    thermo = getattr(model, "thermo_optic", None)
    if thermo is not None and thermo.valid_temperature_c is not None:
        low, high = thermo.valid_temperature_c
        if not low <= temperature_c <= high:
            return (ErrorCode.MATERIAL_OUT_OF_RANGE,
                    f"material {name}: its thermo-optic coefficients hold for {low}-{high} C; the material is at {temperature_c:.6g} C",
                    {"material": name, "temperature_c": temperature_c, "valid_min_c": low, "valid_max_c": high})
    return None


def law_material(material: UniaxialMaterial | GradientIndexMaterial | AxialGradientIndexMaterial, law: IndexLaw
                 ) -> ConstantIndexMaterial | SellmeierMaterial | TabulatedIndexMaterial:
    """One index of a uniaxial crystal (or the index on the axis of a gradient-index medium) as an isotropic material model of
    the material's id, band and absorption (what the index, group-index and dispersion evaluators read); a gradient-index
    medium's bulk scattering is kept with it, and a spectral bulk loss with either."""
    common: dict[str, Any] = {"material_id": material.material_id, "valid_wavelength_nm": material.valid_wavelength_nm,
                              "absorption_coefficient_per_mm": material.absorption_coefficient_per_mm}
    if material.bulk_loss is not None:
        common["bulk_loss"] = material.bulk_loss
    if isinstance(material, GRADIENT_MEDIA):
        common.update(scattering_coefficient_per_mm=material.scattering_coefficient_per_mm, anisotropy_g=material.anisotropy_g)
    if isinstance(law, ConstantIndexLaw):
        return ConstantIndexMaterial(refractive_index=law.refractive_index, **common)
    if isinstance(law, SellmeierIndexLaw):
        return SellmeierMaterial(b=law.b, c_um2=law.c_um2, **common)
    return TabulatedIndexMaterial(wavelength_nm=law.wavelength_nm, refractive_index=law.refractive_index,
                                  interpolation=law.interpolation, **common)


def build_evaluator(model: MaterialOpticalModel, cfg: NumericalConfig, temperature_c: float | None = None) -> MaterialEvaluator:
    """The evaluator of a material at a temperature (``temperature_c``: the library's; the material's own ``temperature_c``
    takes precedence; default 20 C). A crystal is read through its ordinary index, a gradient-index medium through its index
    on the axis."""

    if isinstance(model, UniaxialMaterial):  # (its ordinary index: what one isotropic reading of the crystal sees)
        model = law_material(model, model.ordinary)
    elif isinstance(model, GRADIENT_MEDIA):  # (its index on the axis, or at the axis point)
        model = law_material(model, model.base)
    temperature = material_temperature(model, temperature_c)
    reference = getattr(model, "index_reference", None)
    thermo = getattr(model, "thermo_optic", None)
    fn = _composed(law_function(model, cfg, temperature), reference, thermo, temperature)
    return MaterialEvaluator(model.material_id, model.valid_wavelength_nm, model.absorption_coefficient_per_mm, fn, cfg,
                             model.scattering_coefficient_per_mm or 0.0, model.anisotropy_g, _loss_function(model, cfg),
                             _refusal(model, temperature), reference is None or reference.stated, temperature)


def offset_evaluator(base: MaterialEvaluator, offset: Callable[[float], float | None]) -> MaterialEvaluator:
    """``base`` with ``offset(wavelength)`` added to its index (a crystal at its temperature, ``optics/materials/thermo.py``);
    where ``offset`` is ``None`` the index is not stated (``MATERIAL_INVALID``)."""

    law = base._eval

    def evaluate(wavelength_nm: float) -> float:
        delta = offset(wavelength_nm)
        if delta is None:
            raise _MaterialEvalError(f"the thermo-optic coefficients of {base.material_id.rsplit(':', 1)[-1]} are not stated at "
                                     f"{wavelength_nm} nm (outside the rows of its thermo_optic table)")
        return law(wavelength_nm) + delta

    return MaterialEvaluator(base.material_id, base.band_nm, base.absorption_per_mm, evaluate, base.cfg, base.scattering_per_mm,
                             base.anisotropy_g, base._loss, base._refusal, base.convention_stated, base.temperature_c)
