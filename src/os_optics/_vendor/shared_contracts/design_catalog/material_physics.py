"""The physics of a refractive material beyond its dispersion law, and the one rule that resolves a material name.

Catalog schema 3.6 publishes, and a Project declares, these optional parts of a material (``PublicMaterialV1`` /
``ProjectModelMaterialV1``); a material without them keeps its earlier payload and meaning.

**Spectral coordinate.** Every Engine computation runs on the *vacuum* wavelength (equivalently the frequency). A
dispersion law may be stated for another coordinate: ``index_reference`` (``IndexReferenceV1``) states how its source wrote
it -- the wavelength the law takes (``VACUUM`` or ``AIR`` of stated reference conditions) and what its index is relative to
(``ABSOLUTE``: vacuum, or ``RELATIVE_TO_AIR``: that reference air). At a vacuum wavelength ``l`` the material's absolute
index is

* ``l_law = l / n_ref(l)`` for a law stated on air wavelengths, else ``l_law = l``;
* ``n_law = law(l_law)``;
* ``n_abs = n_law * n_ref(l)`` for an index relative to air, else ``n_abs = n_law``,

with ``n_ref`` the absolute index of the source's reference air (``air_index``). A table (the index, a loss table) is
written on ``l_law``. ``valid_band_nm`` stays a band of vacuum wavelengths: the vacuum wavelengths whose ``l_law`` lies in
the band the source states (``source_band_nm``, on ``l_law``), rounded inward to ``BAND_QUANTUM_NM``
(``vacuum_band``). A source that states no convention is ``NOT_STATED`` and is evaluated in the Engine's own convention
(vacuum wavelength, absolute index), identifiably; a material without ``index_reference`` is in that convention by
definition. The manufacturer's reference air is not the medium a lens operates in: a Project states that
(``optical_design.optical_model.PropagationEnvironmentV1``).

**Reference air.** ``air_index`` is the air formula of Edlen (1953) at 15 C and 1 atm, scaled to a temperature and pressure
as OpticStudio does (``n - 1 = (n_15 - 1) (P / 1 atm) / (1 + 3.4785e-3 (T - 15 C))``), evaluated at the vacuum wavelength
(OpticStudio evaluates it at the air wavelength; the difference in ``n_air`` is below 1e-10). The formula is a fit to
visible and near-infrared measurements (``AIR_MODEL_FIT_BAND_NM``); beyond it the formula is extrapolated (its dispersion
terms tend to constants), an approximation, not a measured infrared law of air. It runs with ``+ - * /`` only.

**Bulk loss** (``bulk_loss``): the measured extinction coefficient ``k`` (linear in k on the law's wavelength, then
``alpha = 4 pi k / l_vacuum`` at the queried wavelength) or the measured internal transmittance ``T_i`` at a stated
thickness ``d`` (``alpha = -ln(T_i) / d``, linear in alpha between rows), each on its own wavelengths (``l_law``) and only
there. Surface reflection and coatings are not part of it; a total (external) transmittance is not an internal one. A
material without ``bulk_loss`` has no stated loss -- not a stated zero loss.

**Temperature** (``thermo_optic``): the change of the absolute index with temperature (``SCHOTT_DN_DT``: the Schott
formula of the glass catalogues), or a law of index and temperature together (``TemperatureSellmeierDispersionV1``, a
dispersion kind: the NASA CHARMS form). A material's thermo-optic response is not a lens's athermal behaviour, which also
depends on its geometry, expansion and mount.

**Identity** (``identity``): the maker, the maker's exact designation, documented aliases, the catalogue selection tags a
prescription's catalogue list names and an anneal or other variant's base record -- what an installed client resolves a
name with (``resolve_material_name``) from the published catalog alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, ROUND_HALF_EVEN, localcontext
from typing import Annotated, Any, ClassVar, Iterable, Literal, TypeVar

from pydantic import Field, StringConstraints, model_validator

from ..base import OmitAbsentFields, StrictContractModel
from ..common.decimal_string import DecimalString

_CONTEXT = Context(prec=34, rounding=ROUND_HALF_EVEN)
_ASCII_NAME = r"^[ -~]*[!-~][ -~]*$"
_Name = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=200, pattern=_ASCII_NAME)]
_Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=1000, pattern=_ASCII_NAME)]
_MaterialRef = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[A-Z0-9][A-Z0-9_.-]*$")]
CatalogueTag = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[A-Z0-9][A-Z0-9_.+-]*$"),
    Field(description="A catalogue selection tag (upper case): a maker code ('SCHOTT', 'CDGM') or a catalogue file's canonical name."),
]

# ------------------------------------------------------------------------------------------------------- air

#: The air model of every reference air and of a Project's environment.
AIR_MODEL_EDLEN_1953 = "EDLEN_1953_OPTICSTUDIO"
AirModel = Literal["EDLEN_1953_OPTICSTUDIO"]
#: The glass catalogues' standard reference air (and a Project environment's default).
STANDARD_TEMPERATURE_C = Decimal(20)
STANDARD_PRESSURE_PA = Decimal(101325)
#: Vacuum wavelengths (nm) the air formula is fitted over; outside it the formula is extrapolated (an approximation).
AIR_MODEL_FIT_BAND_NM = (Decimal(200), Decimal(2000))
#: The vacuum wavelengths (nm) the air formula is evaluated at all (it has poles at 83 nm and 156 nm); outside
#: ``AIR_MODEL_FIT_BAND_NM`` it is extrapolated (an approximation, recorded with the material).
AIR_MODEL_BAND_NM = (Decimal(180), Decimal(100000))
#: ``valid_band_nm`` of a law on air wavelengths is its source band transformed to vacuum, rounded inward to this (nm).
BAND_QUANTUM_NM = Decimal("0.001")
_ATM_PA = Decimal(101325)


def air_index_exact(vacuum_nm: Decimal, temperature_c: Decimal = STANDARD_TEMPERATURE_C,
                    pressure_pa: Decimal = STANDARD_PRESSURE_PA) -> Decimal:
    """The absolute index of air (module docstring) in exact decimal arithmetic (34 digits)."""

    with localcontext(_CONTEXT):
        lam = Decimal(vacuum_nm) / 1000
        s = lam * lam
        standard = (Decimal("6432.8") + Decimal(2949810) * s / (146 * s - 1) + Decimal(25540) * s / (41 * s - 1)) / Decimal(10) ** 8
        return 1 + standard * (Decimal(pressure_pa) / _ATM_PA) / (1 + Decimal("0.0034785") * (Decimal(temperature_c) - 15))


def air_index(vacuum_nm: float, temperature_c: float = 20.0, pressure_pa: float = 101325.0) -> float:
    """The absolute index of air (module docstring) in binary floating point, with ``+ - * /`` only (the same bits on
    every runtime)."""

    lam = vacuum_nm / 1000.0
    s = lam * lam
    standard = (6432.8 + 2949810.0 * s / (146.0 * s - 1.0) + 25540.0 * s / (41.0 * s - 1.0)) * 1e-8
    return 1.0 + standard * (pressure_pa / 101325.0) / (1.0 + 0.0034785 * (temperature_c - 15.0))


def air_index_slope(vacuum_nm: float, temperature_c: float = 20.0, pressure_pa: float = 101325.0) -> tuple[float, float]:
    """``(dn_air/dl, d2n_air/dl2)`` per nm and per nm^2 at a vacuum wavelength (exact derivatives of ``air_index``)."""

    lam = vacuum_nm / 1000.0
    s = lam * lam
    scale = 1e-8 * (pressure_pa / 101325.0) / (1.0 + 0.0034785 * (temperature_c - 15.0))
    # d/ds [A s / (a s - 1)] = -A / (a s - 1)^2 ;  d2/ds2 = 2 A a / (a s - 1)^3
    d1 = d2 = 0.0
    for big, a in ((2949810.0, 146.0), (25540.0, 41.0)):
        q = a * s - 1.0
        d1 += -big / (q * q)
        d2 += 2.0 * big * a / (q * q * q)
    # s = (l/1000)^2: ds/dl = 2 l / 1e6, d2s/dl2 = 2 / 1e6
    ds = 2.0 * vacuum_nm / 1e6
    return scale * d1 * ds, scale * (d2 * ds * ds + d1 * 2.0 / 1e6)


class ReferenceAirV1(StrictContractModel):
    """Air of stated conditions, with the air model its index is computed by (``air_index``)."""

    model: AirModel = AIR_MODEL_EDLEN_1953
    temperature_c: DecimalString
    pressure_pa: DecimalString

    @model_validator(mode="after")
    def _physical(self) -> ReferenceAirV1:
        if not Decimal(self.temperature_c) > Decimal("-273.15"):
            raise ValueError("temperature_c must be above absolute zero")
        if Decimal(self.pressure_pa) < 0:
            raise ValueError("pressure_pa cannot be negative")
        return self

    def index_exact(self, vacuum_nm: Decimal) -> Decimal:
        return air_index_exact(vacuum_nm, Decimal(self.temperature_c), Decimal(self.pressure_pa))

    def index(self, vacuum_nm: float) -> float:
        return air_index(vacuum_nm, float(self.temperature_c), float(self.pressure_pa))


STANDARD_REFERENCE_AIR = ReferenceAirV1(temperature_c="20", pressure_pa="101325")

# --------------------------------------------------------------------------------------------- index reference

WavelengthConvention = Literal["VACUUM", "AIR", "NOT_STATED"]
IndexConvention = Literal["ABSOLUTE", "RELATIVE_TO_AIR", "NOT_STATED"]
ConventionStatedBy = Literal["CATALOGUE_FORMAT", "SOURCE_TEXT", "PUBLISHER", "NOT_STATED"]


class IndexReferenceV1(StrictContractModel):
    """How the material's source states its dispersion law (module docstring): the wavelength the law takes, what its index
    is relative to, the reference air of either (required when one is ``AIR`` / ``RELATIVE_TO_AIR``), the band the source
    states on that wavelength (``source_band_nm``), and who states the convention."""

    wavelength: WavelengthConvention
    index: IndexConvention
    reference_air: ReferenceAirV1 | None = None
    source_band_nm: tuple[DecimalString, DecimalString]
    stated_by: ConventionStatedBy
    basis: _Text | None = None

    @model_validator(mode="after")
    def _consistent(self) -> IndexReferenceV1:
        needs_air = self.wavelength == "AIR" or self.index == "RELATIVE_TO_AIR"
        if needs_air and self.reference_air is None:
            raise ValueError("a law on air wavelengths or relative to air states its reference_air")
        if not needs_air and self.reference_air is not None:
            raise ValueError("reference_air belongs to a law on air wavelengths or relative to air")
        if (self.stated_by == "NOT_STATED") != (self.wavelength == "NOT_STATED" and self.index == "NOT_STATED"):
            raise ValueError("stated_by is NOT_STATED exactly when neither the wavelength nor the index convention is stated")
        low, high = (Decimal(v) for v in self.source_band_nm)
        if not Decimal(0) < low < high:
            raise ValueError("source_band_nm must satisfy 0 < min < max")
        if self.wavelength == "AIR" and low < AIR_MODEL_BAND_NM[0] * Decimal("0.999"):
            raise ValueError(f"a law on air wavelengths is evaluated from {AIR_MODEL_BAND_NM[0]} nm (the air formula's range)")
        return self

    @property
    def air_wavelength(self) -> bool:
        return self.wavelength == "AIR"

    @property
    def relative_to_air(self) -> bool:
        return self.index == "RELATIVE_TO_AIR"

    def law_wavelength_exact(self, vacuum_nm: Decimal) -> Decimal:
        """``l_law`` of a vacuum wavelength (exact decimal)."""

        if not self.air_wavelength:
            return Decimal(vacuum_nm)
        with localcontext(_CONTEXT):
            return Decimal(vacuum_nm) / self.reference_air.index_exact(Decimal(vacuum_nm))  # type: ignore[union-attr]

    def law_wavelength(self, vacuum_nm: float) -> float:
        return vacuum_nm / self.reference_air.index(vacuum_nm) if self.air_wavelength else vacuum_nm  # type: ignore[union-attr]

    def vacuum_wavelength_exact(self, law_nm: Decimal) -> Decimal:
        """The vacuum wavelength whose ``l_law`` is ``law_nm`` (fixed point; ``n_air`` varies slowly)."""

        if not self.air_wavelength:
            return Decimal(law_nm)
        with localcontext(_CONTEXT):
            vacuum = Decimal(law_nm)
            for _ in range(6):
                vacuum = Decimal(law_nm) * self.reference_air.index_exact(vacuum)  # type: ignore[union-attr]
            return vacuum

    def vacuum_band(self) -> tuple[Decimal, Decimal]:
        """``valid_band_nm`` of the law: the vacuum wavelengths whose ``l_law`` lies in ``source_band_nm``, rounded inward
        to ``BAND_QUANTUM_NM`` (exactly the source band for a law on vacuum wavelengths)."""

        low, high = (Decimal(v) for v in self.source_band_nm)
        if not self.air_wavelength:
            return low, high
        with localcontext(_CONTEXT):
            lo = self.vacuum_wavelength_exact(low).quantize(BAND_QUANTUM_NM, rounding=ROUND_CEILING)
            while self.law_wavelength_exact(lo) < low:
                lo += BAND_QUANTUM_NM
            hi = self.vacuum_wavelength_exact(high).quantize(BAND_QUANTUM_NM, rounding=ROUND_FLOOR)
            while self.law_wavelength_exact(hi) > high:
                hi -= BAND_QUANTUM_NM
            return lo.normalize(), hi.normalize()

    def check_vacuum_band(self, valid_min_nm: Decimal, valid_max_nm: Decimal) -> str | None:
        """Why ``valid_band_nm`` is not inside the source band on the law's wavelength, or None."""

        low, high = (Decimal(v) for v in self.source_band_nm)
        if self.law_wavelength_exact(valid_min_nm) < low or self.law_wavelength_exact(valid_max_nm) > high:
            what = "air wavelengths" if self.air_wavelength else "wavelengths"
            return (f"the valid band {valid_min_nm.normalize():f}-{valid_max_nm.normalize():f} nm (vacuum) reaches beyond the source band "
                    f"{low.normalize():f}-{high.normalize():f} nm ({what})")
        return None


# ------------------------------------------------------------------------------------------- dispersion laws

#: The most terms of a Sellmeier law (a constant term is a term with ``c = 0``) and of a power series.
MAX_SELLMEIER_TERMS = 10
MAX_POWER_SERIES_TERMS = 16
#: The most coefficients of one temperature polynomial of a temperature Sellmeier law.
MAX_TEMPERATURE_TERMS = 8
#: Exponents of a power series are integers within this magnitude (``l`` in um).
MAX_POWER_SERIES_EXPONENT = 24
#: The constant of the Herzberger formula's ``L = 1 / (l^2 - 0.028)`` (um^2).
HERZBERGER_CONSTANT_UM2 = Decimal("0.028")
#: The most rows of a measured table (index or loss).
MAX_TABLE_POINTS = 4096


class PowerSeriesDispersionV1(StrictContractModel):
    """``n^2 = sum_i a_i l^e_i`` (``l`` in um, integer exponents ``e_i``): the Schott power series
    (``a0 + a1 l^2 + a2 l^-2 + ...``, AGF formula 1) and refractiveindex.info formula 3 with integer exponents."""

    kind: Literal["POWER_SERIES"] = "POWER_SERIES"
    coefficients: tuple[DecimalString, ...] = Field(min_length=1, max_length=MAX_POWER_SERIES_TERMS)
    exponents: tuple[int, ...] = Field(min_length=1, max_length=MAX_POWER_SERIES_TERMS)

    @model_validator(mode="after")
    def _terms(self) -> PowerSeriesDispersionV1:
        if len(self.coefficients) != len(self.exponents):
            raise ValueError("coefficients and exponents must have equal length")
        if any(abs(e) > MAX_POWER_SERIES_EXPONENT for e in self.exponents):
            raise ValueError(f"exponents must lie within +-{MAX_POWER_SERIES_EXPONENT}")
        return self

    def n_squared_exact(self, law_um: Decimal) -> Decimal:
        with localcontext(_CONTEXT):
            return sum((Decimal(a) * law_um ** e for a, e in zip(self.coefficients, self.exponents)), Decimal(0))


class HerzbergerDispersionV1(StrictContractModel):
    """``n = A + B L + C L^2 + D l^2 + E l^4 + F l^6`` with ``L = 1 / (l^2 - 0.028)`` (``l`` in um; AGF formula 3)."""

    kind: Literal["HERZBERGER"] = "HERZBERGER"
    coefficients: tuple[DecimalString, DecimalString, DecimalString, DecimalString, DecimalString, DecimalString]

    def index_exact(self, law_um: Decimal) -> Decimal:
        a, b, c, d, e, f = (Decimal(v) for v in self.coefficients)
        with localcontext(_CONTEXT):
            s = law_um * law_um
            big_l = 1 / (s - HERZBERGER_CONSTANT_UM2)
            return a + b * big_l + c * big_l * big_l + d * s + e * s * s + f * s * s * s


class TemperatureSellmeierDispersionV1(StrictContractModel):
    """A law of index and temperature together (NASA CHARMS form, Frey, Leviton and Madison 2006):
    ``n^2 - 1 = sum_i S_i(T) l^2 / (l^2 - l_i(T)^2)`` with ``S_i(T) = sum_j s[i][j] T^j`` and ``l_i(T) = sum_j p[i][j] T^j``
    (``T`` in kelvin, ``l`` and ``l_i`` in um), valid on ``temperature_k_range`` (the measured temperatures) and the material's
    band. ``reference_temperature_k`` is where it is evaluated when no temperature is stated. The stated temperature and
    wavelength ranges do not license evaluation on a moving Sellmeier pole: an evaluator must refuse the pole's invalid
    neighbourhood rather than clip its output or extrapolate it across the singularity."""

    kind: Literal["TEMPERATURE_SELLMEIER"] = "TEMPERATURE_SELLMEIER"
    s_coefficients: tuple[tuple[DecimalString, ...], ...] = Field(min_length=1, max_length=MAX_SELLMEIER_TERMS)
    pole_coefficients_um: tuple[tuple[DecimalString, ...], ...] = Field(min_length=1, max_length=MAX_SELLMEIER_TERMS)
    temperature_k_range: tuple[DecimalString, DecimalString]
    reference_temperature_k: DecimalString

    @model_validator(mode="after")
    def _shape(self) -> TemperatureSellmeierDispersionV1:
        if len(self.s_coefficients) != len(self.pole_coefficients_um):
            raise ValueError("s_coefficients and pole_coefficients_um must have one row per term")
        if any(not 1 <= len(row) <= MAX_TEMPERATURE_TERMS for row in (*self.s_coefficients, *self.pole_coefficients_um)):
            raise ValueError("each temperature polynomial has 1 to 8 coefficients")
        low, high = (Decimal(v) for v in self.temperature_k_range)
        if not Decimal(0) < low < high:
            raise ValueError("temperature_k_range must satisfy 0 < min < max")
        if not low <= Decimal(self.reference_temperature_k) <= high:
            raise ValueError("reference_temperature_k must lie in temperature_k_range")
        return self

    def terms_at(self, temperature_k: Decimal) -> list[tuple[Decimal, Decimal]]:
        """``(S_i, l_i^2)`` at a temperature (exact decimal)."""

        with localcontext(_CONTEXT):
            out = []
            for s_row, p_row in zip(self.s_coefficients, self.pole_coefficients_um):
                s_value = sum((Decimal(c) * temperature_k ** j for j, c in enumerate(s_row)), Decimal(0))
                pole = sum((Decimal(c) * temperature_k ** j for j, c in enumerate(p_row)), Decimal(0))
                out.append((s_value, pole * pole))
            return out


class TableMeasurementV1(StrictContractModel):
    """The conditions a measured table was taken at, as its source states them."""

    temperature_c: DecimalString | None = None
    basis: _Text


# ------------------------------------------------------------------------------------------------- bulk loss


class ExtinctionCoefficientLossV1(StrictContractModel):
    """Measured extinction coefficient ``k`` (the imaginary part of the complex index) at the wavelengths ``wavelength_nm``
    (on ``l_law``); ``alpha = 4 pi k / l`` with ``l`` the queried vacuum wavelength. Linear in the measured ``k``
    between points on the law's wavelength, then converted to alpha; nothing outside them."""

    kind: Literal["EXTINCTION_COEFFICIENT"] = "EXTINCTION_COEFFICIENT"
    wavelength_nm: tuple[DecimalString, ...] = Field(min_length=1, max_length=MAX_TABLE_POINTS)
    k: tuple[DecimalString, ...] = Field(min_length=1, max_length=MAX_TABLE_POINTS)
    basis: _Text

    @model_validator(mode="after")
    def _table(self) -> ExtinctionCoefficientLossV1:
        _check_rows(self.wavelength_nm, self.k, "k")
        if any(Decimal(v) < 0 for v in self.k):
            raise ValueError("k cannot be negative")
        return self


class InternalTransmittanceLossV1(StrictContractModel):
    """Measured internal transmittance ``T_i`` (bulk only: no surface reflection, no coating) of a sample of thickness
    ``thickness_mm`` at each wavelength ``wavelength_nm`` (on ``l_law``); ``alpha = -ln(T_i) / d``, which scales to any
    thickness ``L`` as ``T_i(L) = exp(-alpha L)``. Linear in ``alpha`` between points, nothing outside them."""

    kind: Literal["INTERNAL_TRANSMITTANCE"] = "INTERNAL_TRANSMITTANCE"
    wavelength_nm: tuple[DecimalString, ...] = Field(min_length=1, max_length=MAX_TABLE_POINTS)
    internal_transmittance: tuple[DecimalString, ...] = Field(min_length=1, max_length=MAX_TABLE_POINTS)
    thickness_mm: tuple[DecimalString, ...] = Field(min_length=1, max_length=MAX_TABLE_POINTS)
    basis: _Text

    @model_validator(mode="after")
    def _table(self) -> InternalTransmittanceLossV1:
        _check_rows(self.wavelength_nm, self.internal_transmittance, "internal_transmittance")
        if len(self.thickness_mm) != len(self.wavelength_nm):
            raise ValueError("thickness_mm must state one thickness per wavelength")
        if any(not Decimal(0) < Decimal(v) <= 1 for v in self.internal_transmittance):
            raise ValueError("internal_transmittance must lie in (0, 1]")
        if any(Decimal(v) <= 0 for v in self.thickness_mm):
            raise ValueError("thickness_mm must be positive")
        return self


BulkLossV1 = Annotated[ExtinctionCoefficientLossV1 | InternalTransmittanceLossV1, Field(discriminator="kind")]


def _check_rows(wavelengths: tuple[str, ...], values: tuple[str, ...], label: str) -> None:
    if len(wavelengths) != len(values):
        raise ValueError(f"wavelength_nm and {label} must have equal length")
    points = [Decimal(w) for w in wavelengths]
    if any(w <= 0 for w in points):
        raise ValueError("wavelength_nm must be positive")
    if any(b <= a for a, b in zip(points, points[1:])):
        raise ValueError("wavelength_nm must be strictly increasing")


def loss_alpha_per_mm(loss: ExtinctionCoefficientLossV1 | InternalTransmittanceLossV1) -> list[tuple[float, float]]:
    """``(l_law nm, alpha per mm)`` of each row of a loss table; for ``k`` the vacuum wavelength is taken equal to
    ``l_law`` here (callers with an air-wavelength law convert: ``alpha = 4 pi k / l_vacuum``)."""

    if isinstance(loss, ExtinctionCoefficientLossV1):
        return [(float(w), 4.0 * math.pi * float(k) / (float(w) * 1e-6)) for w, k in zip(loss.wavelength_nm, loss.k)]
    return [(float(w), -math.log(float(t)) / float(d))
            for w, t, d in zip(loss.wavelength_nm, loss.internal_transmittance, loss.thickness_mm)]


# ----------------------------------------------------------------------------------------------- temperature


_SchottNumber = TypeVar("_SchottNumber", float, Decimal)


def schott_e_term(wavelength: _SchottNumber, lambda_tk_um: _SchottNumber, numerator: _SchottNumber, *,
                  wavelength_is_nm: bool = False) -> _SchottNumber:
    """The Schott E term in the caller's arithmetic, with an inclusive half-last-decimal-unit exclusion around its pole.

    The wavelength is the law's coordinate in micrometres or, with ``wavelength_is_nm``, nanometres. Compare its decimal
    value against the exact endpoints before converting units in the caller's arithmetic. Pole precision comes from the
    Engine's shortest float spelling. A zero numerator leaves a removable pole. The base dispersion band is independent.
    """

    if numerator == 0:
        return abs(numerator)
    pole = lambda_tk_um.copy_abs() if isinstance(lambda_tk_um, Decimal) else abs(lambda_tk_um)
    float_pole = float(pole)
    near_pole = False
    if 0.0 < float_pole < math.inf:
        last_place = Decimal(str(float_pole)).as_tuple().exponent
        margin = Decimal((0, (5,), last_place - 1))
        law_coordinate = Decimal(str(wavelength))
        with localcontext(_CONTEXT):
            center = Decimal(str(pole))
            lower, upper = center - margin, center + margin
            if wavelength_is_nm:
                lower, upper = lower.scaleb(3), upper.scaleb(3)
        near_pole = law_coordinate.is_finite() and lower <= law_coordinate <= upper
    law_um = wavelength
    if wavelength_is_nm:
        law_um = wavelength / 1000 if isinstance(wavelength, Decimal) else wavelength * 0.001
    denominator = law_um * law_um - lambda_tk_um * lambda_tk_um
    if denominator == 0 or near_pole:
        law_nm = wavelength if wavelength_is_nm else law_um * 1000
        raise ValueError(f"Schott thermo-optic pole neighbourhood at {law_nm:.9g} nm "
                         f"(lambda_tk = {lambda_tk_um * 1000:.9g} nm on the law's wavelength); "
                         "the base dispersion remains available at the reference temperature")
    return numerator / denominator


class SchottThermoOpticV1(StrictContractModel):
    """The Schott formula of the change of the absolute index with temperature (the glass catalogues' TD data):
    ``dn_abs(l, T) = (n^2 - 1) / (2 n) (D0 dT + D1 dT^2 + D2 dT^3 + (E0 dT + E1 dT^2) / (l^2 - l_tk^2))`` with
    ``dT = T - T0``, ``n`` the index the law gives at ``T0`` and ``l`` its wavelength (um, ``l_law``).
    ``valid_temperature_c``: the temperatures the source states the coefficients for (absent: not stated). A wavelength
    neighbourhood of ``lambda_tk_um`` is not a valid temperature correction when its E-term numerator is nonzero, even
    when it lies within the base dispersion's band; a zero E numerator is removable, and at ``T0`` the base remains available."""

    kind: Literal["SCHOTT_DN_DT"] = "SCHOTT_DN_DT"
    d0: DecimalString
    d1: DecimalString
    d2: DecimalString
    e0: DecimalString
    e1: DecimalString
    lambda_tk_um: DecimalString
    reference_temperature_c: DecimalString
    valid_temperature_c: tuple[DecimalString, DecimalString] | None = None
    basis: _Text

    def delta_n(self, n: float, law_um: float, temperature_c: float) -> float:
        dt = temperature_c - float(self.reference_temperature_c)
        if dt == 0.0:
            return 0.0
        numerator = float(self.e0) * dt + float(self.e1) * dt * dt
        e_term = schott_e_term(law_um, float(self.lambda_tk_um), numerator)
        return (n * n - 1.0) / (2.0 * n) * (float(self.d0) * dt + float(self.d1) * dt * dt + float(self.d2) * dt * dt * dt + e_term)


ThermoOpticV1 = Annotated[SchottThermoOpticV1, Field(discriminator="kind")]

# ---------------------------------------------------------------------------------------------------- identity

DesignAvailability = Literal["CURRENT", "LEGACY", "NOT_STATED"]


class OrderingStatusV1(StrictContractModel):
    """The maker's ordering status of a grade as of an edition. It never blocks designing with the material: a legacy grade
    stays usable to reproduce designs."""

    code: _Name
    design_availability: DesignAvailability
    as_of: _Text


class MaterialIdentityV1(OmitAbsentFields, StrictContractModel):
    """Who makes the material and what it is called (module docstring). ``designation``: the maker's exact name (for a
    literature model, the substance's name); ``aliases``: documented other spellings; ``catalogue_tags``: the maker code and
    catalogue names a prescription's catalogue list may give; ``variant_of``: the base record of a variant (an anneal-rate
    variant, ``variant``); ``source_edition``: the edition of the source the optical data come from."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("manufacturer", "variant_of", "variant", "ordering_status")

    manufacturer: _Name | None = None
    designation: _Name
    aliases: tuple[_Name, ...] = ()
    catalogue_tags: tuple[CatalogueTag, ...] = ()
    variant_of: _MaterialRef | None = None
    variant: _Text | None = None
    ordering_status: OrderingStatusV1 | None = None
    source_edition: _Text

    @model_validator(mode="after")
    def _consistent(self) -> MaterialIdentityV1:
        if (self.variant_of is None) != (self.variant is None):
            raise ValueError("variant_of and variant are stated together")
        if len({a.upper() for a in self.aliases}) != len(self.aliases):
            raise ValueError("aliases must be distinct")
        if list(self.catalogue_tags) != sorted(set(self.catalogue_tags)):
            raise ValueError("catalogue_tags must be distinct and sorted")
        return self


class CatalogCopyV1(StrictContractModel):
    """Where a Project's copy of a catalog material comes from: the catalog release, the material id chosen in it and the
    edition of the source its optical data come from. The
    copy itself carries the complete optical definition, so the Project reloads without the catalog."""

    catalog_version: Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}\.[1-9][0-9]*$")]
    material_id: _MaterialRef
    source_edition: _Text


# --------------------------------------------------------------------------------------------------- resolver

_LOOSE_DROP = str.maketrans("", "", " -_.")


def loose_name(text: str) -> str:
    """A name with case, spaces, ``-``, ``_`` and ``.`` ignored (``nbk7`` == ``N-BK7``)."""

    return text.upper().translate(_LOOSE_DROP)


@dataclass(frozen=True)
class MaterialResolution:
    """How a name resolved: the material ids it names (exactly one when resolved), how, and the candidates of an
    ambiguous name. A bare designation shared with a stable id is ambiguous; ``ID:<id>`` selects that id explicitly."""

    material_ids: tuple[str, ...]
    how: str  # material_id (explicit) | material_id | manufacturer designation | catalogue order | designation | loose match | ambiguous | none
    candidates: tuple[str, ...] = ()
    also_matches: tuple[str, ...] = field(default=())

    @property
    def resolved(self) -> str | None:
        return self.material_ids[0] if len(self.material_ids) == 1 and self.how != "ambiguous" else None

    @property
    def ambiguous(self) -> bool:
        return self.how == "ambiguous"


@dataclass(frozen=True)
class NamedMaterial:
    """What name resolution reads of a material: its id and identity (a ``PublicMaterialV1``, a library record, ...)."""

    material_id: str
    designation: str | None
    manufacturer: str | None = None
    aliases: tuple[str, ...] = ()
    catalogue_tags: tuple[str, ...] = ()

    @classmethod
    def of(cls, material: Any) -> NamedMaterial:
        identity = getattr(material, "identity", None)
        if identity is None:
            return cls(material.material_id, None)
        return cls(material.material_id, identity.designation, identity.manufacturer, tuple(identity.aliases), tuple(identity.catalogue_tags))

    def names(self) -> list[str]:
        return [n for n in (self.designation, *self.aliases) if n]

    def made_by(self, manufacturer: str) -> bool:
        wanted = manufacturer.strip().upper()
        return bool(wanted) and (wanted == (self.manufacturer or "").upper() or wanted in {t.upper() for t in self.catalogue_tags})


class MaterialNameIndex:
    """The names of a set of materials, and the one rule that resolves a name among them (``resolve``).

    1. ``ID:<id>`` selects that stable id explicitly, without reading it as a designation or alias.
    2. ``manufacturer``: that maker's record answering to the name (its id, designation or an alias), else none; several:
       ``ambiguous``.
    3. A name several records answer to (ids, designations, aliases) resolves through ``catalogue_order`` when the input
       format supplies one (a prescription's documented catalogue list: maker codes or catalogue names, in order); without
       a deciding maker or order it is ``ambiguous`` (including ``F2``: SCHOTT's stable id and CDGM's designation).
    4. A unique exact id, designation or alias resolves to its record.
    5. ``loose`` (a lookup convenience, never a prescription's glass): one loose match (``loose_name``); several: ``ambiguous``.
    """

    def __init__(self, materials: Iterable[Any]) -> None:
        self.materials: dict[str, NamedMaterial] = {}
        for material in materials:
            named = material if isinstance(material, NamedMaterial) else NamedMaterial.of(material)
            self.materials[named.material_id.upper()] = named
        self._by_name: dict[str, set[str]] = {}
        self._by_loose: dict[str, set[str]] = {}
        for key, named in self.materials.items():
            for name in named.names():
                self._by_name.setdefault(name.upper(), set()).add(key)
            for name in (key, *named.names()):
                self._by_loose.setdefault(loose_name(name), set()).add(key)

    def resolve(self, text: str, *, manufacturer: str | None = None, catalogue_order: Iterable[str] | None = None,
                loose: bool = False) -> MaterialResolution:
        key = " ".join(text.split()).upper()
        if not key:
            return MaterialResolution((), "none")
        if key.startswith("ID:"):
            material_id = key[3:].strip()
            return (MaterialResolution((material_id,), "material_id (explicit)") if material_id in self.materials
                    else MaterialResolution((), "none"))
        named = sorted(self._by_name.get(key, set()))
        exact = [key] if key in self.materials else []
        if manufacturer is not None:
            hits = tuple(m for m in sorted(set(named) | set(exact)) if self.materials[m].made_by(manufacturer))
            if not hits and loose:
                hits = tuple(m for m in sorted(self._by_loose.get(loose_name(key), set())) if self.materials[m].made_by(manufacturer))
            how = "manufacturer designation" if len(hits) == 1 else ("ambiguous" if hits else "none")
            return MaterialResolution(hits, how, candidates=hits if len(hits) > 1 else ())
        shared = sorted(set(named) | set(exact))
        order = [t.strip().upper() for t in catalogue_order or () if t.strip()]
        if order and len(shared) > 1:
            for tag in order:
                hits = [m for m in shared if tag in {t.upper() for t in self.materials[m].catalogue_tags}
                        or tag == (self.materials[m].manufacturer or "").upper()]
                if len(hits) == 1:
                    return MaterialResolution((hits[0],), "catalogue order", candidates=tuple(shared))
                if len(hits) > 1:
                    return MaterialResolution(tuple(hits), "ambiguous", candidates=tuple(hits))
        if len(shared) > 1:
            return MaterialResolution(tuple(shared), "ambiguous", candidates=tuple(shared))
        if exact:
            return MaterialResolution((key,), "material_id")
        if named:
            return MaterialResolution((named[0],), "designation")
        if loose:
            hits = sorted(self._by_loose.get(loose_name(key), set()))
            if len(hits) == 1:
                return MaterialResolution((hits[0],), "loose match")
            if hits:
                return MaterialResolution(tuple(hits), "ambiguous", candidates=tuple(hits))
        return MaterialResolution((), "none")


def resolve_material_name(text: str, materials: Iterable[Any], *, manufacturer: str | None = None,
                          catalogue_order: Iterable[str] | None = None, loose: bool = False) -> MaterialResolution:
    """``MaterialNameIndex(materials).resolve(...)`` (build the index once to resolve many names)."""

    return MaterialNameIndex(materials).resolve(text, manufacturer=manufacturer, catalogue_order=catalogue_order, loose=loose)


__all__ = [
    "AIR_MODEL_BAND_NM", "AIR_MODEL_EDLEN_1953", "AIR_MODEL_FIT_BAND_NM", "BAND_QUANTUM_NM", "HERZBERGER_CONSTANT_UM2",
    "MAX_POWER_SERIES_EXPONENT", "MAX_POWER_SERIES_TERMS", "MAX_SELLMEIER_TERMS", "MAX_TABLE_POINTS", "STANDARD_PRESSURE_PA",
    "STANDARD_REFERENCE_AIR", "STANDARD_TEMPERATURE_C", "AirModel", "BulkLossV1", "CatalogCopyV1", "CatalogueTag",
    "ExtinctionCoefficientLossV1", "HerzbergerDispersionV1", "IndexReferenceV1", "InternalTransmittanceLossV1",
    "MaterialIdentityV1", "MaterialNameIndex", "MaterialResolution", "NamedMaterial", "OrderingStatusV1",
    "PowerSeriesDispersionV1", "ReferenceAirV1", "SchottThermoOpticV1", "TableMeasurementV1",
    "TemperatureSellmeierDispersionV1", "ThermoOpticV1", "air_index", "air_index_exact", "air_index_slope", "loose_name",
    "loss_alpha_per_mm", "resolve_material_name", "schott_e_term",
]
