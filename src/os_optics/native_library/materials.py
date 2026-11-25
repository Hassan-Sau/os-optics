"""Materials of native lenses, plates and cells: where their optical data comes from, and their refractive index.

A refractive native class takes its material in one of these forms (the ``material`` parameter); the library never
supplies glass data of its own:

* ``"N-BK7"`` or ``{"catalog": "N-BK7"}``: a copy of the public catalog's published material model (its Sellmeier or
  tabulated dispersion and valid band, ``basis`` naming the catalog release). Needs a catalog release on this device.
* ``{"constant_index": "1.5", "valid_band_nm": [400, 700]}``: a wavelength-independent index over a stated band.
* ``{"sellmeier": {"b": [b1, b2, b3], "c_um2": [c1, c2, c3]}, "valid_band_nm": [min, max]}``: a three-term Sellmeier
  formula ``n^2 = 1 + sum b_i l^2 / (l^2 - c_i)`` (``l`` in um) over a stated band.
* a complete ``ProjectModelMaterialV1`` document (``material_id``, ``display_name``, ``dispersion``, ``valid_band_nm``).

A crystal class (a beam displacer, a Glan polarizer) takes a uniaxial crystal (``resolve_crystal``): a complete
``ProjectModelMaterialV1`` document with ``birefringence``, or ``{"ordinary": INDEX, "extraordinary": INDEX, "valid_band_nm":
[min, max]}`` with each ``INDEX`` ``{"constant_index": n}`` or ``{"sellmeier": {"b": [..3], "c_um2": [..3]}}`` (default id
``CRYSTAL-<digest>``). ``extraordinary_index(wavelength_nm)`` evaluates its extraordinary index like ``index`` the ordinary one.

The declared forms also take ``material_id`` (default: ``INDEX-<n>`` / ``SELLMEIER-<digest>``), ``display_name``,
``absorption_coefficient_per_mm`` (bulk absorption ``alpha``: power decays as ``exp(-alpha * path)``) and ``basis``.

``index(wavelength_nm)`` evaluates the material at the Project's default temperature, 20 C, with the Engine's index
rule in decimal arithmetic: a Sellmeier sum and square root, PCHIP through tabulated points, a power series, Herzberger
or temperature Sellmeier law at 293.15 K; ``index_reference`` converts air wavelengths and relative indices to the
absolute index, then ``thermo_optic`` corrects that absolute index from its reference temperature. The result is relative
to the **design air** -- air at 20 C and 101325 Pa, the default Project environment -- so the native lens design and its
saved Project run have the same refractive power. It is defined only inside the material's valid wavelength and
applicable temperature bands. Materials without thermal data retain their source index unchanged at 20 C.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.design_catalog import (
    FactStatus,
    PublicOpticalCatalogV2,
    SellmeierDispersionV1,
    TabulatedDispersionV1,
    WavelengthBandV1,
)
from .._vendor.shared_contracts.design_catalog.material_physics import (
    HerzbergerDispersionV1,
    PowerSeriesDispersionV1,
    TemperatureSellmeierDispersionV1,
    air_index_exact,
    schott_e_term,
)
from .._vendor.shared_contracts.optical_design import ConstantIndexDispersionV1, ProjectModelMaterialV1
from ..catalog_copy import copy_catalog_material
from .errors import NativeComponentError, NativeErrorCode, NativeProblem, invalid
from .numbers import NM_PER_UM, ONE, TWO, WORK_CONTEXT, ZERO, text, to_decimal

#: Length of the digest in a generated Sellmeier material id.
MATERIAL_DIGEST_LENGTH = 10
DECLARED_KEYS = frozenset({"material_id", "display_name", "absorption_coefficient_per_mm", "basis", "valid_band_nm"})
CRYSTAL_LAWS = ("ordinary", "extraordinary")
THREE = Decimal(3)


@dataclass(frozen=True)
class ResolvedMaterial:
    """A material ready to be written into a Project (``material``) and evaluated (``index``)."""

    material: ProjectModelMaterialV1
    from_catalog: bool
    catalog_version: str | None = None

    @property
    def material_id(self) -> str:
        return self.material.material_id

    @property
    def band(self) -> tuple[Decimal, Decimal]:
        return Decimal(self.material.valid_band_nm.min_nm), Decimal(self.material.valid_band_nm.max_nm)

    def index(self, wavelength_nm: Decimal) -> Decimal:
        """Refractive index at the vacuum wavelength ``wavelength_nm`` relative to the design air (module docstring);
        ``ValueError`` outside the valid band."""

        low, high = self.band
        if not low <= wavelength_nm <= high:
            raise ValueError(f"{text(wavelength_nm)} nm is outside the valid band [{text(low)}, {text(high)}] nm of material "
                             f"{self.material_id}")
        # The Engine may narrow a temperature law's measured domain or refuse an unstable thermal correction within an
        # otherwise valid band. Apply its availability rule too; keep Decimal arithmetic for the actual design index.
        if self.material.thermo_optic is not None or isinstance(self.material.dispersion, TemperatureSellmeierDispersionV1):
            from ..config import DEFAULT_NUMERICAL_CONFIG
            from ..local.compile import engine_material
            from ..optics.materials.index import build_evaluator

            engine = engine_material(self.material, "", temperature_c=float(DESIGN_AIR_TEMPERATURE_C))
            _n, issue = build_evaluator(engine, DEFAULT_NUMERICAL_CONFIG, float(DESIGN_AIR_TEMPERATURE_C)).index(float(wavelength_nm))
            if issue is not None:
                raise ValueError(issue.message)
        return design_index(self.material, self.material.dispersion, wavelength_nm)

    def extraordinary_index(self, wavelength_nm: Decimal) -> Decimal:
        """Extraordinary index of a crystal at ``wavelength_nm`` (``ValueError`` outside the band or for an isotropic material)."""

        if self.material.birefringence is None:
            raise ValueError(f"material {self.material_id} is not a crystal")
        self.index(wavelength_nm)  # (the band check)
        return design_index(self.material, self.material.birefringence.extraordinary, wavelength_nm)


#: The air and material temperature native designs use: a Project's default environment (20 C, 101325 Pa).
DESIGN_AIR_TEMPERATURE_C = Decimal(20)
DESIGN_AIR_PRESSURE_PA = Decimal(101325)
KELVIN_OFFSET = Decimal("273.15")


def design_air_index(wavelength_nm: Decimal) -> Decimal:
    """The absolute index of the design air at a vacuum wavelength (``air_index_exact``)."""

    with localcontext(WORK_CONTEXT):
        return air_index_exact(wavelength_nm, DESIGN_AIR_TEMPERATURE_C, DESIGN_AIR_PRESSURE_PA)


def absolute_index(material: Any, dispersion: Any, wavelength_nm: Decimal) -> Decimal:
    """The absolute index at a vacuum wavelength and default Project temperature: law wavelength, reference air,
    then Schott thermal correction on the law's index and wavelength (the Engine's composition order)."""

    reference = getattr(material, "index_reference", None)
    thermo = getattr(material, "thermo_optic", None)
    with localcontext(WORK_CONTEXT):
        n_ref = (reference.reference_air.index_exact(wavelength_nm)
                 if reference is not None and (reference.air_wavelength or reference.relative_to_air) else ONE)
        law_nm = wavelength_nm / n_ref if reference is not None and reference.air_wavelength else wavelength_nm
        n_law = dispersion_index(dispersion, law_nm)
        n_abs = n_law * n_ref if reference is not None and reference.relative_to_air else n_law
        if thermo is not None:
            if thermo.valid_temperature_c is not None:
                low, high = (Decimal(v) for v in thermo.valid_temperature_c)
                if not low <= DESIGN_AIR_TEMPERATURE_C <= high:
                    raise ValueError(f"material {material.material_id}: thermo-optic coefficients do not hold at "
                                     f"{text(DESIGN_AIR_TEMPERATURE_C)} C (valid {text(low)}-{text(high)} C)")
            dt = DESIGN_AIR_TEMPERATURE_C - Decimal(thermo.reference_temperature_c)
            if dt:
                numerator = Decimal(thermo.e0) * dt + Decimal(thermo.e1) * dt * dt
                e_term = schott_e_term(law_nm, Decimal(thermo.lambda_tk_um), numerator, wavelength_is_nm=True)
                correction = (n_law * n_law - ONE) / (TWO * n_law) * (
                    Decimal(thermo.d0) * dt + Decimal(thermo.d1) * dt * dt + Decimal(thermo.d2) * dt ** 3 + e_term)
                n_abs += correction
        return n_abs


def design_index(material: Any, dispersion: Any, wavelength_nm: Decimal) -> Decimal:
    """``absolute_index`` relative to the design air (module docstring)."""

    with localcontext(WORK_CONTEXT):
        return absolute_index(material, dispersion, wavelength_nm) / design_air_index(wavelength_nm)


def dispersion_index(dispersion: Any, wavelength_nm: Decimal) -> Decimal:
    """The value of a law at its own wavelength (nm): constant, Sellmeier (any number of terms), power series, Herzberger,
    PCHIP table or temperature law at the Project's default 293.15 K."""

    with localcontext(WORK_CONTEXT):
        if isinstance(dispersion, ConstantIndexDispersionV1):
            return Decimal(dispersion.refractive_index)
        if isinstance(dispersion, PowerSeriesDispersionV1):
            n_sq = dispersion.n_squared_exact(wavelength_nm / NM_PER_UM)
            if n_sq <= 0:
                raise ValueError(f"the power series gives n^2 <= 0 at {text(wavelength_nm)} nm")
            return n_sq.sqrt()
        if isinstance(dispersion, HerzbergerDispersionV1):
            return dispersion.index_exact(wavelength_nm / NM_PER_UM)
        if isinstance(dispersion, TemperatureSellmeierDispersionV1):
            temperature_k = DESIGN_AIR_TEMPERATURE_C + KELVIN_OFFSET
            low, high = (Decimal(v) for v in dispersion.temperature_k_range)
            if not low <= temperature_k <= high:
                raise ValueError(f"the temperature law holds for {text(low)}-{text(high)} K, not {text(temperature_k)} K")
            lam_sq = (wavelength_nm / NM_PER_UM) ** 2
            total = ONE
            for strength, pole_sq in dispersion.terms_at(temperature_k):
                if lam_sq == pole_sq:
                    raise ValueError(f"the temperature law has a pole at {text(wavelength_nm)} nm and {text(temperature_k)} K")
                total += strength * lam_sq / (lam_sq - pole_sq)
            if total <= 0:
                raise ValueError(f"the temperature law gives n^2 <= 0 at {text(wavelength_nm)} nm and {text(temperature_k)} K")
            return total.sqrt()
        if isinstance(dispersion, SellmeierDispersionV1):
            lam_sq = (wavelength_nm / NM_PER_UM) ** 2
            total = ONE
            for b, c in zip(dispersion.b, dispersion.c_um2, strict=True):
                denominator = lam_sq - Decimal(c)
                if denominator == 0:
                    raise ValueError(f"the Sellmeier formula has a pole at {text(wavelength_nm)} nm")
                total += Decimal(b) * lam_sq / denominator
            if total <= 0:
                raise ValueError(f"the Sellmeier formula gives n^2 <= 0 at {text(wavelength_nm)} nm")
            return total.sqrt()
        return _pchip(dispersion, wavelength_nm)


def _sign(value: Decimal) -> int:
    return (value > 0) - (value < 0)


def _edge_derivative(h0: Decimal, h1: Decimal, m0: Decimal, m1: Decimal) -> Decimal:
    """SciPy ``PchipInterpolator`` end condition: a one-sided three-point estimate, kept shape-preserving."""

    d = ((TWO * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if _sign(d) != _sign(m0):
        return ZERO
    if _sign(m0) != _sign(m1) and abs(d) > THREE * abs(m0):
        return THREE * m0
    return d


def _pchip(dispersion: TabulatedDispersionV1, wavelength_nm: Decimal) -> Decimal:
    x = [Decimal(v) for v in dispersion.wavelength_nm]
    y = [Decimal(v) for v in dispersion.refractive_index]
    if not x[0] <= wavelength_nm <= x[-1]:
        raise ValueError(f"{text(wavelength_nm)} nm is outside the tabulated range")
    h = [b - a for a, b in zip(x, x[1:])]
    m = [(y[k + 1] - y[k]) / h[k] for k in range(len(h))]
    if len(x) == 2:
        d = [m[0], m[0]]
    else:
        d = [ZERO] * len(x)
        for k in range(1, len(x) - 1):
            if _sign(m[k - 1]) != _sign(m[k]) or m[k - 1] == 0 or m[k] == 0:
                d[k] = ZERO
            else:
                w1 = TWO * h[k] + h[k - 1]
                w2 = h[k] + TWO * h[k - 1]
                d[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k])
        d[0] = _edge_derivative(h[0], h[1], m[0], m[1])
        d[-1] = _edge_derivative(h[-1], h[-2], m[-1], m[-2])
    k = next(i for i in range(len(h)) if wavelength_nm <= x[i + 1])
    t = (wavelength_nm - x[k]) / h[k]
    t2, t3 = t * t, t * t * t
    h00 = TWO * t3 - THREE * t2 + ONE
    h10 = t3 - TWO * t2 + t
    h01 = -TWO * t3 + THREE * t2
    h11 = t3 - t2
    return h00 * y[k] + h10 * h[k] * d[k] + h01 * y[k + 1] + h11 * h[k] * d[k + 1]


def _contract_message(exc: ValidationError) -> str:
    """The contract's first refusal of a material, as a customer reads it (field path, without pydantic's prefix)."""

    first = exc.errors(include_url=False)[0]
    message = str(first["msg"]).removeprefix("Value error, ")
    where = ".".join(str(p) for p in first["loc"])
    return f"{where}: {message}" if where else message


def _digest(document: object) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:MATERIAL_DIGEST_LENGTH].upper()


def _band(raw: Any, parameter: str) -> WavelengthBandV1:
    if isinstance(raw, Mapping):
        raw = [raw.get("min_nm"), raw.get("max_nm")]
    if isinstance(raw, str):
        raw = raw.split(",")
    values = [to_decimal(v) for v in raw] if isinstance(raw, (list, tuple)) and len(raw) == 2 else None
    if values is None or any(v is None or v <= 0 for v in values) or not values[0] < values[1]:  # type: ignore[operator]
        raise invalid("valid_band_nm must be [min_nm, max_nm] with 0 < min_nm < max_nm", parameter)
    return WavelengthBandV1(min_nm=text(values[0]), max_nm=text(values[1]), status=FactStatus.DESIGN_BASIS)  # type: ignore[arg-type]


class MaterialResolver:
    """Resolves material parameters; reads the public catalog only when a catalog material is named."""

    def __init__(self, catalog: PublicOpticalCatalogV2 | None = None) -> None:
        self._catalog = catalog

    def catalog(self, parameter: str) -> PublicOpticalCatalogV2:
        """The public catalog release (loaded once, on first use)."""

        if self._catalog is None:
            from ..local.catalog import load_public_catalog
            from ..local.errors import LocalRunError

            try:
                self._catalog = load_public_catalog()
            except LocalRunError as exc:
                raise NativeComponentError([NativeProblem(
                    NativeErrorCode.CATALOG_DATA_UNAVAILABLE,
                    "a catalog material or coating needs a public catalog release on this device (os-optics catalog sync); "
                    f"or declare it by its own values ({exc.problems[0].message})", parameter)]) from None
        return self._catalog

    def resolve(self, spec: Any, parameter: str) -> ResolvedMaterial:
        if isinstance(spec, str):
            spec = {"catalog": spec}
        if not isinstance(spec, Mapping):
            raise invalid("expected a catalog material id or a material object", parameter)
        if "catalog" in spec:
            if set(spec) != {"catalog"} or not isinstance(spec["catalog"], str):
                raise invalid('a catalog material is {"catalog": "<MATERIAL_ID>"} (nothing else)', parameter)
            return self._from_catalog(spec["catalog"], parameter)
        if "dispersion" in spec:
            try:
                return ResolvedMaterial(ProjectModelMaterialV1.model_validate(dict(spec)), from_catalog=False)
            except ValidationError as exc:
                raise invalid(f"not a valid material document: {_contract_message(exc)}", parameter) from None
        return self._declared(spec, parameter)

    def resolve_crystal(self, spec: Any, parameter: str) -> ResolvedMaterial:
        """A uniaxial crystal (module docstring); ``NATIVE_PARAMETER_INVALID`` for anything else."""

        help_text = ('a crystal is a material document with birefringence, or {"ordinary": INDEX, "extraordinary": INDEX, '
                     '"valid_band_nm": [min, max]} with INDEX {"constant_index": n} or {"sellmeier": {"b": [..3], "c_um2": [..3]}}')
        if not isinstance(spec, Mapping):
            raise invalid(help_text, parameter)
        if "dispersion" in spec:
            resolved = self.resolve(spec, parameter)
            if not resolved.material.is_crystal:
                raise invalid(f"material {resolved.material_id} states no birefringence: {help_text}", parameter)
            return resolved
        if set(CRYSTAL_LAWS) - set(spec) or set(spec) - DECLARED_KEYS - set(CRYSTAL_LAWS):
            raise invalid(f"{help_text} (optional: material_id, display_name, absorption_coefficient_per_mm, basis); got keys {sorted(spec)}",
                          parameter)
        if "valid_band_nm" not in spec:
            raise NativeComponentError([NativeProblem(NativeErrorCode.PARAMETER_MISSING,
                                                      "a declared crystal states the band it is valid in (valid_band_nm)", parameter)])
        band = _band(spec["valid_band_nm"], parameter)
        laws = {}
        for name in CRYSTAL_LAWS:
            law = spec[name]
            if isinstance(law, Mapping) and set(law) == {"constant_index"}:
                index = to_decimal(law["constant_index"])
                if index is None or index <= 0:
                    raise invalid(f"{name}.constant_index must be a positive decimal", parameter)
                laws[name] = {"kind": "CONSTANT_INDEX", "refractive_index": text(index)}
            elif isinstance(law, Mapping) and set(law) == {"sellmeier"} and isinstance(law["sellmeier"], Mapping)                     and set(law["sellmeier"]) == {"b", "c_um2"}:
                terms = []
                for key in ("b", "c_um2"):
                    values = law["sellmeier"][key]
                    numbers = [to_decimal(v) for v in values] if isinstance(values, (list, tuple)) and len(values) == 3 else None
                    if numbers is None or any(v is None for v in numbers):
                        raise invalid(f"{name}.sellmeier.{key} must be three decimals", parameter)
                    terms.append([text(v) for v in numbers])  # type: ignore[arg-type]
                laws[name] = {"kind": "SELLMEIER", "b": terms[0], "c_um2": terms[1]}
            else:
                raise invalid(f"{name}: expected {{\"constant_index\": n}} or {{\"sellmeier\": {{\"b\": [..3], \"c_um2\": [..3]}}}}", parameter)
        default_id = f"CRYSTAL-{_digest({'o': laws['ordinary'], 'e': laws['extraordinary'], 'band': [band.min_nm, band.max_nm]})}"
        document: dict[str, Any] = {
            "material_id": spec.get("material_id", default_id), "display_name": spec.get("display_name", "Declared uniaxial crystal"),
            "dispersion": laws["ordinary"], "valid_band_nm": band.model_dump(mode="json"),
            "basis": spec.get("basis", "Declared ordinary and extraordinary indices"),
            "birefringence": {"kind": "UNIAXIAL", "extraordinary": laws["extraordinary"]},
        }
        if spec.get("absorption_coefficient_per_mm") is not None:
            alpha = to_decimal(spec["absorption_coefficient_per_mm"])
            if alpha is None or alpha < 0:
                raise invalid("absorption_coefficient_per_mm must be a non-negative decimal", parameter)
            document["absorption_coefficient_per_mm"] = text(alpha)
        try:
            material = ProjectModelMaterialV1.model_validate(document)
        except ValidationError as exc:
            raise invalid(_contract_message(exc), parameter) from None
        return ResolvedMaterial(material, from_catalog=False)

    def _from_catalog(self, material_id: str, parameter: str) -> ResolvedMaterial:
        catalog = self.catalog(parameter)
        published = catalog.material(material_id)
        if published is None:
            raise NativeComponentError([NativeProblem(
                NativeErrorCode.CATALOG_DATA_UNAVAILABLE,
                f"public catalog {catalog.catalog_version} has no material {material_id!r}; its materials are listed by "
                "catalog_materials()", parameter)])
        return ResolvedMaterial(copy_catalog_material(published, catalog.catalog_version), from_catalog=True,
                                catalog_version=catalog.catalog_version)

    def _declared(self, spec: Mapping[str, Any], parameter: str) -> ResolvedMaterial:
        kinds = [key for key in ("constant_index", "sellmeier") if key in spec]
        if len(kinds) != 1 or set(spec) - DECLARED_KEYS - set(kinds):
            raise invalid("a declared material has exactly one of constant_index and sellmeier, with valid_band_nm "
                          f"(optional: material_id, display_name, absorption_coefficient_per_mm, basis); got keys {sorted(spec)}",
                          parameter)
        if "valid_band_nm" not in spec:
            raise NativeComponentError([NativeProblem(NativeErrorCode.PARAMETER_MISSING,
                                                      "a declared material states the band it is valid in (valid_band_nm)", parameter)])
        band = _band(spec["valid_band_nm"], parameter)
        if kinds[0] == "constant_index":
            index = to_decimal(spec["constant_index"])
            if index is None or index <= 0:
                raise invalid("constant_index must be a positive decimal", parameter)
            dispersion: dict[str, Any] = {"kind": "CONSTANT_INDEX", "refractive_index": text(index)}
            default_id, default_name = f"INDEX-{text(index)}", f"Constant index {text(index)}"
            basis = "Declared constant refractive index"
        else:
            sellmeier = spec["sellmeier"]
            if not isinstance(sellmeier, Mapping) or set(sellmeier) != {"b", "c_um2"}:
                raise invalid('sellmeier is {"b": [b1, b2, b3], "c_um2": [c1, c2, c3]}', parameter)
            terms = []
            for key in ("b", "c_um2"):
                values = sellmeier[key]
                numbers = [to_decimal(v) for v in values] if isinstance(values, (list, tuple)) and len(values) == 3 else None
                if numbers is None or any(v is None for v in numbers):
                    raise invalid(f"sellmeier.{key} must be three decimals", parameter)
                terms.append([text(v) for v in numbers])  # type: ignore[arg-type]
            dispersion = {"kind": "SELLMEIER", "b": terms[0], "c_um2": terms[1]}
            default_id = f"SELLMEIER-{_digest({'d': dispersion, 'band': [band.min_nm, band.max_nm]})}"
            default_name, basis = "Declared Sellmeier material", "Declared Sellmeier dispersion formula"
        document: dict[str, Any] = {
            "material_id": spec.get("material_id", default_id),
            "display_name": spec.get("display_name", default_name),
            "dispersion": dispersion,
            "valid_band_nm": band.model_dump(mode="json"),
            "basis": spec.get("basis", basis),
        }
        if spec.get("absorption_coefficient_per_mm") is not None:
            alpha = to_decimal(spec["absorption_coefficient_per_mm"])
            if alpha is None or alpha < 0:
                raise invalid("absorption_coefficient_per_mm must be a non-negative decimal", parameter)
            document["absorption_coefficient_per_mm"] = text(alpha)
        try:  # the contract refuses an index that is not real and at least 1 throughout the band (a pole, n < 1)
            material = ProjectModelMaterialV1.model_validate(document)
        except ValidationError as exc:
            raise invalid(_contract_message(exc), parameter) from None
        return ResolvedMaterial(material, from_catalog=False)


def band_intersection(bands: list[tuple[Decimal, Decimal]], parameter: str) -> tuple[Decimal, Decimal]:
    """The common part of several bands; ``NATIVE_DESIGN_INFEASIBLE`` when they do not overlap."""

    low, high = max(b[0] for b in bands), min(b[1] for b in bands)
    if not low < high:
        raise NativeComponentError([NativeProblem(NativeErrorCode.DESIGN_INFEASIBLE,
                                                  "the materials / coatings of this element have no common valid band", parameter)])
    return low, high


__all__ = ["MaterialResolver", "ResolvedMaterial", "band_intersection", "dispersion_index"]
