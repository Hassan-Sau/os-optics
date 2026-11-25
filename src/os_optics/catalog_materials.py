"""Material discovery in the public optical catalog: find, resolve and compare published materials (``os-optics catalog
materials`` / ``catalog material``).

Every published material is summarized from its own record (``material_summary``): its identity (maker, exact
designation, aliases, catalogue tags, variant base, the maker's ordering status and the source edition; a material of a
catalog before schema 3.6 states none), its valid band (vacuum wavelengths), its dispersion law, the convention its source
states the law in (the wavelength it takes and what its index is relative to; ``NOT_STATED``: evaluated in the Engine's
convention, vacuum wavelength and absolute index), its measured bulk loss and the band that covers (absent: no stated
loss -- not a zero loss) and its thermo-optic data.

``find_materials`` filters by maker (``manufacturer`` or catalogue tag), designation (exact: the id, designation or a
documented alias; ``loose``: case, spaces, ``-``, ``_`` and ``.`` ignored), a text search, the vacuum band the material's
valid band must cover, stated bulk loss over that band, and the maker's design availability. ``resolve_name`` resolves a
name exactly as prescription import does (the shared resolver, ``material_physics.MaterialNameIndex``).

Selecting on infrared (or any) wavelengths uses the Engine's own material evaluation (``optics.materials.physics``: the
same evaluator the tracer uses; nothing is recomputed here), never the visible ``nd`` / ``Vd``:

* ``at_nm``: the absolute index (and group index, GDD where the law supports a derivative claim) at that vacuum wavelength
  and ``temperature_c``;
* ``band_nm`` ``(lo, hi)``: the index at both ends and the middle, and the dispersion figure ``(n_mid - 1) / (n_lo -
  n_hi)`` -- the Abbe-number analogue of that band (larger: less dispersion; e.g. 8-12 um: n at 8, 10 and 12 um);
* ``thickness_mm`` (with ``at_nm`` and / or ``band_nm``): the internal (bulk) transmittance of that thickness from the
  material's measured bulk loss -- ``null`` with the reason when the material states no loss there (never 100 %). Surface
  reflection and coatings are not part of it.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Iterable

from ._vendor.shared_contracts.design_catalog.material_physics import (
    ExtinctionCoefficientLossV1,
    MaterialNameIndex,
    NamedMaterial,
    loose_name,
)

#: What the summary says of a material without an index convention.
ENGINE_CONVENTION = "NOT_STATED: evaluated as vacuum wavelength and absolute index (the Engine convention)"
NO_LOSS = "no stated bulk loss (not a zero loss): the internal transmittance is not established"
#: The material temperature (C) when none is stated (the glass catalogues' reference temperature).
DEFAULT_TEMPERATURE_C = 20.0
#: The physical lower bound of Celsius temperature.
ABSOLUTE_ZERO_C = -273.15
#: Decimal places reported: an index (and group index), a GDD (mm^2 per mm), a transmittance, the dispersion figure.
INDEX_DIGITS = 9
GDD_DIGITS = 15
TRANSMITTANCE_DIGITS = 6
FIGURE_DIGITS = 4
IR_UNAVAILABLE = "this Engine does not provide material evaluation (optics.materials.physics); upgrade os-optics"


def _text(value: Decimal | str) -> str:
    return f"{Decimal(value).normalize():f}"


def _vacuum(material: Any, law_nm: str) -> str:
    reference = getattr(material, "index_reference", None)
    if reference is None or not reference.air_wavelength:
        return _text(law_nm)
    # Preserve the actual transformed endpoint: rounding a lower edge down by 0.001 nm would advertise loss that the
    # evaluator correctly refuses (for example HWS9 starts at 2500.6704516 nm, not 2500.670 nm).
    return _text(reference.vacuum_wavelength_exact(Decimal(law_nm)))


def loss_band_nm(material: Any) -> tuple[Decimal, Decimal] | None:
    """The vacuum band a material's bulk-loss rows cover (``None``: no stated loss)."""

    loss = getattr(material, "bulk_loss", None)
    if loss is None:
        return None
    return Decimal(_vacuum(material, loss.wavelength_nm[0])), Decimal(_vacuum(material, loss.wavelength_nm[-1]))


def _identity(material: Any) -> dict[str, Any] | None:
    identity = getattr(material, "identity", None)
    if identity is None:
        return None
    status = identity.ordering_status
    return {
        "manufacturer": identity.manufacturer, "designation": identity.designation, "aliases": list(identity.aliases),
        "catalogue_tags": list(identity.catalogue_tags), "variant_of": identity.variant_of, "variant": identity.variant,
        "ordering_status": None if status is None else {"code": status.code, "design_availability": status.design_availability,
                                                        "as_of": status.as_of},
        "source_edition": identity.source_edition,
    }


def _convention(material: Any) -> dict[str, Any]:
    reference = getattr(material, "index_reference", None)
    if reference is None:
        return {"wavelength": "NOT_STATED", "index": "NOT_STATED", "stated_by": "NOT_STATED", "evaluated_as": ENGINE_CONVENTION}
    air = reference.reference_air
    out: dict[str, Any] = {
        "wavelength": reference.wavelength, "index": reference.index, "stated_by": reference.stated_by,
        "reference_air": None if air is None else {"model": air.model, "temperature_c": air.temperature_c, "pressure_pa": air.pressure_pa},
        "source_band_nm": {"min_nm": reference.source_band_nm[0], "max_nm": reference.source_band_nm[1],
                           "on": "air wavelengths" if reference.air_wavelength else "vacuum wavelengths"},
    }
    if reference.wavelength == "NOT_STATED" and reference.index == "NOT_STATED":
        out["evaluated_as"] = ENGINE_CONVENTION
    return out


def _dispersion(material: Any) -> dict[str, Any]:
    law = material.dispersion
    out: dict[str, Any] = {"kind": law.kind}
    if law.kind == "SELLMEIER":
        out["terms"] = len(law.b)
    elif law.kind == "TABULATED":
        out.update(origin=law.origin or "SOURCE_FORMULA", points=len(law.wavelength_nm),
                   table_nm={"min_nm": law.wavelength_nm[0], "max_nm": law.wavelength_nm[-1]})
    elif law.kind == "TEMPERATURE_SELLMEIER":
        out.update(temperature_k_range=list(law.temperature_k_range), reference_temperature_k=law.reference_temperature_k)
    return out


def _loss(material: Any) -> dict[str, Any] | None:
    loss = getattr(material, "bulk_loss", None)
    if loss is None:
        return None
    band = loss_band_nm(material)
    out: dict[str, Any] = {"kind": loss.kind, "points": len(loss.wavelength_nm), "basis": loss.basis,
                           "band_nm": {"min_nm": _text(band[0]), "max_nm": _text(band[1])}}  # type: ignore[index]
    if not isinstance(loss, ExtinctionCoefficientLossV1):
        out["sample_thickness_mm"] = sorted(set(loss.thickness_mm), key=Decimal)
    return out


def _thermo(material: Any) -> dict[str, Any] | None:
    thermo = getattr(material, "thermo_optic", None)
    if thermo is None:
        if material.dispersion.kind == "TEMPERATURE_SELLMEIER":
            return {"kind": "TEMPERATURE_SELLMEIER (in the dispersion law)", "temperature_k_range": list(material.dispersion.temperature_k_range)}
        return None
    return {"kind": thermo.kind, "reference_temperature_c": thermo.reference_temperature_c,
            "valid_temperature_c": None if thermo.valid_temperature_c is None else list(thermo.valid_temperature_c)}


def material_summary(material: Any) -> dict[str, Any]:
    """A published material as an agent selects it (module docstring)."""

    band = material.valid_band_nm
    return {
        "material_id": material.material_id, "display_name": material.display_name, "status": getattr(material.status, "value", material.status),
        "identity": _identity(material),
        "valid_band_nm": {"min_nm": band.min_nm, "max_nm": band.max_nm, "on": "vacuum wavelengths"},
        "dispersion": _dispersion(material), "convention": _convention(material),
        "bulk_loss": _loss(material), "bulk_loss_note": None if getattr(material, "bulk_loss", None) is not None else NO_LOSS,
        "thermo_optic": _thermo(material), "crystal": getattr(material, "birefringence", None) is not None,
        "basis": material.basis,
    }


def _covers(material: Any, low: Decimal | None, high: Decimal | None) -> bool:
    band = material.valid_band_nm
    return (low is None or Decimal(band.min_nm) <= low) and (high is None or high <= Decimal(band.max_nm))


def _loss_covers(material: Any, low: Decimal | None, high: Decimal | None) -> bool:
    """The selection rule is the evaluator's loss rule at *each requested endpoint*, not an independently rounded band."""

    if getattr(material, "bulk_loss", None) is None:
        return False
    if low is None and high is None:
        return True
    from .optics.materials.physics import bulk_alpha_per_mm

    return all(bulk_alpha_per_mm(material, float(wavelength_nm))[0] is not None
               for wavelength_nm in (low, high) if wavelength_nm is not None)


def find_materials(materials: Iterable[Any], *, manufacturer: str | None = None, designation: str | None = None, loose: bool = False,
                   search: str | None = None, covers_min_nm: Decimal | None = None, covers_max_nm: Decimal | None = None,
                   with_bulk_loss: bool = False, availability: str | None = None) -> list[Any]:
    """The published materials matching every filter given (module docstring), in id order."""

    out = []
    wanted = designation.strip().upper() if designation else None
    wanted_loose = loose_name(designation) if designation else None
    needle = loose_name(search) if search else None
    for material in sorted(materials, key=lambda m: m.material_id):
        named = NamedMaterial.of(material)
        names = [material.material_id, *named.names()]
        if manufacturer and not named.made_by(manufacturer):
            continue
        if wanted is not None:
            if loose and wanted_loose not in {loose_name(n) for n in names}:
                continue
            if not loose and wanted not in {n.upper() for n in names}:
                continue
        if needle is not None and not any(needle in loose_name(n) for n in names):
            continue
        if not _covers(material, covers_min_nm, covers_max_nm):
            continue
        if with_bulk_loss and not _loss_covers(material, covers_min_nm, covers_max_nm):
            continue
        if availability is not None:
            identity = getattr(material, "identity", None)
            status = identity.ordering_status if identity is not None else None
            if (status.design_availability if status is not None else "NOT_STATED") != availability:
                continue
        out.append(material)
    return out


def resolve_name(materials: Iterable[Any], name: str, *, manufacturer: str | None = None, catalogue_order: Iterable[str] = (),
                 loose: bool = True) -> dict[str, Any]:
    """How ``name`` resolves among ``materials`` by the shared resolver (prescription import's rule; a loose spelling only when
    nothing matches exactly): ``{"resolved": id | None, "how", "candidates", "also_matches"}``."""

    index = MaterialNameIndex(list(materials))
    result = index.resolve(name, manufacturer=manufacturer, catalogue_order=catalogue_order)
    if result.how == "none" and loose:
        result = index.resolve(name, manufacturer=manufacturer, catalogue_order=catalogue_order, loose=True)
        if result.resolved is not None and result.how != "loose match":
            result = type(result)(result.material_ids, "loose match", result.candidates, result.also_matches)
    by_key = {key: named.material_id for key, named in index.materials.items()}
    return {"query": name, "manufacturer": manufacturer, "catalogue_order": list(catalogue_order) or None,
            "resolved": None if result.resolved is None else by_key[result.resolved], "how": result.how,
            "candidates": [by_key[m] for m in result.candidates], "also_matches": [by_key[m] for m in result.also_matches]}


# ------------------------------------------------------------------------------------------- IR selection


def _engine():
    """The Engine's material evaluation (``optics.materials.physics``), or None when this Engine has none."""

    try:
        from .optics.materials import physics
    except ImportError:
        return None
    if not hasattr(physics, "material_point") or not hasattr(physics, "internal_transmittance"):
        return None
    return physics


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _point(engine: Any, material: Any, nm: float, temperature_c: float) -> dict[str, Any]:
    point = engine.material_point(material, nm, temperature_c=temperature_c)
    out: dict[str, Any] = {"wavelength_nm": nm, "n": _round(point.n_abs, INDEX_DIGITS)}
    for name, key, digits in (("n_group", "n_group", INDEX_DIGITS), ("gdd_mm2_per_mm", "gdd_mm2_per_mm", GDD_DIGITS)):
        value = getattr(point, key, None)
        if value is not None:
            out[name] = _round(value, digits)
    for name in ("reason", "derivative_limitation", "temperature_note"):
        value = getattr(point, name, None)
        if value:
            out[name] = value
    convention = getattr(point, "convention", None)
    if convention is not None:
        out["convention"] = convention
    return out


def _transmittance(engine: Any, material: Any, nm: float, thickness_mm: float, temperature_c: float) -> dict[str, Any]:
    value, reason = engine.internal_transmittance(material, nm, thickness_mm, temperature_c=temperature_c)
    out: dict[str, Any] = {"wavelength_nm": nm, "thickness_mm": thickness_mm, "internal_transmittance": _round(value, TRANSMITTANCE_DIGITS)}
    if value is None:
        out["reason"] = reason or NO_LOSS
    return out


def ir_selection(material: Any, *, at_nm: float | None = None, band_nm: tuple[float, float] | None = None,
                 thickness_mm: float | None = None, temperature_c: float = DEFAULT_TEMPERATURE_C) -> dict[str, Any] | None:
    """The selection properties of one material (module docstring), ``None`` when nothing was asked."""

    if at_nm is None and band_nm is None:
        return None
    engine = _engine()
    if engine is None:
        return {"available": False, "reason": IR_UNAVAILABLE}
    out: dict[str, Any] = {"available": True, "temperature_c": temperature_c, "wavelengths": "vacuum"}
    if at_nm is not None:
        out["at"] = _point(engine, material, at_nm, temperature_c)
        if thickness_mm is not None:
            out["at"]["transmission"] = _transmittance(engine, material, at_nm, thickness_mm, temperature_c)
    if band_nm is not None:
        low, high = band_nm
        middle = (low + high) / 2.0
        points = [_point(engine, material, nm, temperature_c) for nm in (low, middle, high)]
        n_lo, n_mid, n_hi = (p["n"] for p in points)
        figure, reason = None, None
        if None in (n_lo, n_mid, n_hi):
            reason = next(p.get("reason") for p in points if p["n"] is None) or "the index is not established across the band"
        elif n_lo == n_hi:
            reason = "no dispersion across the band (n_lo == n_hi)"
        else:
            figure = (n_mid - 1.0) / (n_lo - n_hi)
        out["band"] = {"min_nm": low, "mid_nm": middle, "max_nm": high, "points": points,
                       "dispersion_figure": _round(figure, FIGURE_DIGITS), "dispersion_figure_formula": "(n_mid - 1) / (n_lo - n_hi)",
                       **({"reason": reason} if reason else {})}
        if thickness_mm is not None:
            out["band"]["transmission"] = [_transmittance(engine, material, nm, thickness_mm, temperature_c) for nm in (low, middle, high)]
    return out


# ------------------------------------------------------------------------- the queries (CLI and JSON API)


class MaterialQueryError(ValueError):
    """A material query whose options are inconsistent (the message says which)."""


def _positive(value: Any, label: str) -> Decimal:
    try:
        number = Decimal(str(value).strip())
        bounded = float(number)
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        number, bounded = None, float("nan")
    if isinstance(value, bool) or number is None or not math.isfinite(bounded) or bounded <= 0:
        raise MaterialQueryError(f"{label} must be a finite positive number representable by the Engine, got {value!r}")
    return number


def _temperature(value: Any) -> float:
    if value is None:
        return DEFAULT_TEMPERATURE_C
    try:
        number = float(Decimal(str(value).strip()))
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        number = float("nan")
    if isinstance(value, bool) or not math.isfinite(number) or number <= ABSOLUTE_ZERO_C:
        raise MaterialQueryError(f"temperature_c must be finite and above absolute zero, got {value!r}")
    return number


def _ir_options(at_nm: Any, band_nm: Any, thickness_mm: Any, temperature_c: Any) -> dict[str, Any]:
    at = float(_positive(at_nm, "at_nm")) if at_nm is not None else None
    band = None
    if band_nm is not None:
        if not isinstance(band_nm, (list, tuple)) or len(band_nm) != 2:
            raise MaterialQueryError("band_nm is [lo, hi] in nm")
        band = (float(_positive(band_nm[0], "band_nm lo")), float(_positive(band_nm[1], "band_nm hi")))
        if not band[0] < band[1]:
            raise MaterialQueryError("band_nm is [lo, hi] with lo < hi")
    thickness = float(_positive(thickness_mm, "thickness_mm")) if thickness_mm is not None else None
    if thickness is not None and at is None and band is None:
        raise MaterialQueryError("thickness_mm needs at_nm or band_nm (the wavelength of the transmittance)")
    temperature = _temperature(temperature_c)
    return {"at_nm": at, "band_nm": band, "thickness_mm": thickness, "temperature_c": temperature}


AVAILABILITIES = ("CURRENT", "LEGACY", "NOT_STATED")


def list_materials(catalog: Any, *, manufacturer: str | None = None, designation: str | None = None, loose: bool = False,
                   search: str | None = None, covers_nm: Any = None, wavelength_nm: Any = None, with_bulk_loss: bool = False,
                   availability: str | None = None, at_nm: Any = None, band_nm: Any = None, thickness_mm: Any = None,
                   temperature_c: Any = None, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    """``catalog materials`` (module docstring): ``{catalog_version, schema_version, total, offset, limit, filters,
    materials: [summary (+ ir)]}``; ``MaterialQueryError`` for inconsistent options."""

    options = _ir_options(at_nm, band_nm, thickness_mm, temperature_c)
    if covers_nm is not None and wavelength_nm is not None:
        raise MaterialQueryError("give covers_nm or wavelength_nm, not both")
    if loose and not designation:
        raise MaterialQueryError("loose applies to designation")
    if availability is not None and availability not in AVAILABILITIES:
        raise MaterialQueryError(f"availability is one of {', '.join(AVAILABILITIES)}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or (
            limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0)):
        raise MaterialQueryError("offset and limit must be non-negative integers")
    low = high = None
    if covers_nm is not None:
        if not isinstance(covers_nm, (list, tuple)) or len(covers_nm) != 2:
            raise MaterialQueryError("covers_nm is [min, max] in nm")
        low, high = _positive(covers_nm[0], "covers_nm min"), _positive(covers_nm[1], "covers_nm max")
        if low > high:
            raise MaterialQueryError("covers_nm is [min, max] with min <= max")
    elif wavelength_nm is not None:
        low = high = _positive(wavelength_nm, "wavelength_nm")
    found = find_materials(catalog.materials, manufacturer=manufacturer, designation=designation, loose=loose, search=search,
                           covers_min_nm=low, covers_max_nm=high, with_bulk_loss=with_bulk_loss, availability=availability)
    page = found[offset:] if limit is None else found[offset:offset + limit]
    rows = []
    for material in page:
        row = material_summary(material)
        ir = ir_selection(material, **options)
        if ir is not None:
            row["ir"] = ir
        rows.append(row)
    return {"catalog_version": catalog.catalog_version, "schema_version": catalog.schema_version, "total": len(found),
            "offset": offset, "limit": limit,
            "filters": {"manufacturer": manufacturer, "designation": designation, "loose": loose, "search": search,
                        "covers_nm": None if low is None else [_text(low), _text(high)], "with_bulk_loss": with_bulk_loss,
                        "availability": availability},
            "materials": rows}


def lookup_material(catalog: Any, name: str, *, manufacturer: str | None = None, glass_catalogs: Any = None, at_nm: Any = None,
                    band_nm: Any = None, thickness_mm: Any = None, temperature_c: Any = None) -> dict[str, Any]:
    """``catalog material NAME`` (module docstring): ``{catalog_version, resolution, material, ir}`` when the name resolves,
    else ``{catalog_version, resolution, candidates, next_step}`` (``resolution.resolved`` is null). ``name`` may be
    ``MAKER:DESIGNATION`` or ``ID:<material id>``; ``glass_catalogs``: a catalogue order (a list, or text separated by
    spaces / commas)."""

    if not isinstance(name, str) or not name.strip():
        raise MaterialQueryError("name is a material name, e.g. S-FPL53, CDGM:F5 or ID:F2")
    options = _ir_options(at_nm, band_nm, thickness_mm, temperature_c)
    name, explicit = name.strip(), False
    prefix, colon, rest = name.partition(":")
    if colon and prefix.strip() and rest.strip():
        if prefix.strip().upper() == "ID":
            name, explicit = rest.strip(), True
        else:
            if manufacturer and manufacturer.strip().upper() != prefix.strip().upper():
                raise MaterialQueryError("the name's MAKER: and the manufacturer option differ")
            name, manufacturer = rest.strip(), prefix.strip()
    if isinstance(glass_catalogs, str):
        order = [t for t in glass_catalogs.replace(",", " ").replace(";", " ").split() if t]
    elif glass_catalogs is None:
        order = []
    elif isinstance(glass_catalogs, (list, tuple)) and all(isinstance(t, str) for t in glass_catalogs):
        order = [t.strip() for t in glass_catalogs if t.strip()]
    else:
        raise MaterialQueryError("glass_catalogs is a list of maker codes or catalogue names, first preferred")
    if explicit:
        published = catalog.material(name.upper())
        resolution: dict[str, Any] = {"query": f"ID:{name}", "manufacturer": None, "catalogue_order": None,
                                      "resolved": None if published is None else published.material_id,
                                      "how": "material_id (explicit)" if published is not None else "none", "candidates": [],
                                      "also_matches": []}
    else:
        resolution = resolve_name(catalog.materials, name, manufacturer=manufacturer, catalogue_order=order)
    document: dict[str, Any] = {"catalog_version": catalog.catalog_version, "resolution": resolution}
    if resolution["resolved"] is None:
        by_id = {m.material_id: m for m in catalog.materials}
        document["candidates"] = [material_summary(by_id[m]) for m in resolution["candidates"]]
        document["next_step"] = ("an ambiguous name: give MAKER:DESIGNATION (or the manufacturer), the stable id (ID:<id>) or a "
                                 "catalogue order (glass_catalogs)" if resolution["candidates"] else
                                 "no published material answers to this name: search for close names (catalog materials --search)")
        return document
    material = catalog.material(resolution["resolved"])
    document["material"] = material_summary(material)
    ir = ir_selection(material, **options)
    if ir is not None:
        document["ir"] = ir
    return document


__all__ = ["AVAILABILITIES", "ENGINE_CONVENTION", "NO_LOSS", "MaterialQueryError", "find_materials", "ir_selection", "list_materials",
           "lookup_material", "loss_band_nm", "material_summary", "resolve_name"]
