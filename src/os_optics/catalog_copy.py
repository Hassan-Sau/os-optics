"""Copying a public catalog material into a Project, and telling whether two Project materials are the same physics.

Every route that puts a catalog glass into a Project -- prescription import, the native component library and a declared
copy of a catalog product (``catalog_model`` / ``local.declared.declared_copy``) -- copies it with
``copy_catalog_material``: the complete optical definition verbatim -- the dispersion law (every kind the catalog
publishes), the valid band, a uniaxial crystal's extraordinary index (``birefringence``), the law's wavelength / index
convention (``index_reference``), the measured spectral bulk loss (``bulk_loss``) and the thermo-optic data
(``thermo_optic``) -- the display name, ``basis`` naming the catalog release and, for a material that states its identity
(catalog schema 3.6), ``copied_from`` (``CatalogCopyV1``: the catalog release, the chosen material id and the edition of the
source its optical data come from). The copy is complete: a saved Project reopens without the catalog and without a second
name lookup. (A material without identity -- an earlier catalog -- keeps its earlier copy, without ``copied_from``: the
``basis`` names its release.)

Two Project materials with one ``material_id`` are the same material when their physics is the same
(``physics_difference`` is ``None``): the dispersion (kind and values, compared as numbers), a crystal's extraordinary index
(an isotropic material has none), the valid band, the bulk absorption (absent counts as 0), the index convention (the law's
wavelength and index convention, its reference air and source band; absent: vacuum wavelength, absolute index), the
spectral bulk loss (kind and rows) and the thermo-optic data. The display name, ``basis``, ``copied_from`` and the
provenance texts inside the physics (a convention's ``stated_by`` / ``basis``, a loss table's ``basis``, a measured
table's ``measurement`` and a table's ``max_interpolation_error``) are provenance, not physics, so the same glass copied
by different routes or from different catalog releases is shared; two copies whose physics differs are refused
(``material_conflict``).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from ._vendor.shared_contracts.optical_design.optical_model import ProjectModelMaterialV1

BASIS_MAX = 1000  # the contract's limit on a model material's ``basis``
#: Keys of the optional physics that are provenance texts, not physics (``physics_difference``).
PROVENANCE_KEYS = frozenset({"basis", "stated_by", "measurement", "max_interpolation_error", "origin"})


def copied_material_basis(published_basis: str, catalog_version: str) -> str:
    """``"<published basis>; copied from public optical catalog <version>"``; the published basis alone when the
    combined text would exceed the contract's limit."""

    basis = f"{published_basis.rstrip(' .')}; copied from public optical catalog {catalog_version}"
    return basis if len(basis) <= BASIS_MAX else published_basis


def _dump(value: Any) -> Any:
    return None if value is None else value.model_dump(mode="json", by_alias=True, exclude_none=True)


def copy_catalog_material(published: Any, catalog_version: str) -> ProjectModelMaterialV1:
    """The Project copy of a published catalog material (module docstring); ``ValidationError`` when the contract refuses
    it."""

    extra: dict[str, Any] = {}
    birefringence = getattr(published, "birefringence", None)
    if birefringence is not None:
        extra["birefringence"] = {"kind": "UNIAXIAL", "extraordinary": _dump(birefringence.extraordinary)}
    for name in ("index_reference", "bulk_loss", "thermo_optic"):
        value = getattr(published, name, None)
        if value is not None:
            extra[name] = _dump(value)
    identity = getattr(published, "identity", None)
    if identity is not None:
        extra["copied_from"] = {"catalog_version": catalog_version, "material_id": published.material_id,
                                "source_edition": identity.source_edition}
    return ProjectModelMaterialV1(
        material_id=published.material_id, display_name=published.display_name,
        dispersion=_dump(published.dispersion), valid_band_nm=_dump(published.valid_band_nm),
        basis=copied_material_basis(published.basis, catalog_version), **extra,
    )


def _material(value: ProjectModelMaterialV1 | Mapping[str, Any]) -> ProjectModelMaterialV1:
    return value if isinstance(value, ProjectModelMaterialV1) else ProjectModelMaterialV1.model_validate(dict(value))


def _numbers(values) -> tuple[Decimal, ...]:
    return tuple(Decimal(v) for v in values)


def _dispersion(material: ProjectModelMaterialV1) -> tuple[Any, ...]:
    return _law(material.dispersion)


def _extraordinary(material: ProjectModelMaterialV1) -> tuple[Any, ...] | None:
    return None if material.birefringence is None else _law(material.birefringence.extraordinary)


def _canonical(value: Any) -> Any:
    """A JSON value with every number (and decimal string) as a ``Decimal`` and the provenance keys dropped, so physics
    compares as numbers (``"1.50"`` == ``"1.5"``)."""

    if isinstance(value, Mapping):
        return tuple(sorted((key, _canonical(item)) for key, item in value.items() if key not in PROVENANCE_KEYS))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            number = Decimal(value)
        except InvalidOperation:
            return value
        return number if number.is_finite() else value
    return value


def _law(dispersion) -> tuple[Any, ...]:
    if dispersion.kind == "CONSTANT_INDEX":
        return ("CONSTANT_INDEX", Decimal(dispersion.refractive_index))
    if dispersion.kind == "SELLMEIER":
        return ("SELLMEIER", _numbers(dispersion.b), _numbers(dispersion.c_um2))
    if dispersion.kind == "TABULATED":
        return ("TABULATED", _numbers(dispersion.wavelength_nm), _numbers(dispersion.refractive_index), dispersion.interpolation)
    return (dispersion.kind, _canonical(_dump(dispersion)))  # POWER_SERIES, HERZBERGER, TEMPERATURE_SELLMEIER


def _band(material: ProjectModelMaterialV1) -> tuple[Decimal, Decimal]:
    return Decimal(material.valid_band_nm.min_nm), Decimal(material.valid_band_nm.max_nm)


def _absorption(material: ProjectModelMaterialV1) -> Decimal:
    value = material.absorption_coefficient_per_mm
    return Decimal(0) if value is None else Decimal(value)


def _optional(material: ProjectModelMaterialV1, name: str) -> Any:
    return _canonical(_dump(getattr(material, name)))


def physics_difference(a: ProjectModelMaterialV1 | Mapping[str, Any], b: ProjectModelMaterialV1 | Mapping[str, Any]) -> str | None:
    """What differs physically between two materials (``None`` when they are the same physics; module docstring)."""

    a, b = _material(a), _material(b)
    differences = []
    da, db = _dispersion(a), _dispersion(b)
    if da != db:
        differences.append(f"the dispersion ({da[0]} vs {db[0]})" if da[0] != db[0] else f"the {da[0]} dispersion data")
    if _extraordinary(a) != _extraordinary(b):
        differences.append("the extraordinary index (a uniaxial crystal's)" if None not in (_extraordinary(a), _extraordinary(b))
                           else "whether it is a uniaxial crystal")
    if _band(a) != _band(b):
        differences.append("the valid band ({}-{} nm vs {}-{} nm)".format(*(f"{v.normalize():f}" for v in (*_band(a), *_band(b)))))
    if _absorption(a) != _absorption(b):
        differences.append(f"the bulk absorption ({_absorption(a).normalize():f} vs {_absorption(b).normalize():f} per mm)")
    for name, what in (("index_reference", "the index convention (index_reference)"), ("bulk_loss", "the spectral bulk loss"),
                       ("thermo_optic", "the thermo-optic data")):
        ours, theirs = _optional(a, name), _optional(b, name)
        if ours != theirs:
            stated = (ours is not None, theirs is not None)
            differences.append(what if all(stated) else f"{what} (stated vs not stated)" if stated[0] else f"{what} (not stated vs stated)")
    return ", ".join(differences) or None


def material_conflict(present: ProjectModelMaterialV1 | Mapping[str, Any], added: ProjectModelMaterialV1 | Mapping[str, Any]) -> str | None:
    """``None`` when ``added`` may share the Project's ``present`` material of the same id (the same physics); else the
    customer-facing reason it cannot."""

    difference = physics_difference(present, added)
    if difference is None:
        return None
    material_id = _material(added).material_id
    return (f"the Project already declares material_id {material_id!r} with different physics: {difference} differ "
            f"(Project: {_material(present).basis or 'no basis stated'}; added: {_material(added).basis or 'no basis stated'}); "
            "give one of them another material_id")


__all__ = ["BASIS_MAX", "copied_material_basis", "copy_catalog_material", "material_conflict", "physics_difference"]
