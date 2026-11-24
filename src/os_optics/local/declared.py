"""Declared optical models from the catalog, and a summary of a Project's declared models.

``declared_copy(product, catalog)`` copies a catalog product's simulation model into the Project's own vocabulary: a
``ProjectOpticalModelV1`` (the product's model, frame and design geometry, verbatim) with the ``ProjectModelMaterialV1``
(copied by ``catalog_copy.copy_catalog_material``, as the prescription import and the native library copy a catalog glass:
the same physics, ``basis`` naming the release) and ``PublicCoatingV1`` it uses. Inserted into a Project (``optical_models``, ``model_materials``, ``model_coatings``) and
used by a ``custom_part`` component (``optical_model_id``), it compiles to the same Engine surfaces as the product
itself -- only its Engine materials are named ``project:<MATERIAL_ID>`` instead of ``catalog:<MATERIAL_ID>`` -- so it
traces identically. It is a starting point to edit (radii, thicknesses, glasses, coatings), with no catalog identity: the
component is the customer's own optic.
"""

from __future__ import annotations

from typing import Any

from .._vendor.shared_contracts.design_catalog import JoinedBodiesModelV1, PublicCoatingV1, PublicOpticalCatalogV2, PublicOpticalProductV2
from .._vendor.shared_contracts.optical_design.optical_model import ProjectModelMaterialV1, ProjectOpticalModelV1
from ..catalog_copy import copy_catalog_material
from .compile import CompiledElement, select_model
from .errors import LocalRunErrorCode, fail

#: The longest model id (``ModelId``): a copy is named after its SKU, cut to this length.
MODEL_ID_MAX_LENGTH = 64


def default_model_id(optical_sku: str) -> str:
    """A model id for a copy of ``optical_sku``: the SKU itself, cut to ``MODEL_ID_MAX_LENGTH`` characters."""

    return optical_sku[:MODEL_ID_MAX_LENGTH].rstrip("-")


def declared_copy(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, *, model_kind: str | None = None,
                  model_id: str | None = None) -> tuple[ProjectOpticalModelV1, tuple[ProjectModelMaterialV1, ...], tuple[PublicCoatingV1, ...]]:
    """``(optical_model, model_materials, model_coatings)``: the product's simulation model (``select_model``) as a
    declared model ``model_id`` (default ``default_model_id``) with the materials and coatings it uses (module docstring).
    ``LocalRunError`` when the product has no single model of that kind or a reference is not published."""

    model = select_model(product, model_kind)
    if isinstance(model, JoinedBodiesModelV1):
        raise fail(LocalRunErrorCode.SIMULATION_MODEL_UNSUPPORTED,
                   f"{product.optical_sku} is a polarizing prism of joined bodies: a Project declares it as a COMPOUND unit of one "
                   "SURFACE_SET model per body with its optic axes and the joints; a single declared copy of it is not made here")
    materials = []
    for material_id in model.material_ids() if hasattr(model, "material_ids") else ():
        material = catalog.material(material_id)
        if material is None:
            raise fail(LocalRunErrorCode.CATALOG_INVALID, f"{product.optical_sku}: material {material_id} is not published")
        materials.append(copy_catalog_material(material, catalog.catalog_version))
    coatings = []
    for coating_id in model.coating_ids() if hasattr(model, "coating_ids") else ():
        coating = catalog.coating(coating_id)
        if coating is None:
            raise fail(LocalRunErrorCode.CATALOG_INVALID, f"{product.optical_sku}: coating {coating_id} is not published")
        coatings.append(coating)
    declared = ProjectOpticalModelV1(model_id=model_id or default_model_id(product.optical_sku), display_name=product.display_name, model=model,
                                     frame=product.frame, design_geometry=product.design_geometry)
    return declared, tuple(materials), tuple(coatings)


def declared_model_summary(elements: tuple[CompiledElement, ...]) -> list[dict[str, Any]]:
    """Per component with a declared model, in Project order: ``instance_id``, ``optical_model_id``, ``model_kind``, the
    number of Engine surfaces it compiles to and its Engine materials (and, for a ``COMPOUND`` unit, its members)."""

    rows = []
    for element in elements:
        if element.optical_model_id is None:
            continue
        row: dict[str, Any] = {"instance_id": element.instance_id, "optical_model_id": element.optical_model_id, "model_kind": element.stated_kind,
                               "engine_surfaces": [s.surface_id for s in element.surfaces],
                               "engine_materials": [m.material_id for m in element.materials]}
        if element.is_compound:
            row["members"] = [{"member_id": m.member_id, "optical_model_id": m.optical_model_id, "model_kind": m.model_kind} for m in element.members]
        rows.append(row)
    return rows
