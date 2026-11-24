"""Project-declared optical models for the tests: catalog products turned into declared copies, and hand-written models
(an air-spaced objective, aperture stops, ideal thin lenses, compound units) as plain JSON documents."""

from __future__ import annotations

import copy
import json

from os_optics.local import declared_copy, load_public_catalog

NBK7_SELLMEIER = {"kind": "SELLMEIER", "b": ["1.03961212", "0.231792344", "1.01046945"], "c_um2": ["0.00600069867", "0.0200179144", "103.560653"]}
#: A flint of constant index (the independent references evaluate it exactly).
FLINT_INDEX = "1.7174"
BAND = {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}
SURFACE_APPROXIMATIONS = ["IDEAL_SURFACE_FIGURE", "NOMINAL_PRESCRIPTION"]


def nbk7() -> dict:
    # Test-only ideal, explicitly lossless glass: do not infer zero absorption from a real catalog source.
    return {"material_id": "N-BK7", "display_name": "N-BK7", "dispersion": copy.deepcopy(NBK7_SELLMEIER), "valid_band_nm": dict(BAND),
            "basis": "Schott N-BK7 Sellmeier formula; ideal zero bulk loss assumed for this test model",
            "absorption_coefficient_per_mm": "0"}


def flint() -> dict:
    # This synthetic test flint also states the assumed lossless model explicitly.
    return {"material_id": "FLINT", "display_name": "Flint (constant index)", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": FLINT_INDEX},
            "valid_band_nm": dict(BAND), "absorption_coefficient_per_mm": "0"}


def surface(index: int, z: str, *, radius: str | None = None, clear: str = "9", traced: str = "TRANSMITTED", **extra) -> dict:
    s = {"surface_id": f"S{index}", "vertex_z_mm": z, "clear_radius_mm": clear, "traced": traced, "status": "SPECIFIED", **extra}
    if radius is not None:
        s["radius_mm"] = radius
    return s


def sequence(surfaces: list[dict], media: list[str], *, edge: str = "10", band: dict | None = None) -> dict:
    return {"model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": "SURFACE_PRESCRIPTION", "surfaces": surfaces, "media": media,
            "edge_radius_mm": edge, "valid_band_nm": band or dict(BAND), "approximations": list(SURFACE_APPROXIMATIONS), "status": "SPECIFIED"}


#: An air-spaced objective: a cemented N-BK7 / flint doublet, a 12 mm air gap, an N-BK7 plano-convex lens (mm).
OBJECTIVE_RADII = ("61.47", "-44.64", "-129.94", "52", None)
OBJECTIVE_VERTICES = ("0", "6", "8.5", "20.5", "25")
OBJECTIVE_MEDIA = ("N-BK7", "FLINT", "AIR", "N-BK7")


def objective_model(model_id: str = "objective") -> dict:
    surfaces = [surface(i + 1, z, radius=r) for i, (z, r) in enumerate(zip(OBJECTIVE_VERTICES, OBJECTIVE_RADII))]
    return {"model_id": model_id, "model": sequence(surfaces, list(OBJECTIVE_MEDIA))}


def doublet_model(model_id: str = "doublet") -> dict:
    surfaces = [surface(i + 1, z, radius=r) for i, (z, r) in enumerate(zip(OBJECTIVE_VERTICES[:3], OBJECTIVE_RADII[:3]))]
    return {"model_id": model_id, "model": sequence(surfaces, list(OBJECTIVE_MEDIA[:2]))}


def plano_convex_model(model_id: str = "pcx") -> dict:
    surfaces = [surface(1, "0", radius=OBJECTIVE_RADII[3]), surface(2, "4.5")]
    return {"model_id": model_id, "model": sequence(surfaces, ["N-BK7"])}


def stop_model(model_id: str = "stop", **fields) -> dict:
    model = {"model_kind": "APERTURE_STOP", "model_version": "1", "status": "SPECIFIED"}
    model.update(fields)
    return {"model_id": model_id, "model": model}


def thin_lens_model(model_id: str = "thin", focal_length_mm: str = "100", *, clear: str = "10", edge: str = "12.7") -> dict:
    return {"model_id": model_id, "model": {"model_kind": "IDEAL_THIN_LENS", "model_version": "1", "focal_length_mm": focal_length_mm,
                                            "clear_radius_mm": clear, "edge_radius_mm": edge, "valid_band_nm": dict(BAND), "status": "SPECIFIED"}}


def compound_model(model_id: str, members: list[tuple[str, str, tuple, tuple]], envelope: dict, ports: list[dict] | None = None) -> dict:
    """``members``: ``(member_id, model_id, position_mm, quaternion_wxyz)`` in the unit frame."""

    model = {"model_kind": "COMPOUND", "model_version": "1", "envelope": envelope,
             "members": [{"member_id": m, "model_id": ref, "pose": {"position_mm": list(p), "quaternion_wxyz": list(q)}}
                         for m, ref, p, q in members]}
    if ports is not None:
        model["ports"] = ports
    return {"model_id": model_id, "model": model}


def cylinder(diameter: str, z_min: str, z_max: str) -> dict:
    return {"kind": "CYLINDER", "diameter_mm": diameter, "z_min_mm": z_min, "z_max_mm": z_max, "status": "SPECIFIED"}


def declared_component(instance_id: str, role: str, model_id: str, position=(0.0, 0.0, 0.0), quaternion=(0.5, 0.5, 0.5, 0.5)) -> dict:
    return {"instance_id": instance_id, "optical_role": role, "custom_part": {"custom_part_key": f"own-{instance_id.lower()}", "description": "My own optic."},
            "optical_model_id": model_id, "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(quaternion)}}


def with_models(document: dict, models: list[dict], materials: list[dict] | None = None, coatings: list[dict] | None = None,
                **fields) -> dict:
    """``document`` with the declared models (and their materials / coatings; ``None`` or empty: absent) added."""

    out = copy.deepcopy(document)
    out["optical_models"] = models
    if materials:
        out["model_materials"] = materials
    if coatings:
        out["model_coatings"] = coatings
    out.update(fields)
    return out


def declared_project(document: dict, catalog_source, *, model_kind: str | None = None) -> dict:
    """``document`` with every catalog component replaced by a ``custom_part`` using a declared copy of its product's model
    (``declared_copy``): same poses, roles, observations and targets."""

    catalog = load_public_catalog(catalog_source)
    out = json.loads(json.dumps(document))
    models: dict[str, dict] = {}
    materials: dict[str, dict] = {}
    coatings: dict[str, dict] = {}
    for component in out["components"]:
        product_ref = component.pop("product", None)
        if product_ref is None:
            continue
        product = catalog.get(product_ref["optical_sku"])
        model_id = f"m{len(models) + 1}" if product.optical_sku not in models else None
        if product.optical_sku not in models:
            model, mats, coats = declared_copy(product, catalog, model_kind=model_kind, model_id=model_id)
            models[product.optical_sku] = model.model_dump(mode="json", exclude_none=True)
            materials.update({m.material_id: m.model_dump(mode="json", exclude_none=True) for m in mats})
            coatings.update({c.coating_id: c.model_dump(mode="json", exclude_none=True) for c in coats})
        component["custom_part"] = {"custom_part_key": f"own-{component['instance_id'].lower()}", "description": "Declared copy."}
        component["optical_model_id"] = models[product.optical_sku]["model_id"]
    return with_models(out, list(models.values()), list(materials.values()), list(coatings.values()))


def without_namespace(value) -> str:
    """JSON text of ``value`` with the Engine's material namespaces made equal (``catalog:X`` and ``project:X``)."""

    return json.dumps(value, sort_keys=True).replace('"project:', '"catalog:')
