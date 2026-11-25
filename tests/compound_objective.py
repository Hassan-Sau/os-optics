"""A representative compound objective from the seed catalog, found by query (no SKU is pinned): two cemented achromats
(edge radius 12.7 mm, focal lengths nearest 100 mm) and a plano-convex field lens (focal length nearest 50 mm) on one
axis along design +X, a collimated 633 nm source and an image plane. Also the lens placements of the optical-solid
probes: two copies of one 12 mm-thick achromat overlapping, touching or apart."""

from __future__ import annotations

import math
from decimal import Decimal

from os_optics._vendor.shared_contracts.common.geometry import PoseV1
from os_optics._vendor.shared_contracts.design_catalog import OpticalCategory, PublicOpticalCatalogV2, PublicOpticalProductV2
from os_optics._vendor.shared_contracts.optical_design import OpticalComponentInstanceV1, OpticalProductRefV1, OpticalRoleV1, ProjectDocumentV2
from os_optics._vendor.shared_contracts.optical_design.design_content import CircularApertureV1, ObservationV1, TargetV1

SQ = math.sqrt(0.5)
ALONG_X = (SQ, 0.0, SQ, 0.0)  # local +Z -> design +X
AGAINST_X = (SQ, 0.0, -SQ, 0.0)  # local +Z -> design -X
EDGE_RADIUS = Decimal("12.7")


def _surfaces(product: PublicOpticalProductV2):
    return getattr(product.simulation_models[0], "surfaces", ())


def _round_sequence(product: PublicOpticalProductV2) -> bool:
    model = product.simulation_models[0]
    return model.model_kind == "SURFACE_SEQUENCE" and model.edge_radius_mm is not None and Decimal(model.edge_radius_mm) == EDGE_RADIUS


def objective_parts(catalog: PublicOpticalCatalogV2) -> tuple[PublicOpticalProductV2, PublicOpticalProductV2, PublicOpticalProductV2]:
    lenses = sorted(catalog.select(category=OpticalCategory.LENS, wavelength_nm="633"), key=lambda p: p.optical_sku)
    doublets = [p for p in lenses if len(_surfaces(p)) == 3 and _round_sequence(p) and p.specifications.focal_length_mm is not None]
    doublets.sort(key=lambda p: (abs(Decimal(p.specifications.focal_length_mm.value) - 100), p.optical_sku))
    pcx = [p for p in lenses if len(_surfaces(p)) == 2 and _round_sequence(p) and p.form.value == "PLANO_CONVEX_LENS"
           and p.specifications.focal_length_mm is not None and Decimal(p.specifications.focal_length_mm.value) > 0]
    pcx.sort(key=lambda p: (abs(Decimal(p.specifications.focal_length_mm.value) - 50), p.optical_sku))
    return doublets[0], doublets[1], pcx[0]


def source(x_mm: float = -60.0, waist_mm: float = 4.0) -> OpticalComponentInstanceV1:
    return OpticalComponentInstanceV1.model_validate({
        "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "collimated-laser", "description": "collimated"},
        "nominal_pose": {"position_mm": (x_mm, 0.0, 0.0), "quaternion_wxyz": ALONG_X},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 633.0, "power_w": 0.001, "waist_radius_mm": waist_mm,
                         "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}},
    })


def lens(instance_id: str, product: PublicOpticalProductV2, position, q=ALONG_X) -> OpticalComponentInstanceV1:
    return OpticalComponentInstanceV1(instance_id=instance_id, optical_role=OpticalRoleV1.LENS, product=OpticalProductRefV1(optical_sku=product.optical_sku),
                                      nominal_pose=PoseV1(position_mm=tuple(position), quaternion_wxyz=tuple(q)))


def project_of(components, image_x_mm: float = 110.0) -> ProjectDocumentV2:
    return ProjectDocumentV2.model_validate(ProjectDocumentV2(
        schema="opensci.project", project_schema_version="2.0", components=list(components),
        observations=[ObservationV1(observation_id="image", pose=PoseV1(position_mm=(image_x_mm, 0.0, 0.0), quaternion_wxyz=ALONG_X),
                                    aperture=CircularApertureV1(radius_mm=10.0))],
        targets=[TargetV1(target_id="p", metric="power_W", observation_id="image", minimum=1e-9, unit="W")],
    ).payload())


def objective_project(catalog: PublicOpticalCatalogV2) -> ProjectDocumentV2:
    a, b, c = objective_parts(catalog)
    return project_of([source(), lens("L1", a, (0.0, 0.0, 0.0)), lens("L2", b, (40.0, 0.0, 0.0)), lens("L3", c, (75.0, 0.0, 0.0))])


def thick_achromat(catalog: PublicOpticalCatalogV2) -> PublicOpticalProductV2:
    """The thickest achromat of edge radius 12.7 mm (its centre thickness is the distance between its outer vertices)."""
    candidates = [p for p in catalog.select(category=OpticalCategory.LENS, wavelength_nm="633") if len(_surfaces(p)) == 3 and _round_sequence(p)]
    return max(candidates, key=lambda p: (Decimal(_surfaces(p)[-1].vertex_z_mm), p.optical_sku))


def pair_project(catalog: PublicOpticalCatalogV2, second_x_mm: float, second_q=ALONG_X, second_y_mm: float = 0.0) -> ProjectDocumentV2:
    product = thick_achromat(catalog)
    return project_of([source(), lens("L1", product, (0.0, 0.0, 0.0)), lens("L2", product, (second_x_mm, second_y_mm, 0.0), second_q)], 150.0)
