"""A synthetic Optical Layout Package for the Optical Layout Drawing tests.

A folded 780 nm layout, datum = the customer's own light source ``SRC`` (a custom component) at the origin, emitting
along +X: lens ``L1`` (OpenSci, B01) at x 100, fold mirror ``M1`` (OpenSci, B02) at x 200 turning the beam to +Y, the
customer's own window ``W1`` (a catalog product they already own: modelled as an OpenSci window), lens ``L2`` (OpenSci,
B03) 5 mm higher than the rest (relative heights are kept) and the observation plane ``focal_plane``.
"""

from __future__ import annotations

import math
from typing import Any

from os_optics._vendor.shared_contracts.optical_layout import OpticalLayoutPackageV1, seal_layout

HASH_A, HASH_B = "a" * 64, "b" * 64
RESOLUTION_ID = "00000000-0000-4000-8000-0000000005a0"
ORDER_NUMBER = "OS-20261006-0042"
CUSTOMER = "北京光学实验室有限公司"


def _quaternion_to(axis: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """The rotation taking local +Z to ``axis`` (shortest arc)."""

    x, y, z = axis
    w = 1.0 + z
    if w < 1e-12:
        return (0.0, 1.0, 0.0, 0.0)
    q = (w, -y, x, 0.0)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(round(c / n, 6) for c in q)  # type: ignore[return-value]


def _rotate(q, v):  # noqa: ANN001, ANN202
    w, x, y, z = q
    cx, cy, cz = y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return (v[0] + 2.0 * (w * cx + dx), v[1] + 2.0 * (w * cy + dy), v[2] + 2.0 * (w * cz + dz))


def element(instance_id: str, role: str, position, axis, **extra: Any) -> dict[str, Any]:  # noqa: ANN001
    q = _quaternion_to(axis)
    up = tuple(round(c, 6) + 0.0 for c in _rotate(q, (0.0, 1.0, 0.0)))
    return {"instance_id": instance_id, "optical_role": role, "position_mm": list(position), "quaternion_wxyz": list(q),
            "axis_direction": list(axis), "up_direction": list(up), **extra}


def opensci(instance_id: str, role: str, position, axis, sku: str, name: str, name_zh: str, line: str, outline: dict,  # noqa: ANN001
            dofs=()) -> dict[str, Any]:  # noqa: ANN001
    return element(instance_id, role, position, axis, supplied_by="OPENSCI", opensci_sku=sku, display_name=name,
                   display_name_zh_cn=name_zh, line_key=line, outline=outline, retained_dofs=[{"axis": a, "control": c} for a, c in dofs])


def layout_document(*, ordered: bool = True) -> dict[str, Any]:
    s = 0.7071068
    elements = [
        element("SRC", "source", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), supplied_by="CUSTOMER", supply_reason="CUSTOMER_LIGHT_SOURCE",
                custom_component_key="laser-780"),
        opensci("L1", "lens", (100.0, 0.0, 0.0), (1.0, 0.0, 0.0), "OS-OPT-LENS-PCX-D25P4-F100", "Plano-convex lens D25.4 mm f 100 mm",
                "平凸透镜 D25.4 mm f 100 mm", "B01", {"shape": "ROUND", "diameter_mm": 25.4, "z_min_mm": -3.0, "z_max_mm": 3.0},
                dofs=(("Tz", "MANUAL"),)),
        opensci("M1", "mirror", (200.0, 0.0, 0.0), (-s, s, 0.0), "OS-OPT-MIRROR-FLAT-D25-NIR", "Flat mirror D25 NIR",
                "平面反射镜 D25 近红外", "B02", {"shape": "ROUND", "diameter_mm": 25.4, "z_min_mm": -6.0, "z_max_mm": 0.0},
                dofs=(("Rx", "MANUAL"), ("Ry", "NUMERICAL"))),
        element("W1", "window", (200.0, 80.0, 0.0), (0.0, 1.0, 0.0), supplied_by="CUSTOMER", supply_reason="CUSTOMER_OWNED",
                opensci_sku="OS-OPT-WIN-PLANE-D25P4-T5", display_name="Plane window D25.4 mm t 5 mm", display_name_zh_cn="平面窗口片 D25.4 mm 厚 5 mm",
                outline={"shape": "ROUND", "diameter_mm": 25.4, "z_min_mm": 0.0, "z_max_mm": 5.0}),
        opensci("L2", "lens", (200.0, 150.0, 5.0), (0.0, 1.0, 0.0), "OS-OPT-LENS-PCX-D25P4-F50", "Plano-convex lens D25.4 mm f 50 mm",
                "平凸透镜 D25.4 mm f 50 mm", "B03", {"shape": "RECTANGULAR", "width_mm": 30.0, "height_mm": 20.0, "z_min_mm": -2.0, "z_max_mm": 2.0}),
    ]
    points = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0), (200.0, 80.0, 0.0), (200.0, 150.0, 5.0), (200.0, 200.0, 5.0)]
    lengths = [round(math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(3))), 3) for a, b in zip(points, points[1:])]
    binding = {"project_hash": HASH_A, "local_validation_hash": HASH_B, "public_catalog_version": "2026-10-06.1",
               "commercial_resolution_id": RESOLUTION_ID, "purchase_scope": "OPTICS_ONLY"}
    if ordered:
        binding.update(order_number=ORDER_NUMBER, customer_display_name=CUSTOMER)
    return {
        "schema": "opensci.optical_layout.package", "schema_version": "1.0", "binding": binding,
        "frame": "DESIGN_FRAME_RELATIVE_TO_DATUM", "datum": "SRC", "elements": elements,
        "beam_paths": [{"path_key": "beam-1", "wavelength_nm": 780.0, "through": ["SRC", "L1", "M1", "W1", "L2", "focal_plane"],
                        "points_mm": [list(p) for p in points], "segment_lengths_mm": lengths}],
        "observations": [{"observation_id": "focal_plane", "position_mm": [200.0, 200.0, 5.0], "axis_direction": [0.0, 1.0, 0.0]}],
    }


def layout_package(*, ordered: bool = True) -> OpticalLayoutPackageV1:
    return seal_layout(layout_document(ordered=ordered))
