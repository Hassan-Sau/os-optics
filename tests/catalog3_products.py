"""A synthetic catalog of the schema-3.0 constructs for the Engine tests: a cube non-polarizing and polarizing splitter,
right-angle, penta, dove and wedge prisms, a wedged beam sampler (bare Fresnel front face), a rectangular window, an
off-axis parabolic mirror, a cylindrical lens and a bandpass filter with a typical spectrum.

Every solid is built from its corners in a convenient frame and then moved so that its centroid (computed by the
public contract itself) is the frame origin, as ``BODY_CENTER`` requires. All numbers are exact decimal strings; the
irrational ones (sines, cosines) are 17-digit decimals, well inside the contract's 1e-9 unit-vector allowance.
"""

from __future__ import annotations

import copy
import math
from decimal import Decimal

from os_optics._vendor.shared_contracts.common.decimal_string import decimal_to_string
from os_optics._vendor.shared_contracts.design_catalog import ConvexBodyV1, PublicOpticalCatalogV2

CATALOG_VERSION = "2026-10-06.1"
NBK7 = "N-BK7"
NBK7_SELLMEIER = {"kind": "SELLMEIER", "b": ["1.03961212", "0.231792344", "1.01046945"], "c_um2": ["0.00600069867", "0.0200179144", "103.560653"]}
AR = "AR_400-1100R0.005T0.995_AOI0-10"
NPBS = "BS_CUBE_400-1100R0.5T0.5_AOI45"
PBS = "PBS_600-700_AOI45"
HR_IN_GLASS = "HR_400-1100R0.99_AOI15-30_IN_GLASS"
GOLD = "HR_GOLD_700-2000R0.96_AOI0-50"
SPECTRUM = "BP_550_TYPICAL"

CUBE = "OS-OPT-TEST-BS-CUBE-A12P7"
PBS_CUBE = "OS-OPT-TEST-PBS-CUBE-A12P7"
RIGHT_ANGLE = "OS-OPT-TEST-PRISM-RA-L9"
PENTA = "OS-OPT-TEST-PRISM-PENTA-A10"
DOVE = "OS-OPT-TEST-PRISM-DOVE-L40"
WEDGE = "OS-OPT-TEST-PRISM-WEDGE-W2"
SAMPLER = "OS-OPT-TEST-BS-SAMPLER-WEDGE"
RECT_WINDOW = "OS-OPT-TEST-WIN-RECT-20X12"
OAP = "OS-OPT-TEST-MIR-OAP-90-FP50P8"
CYLINDER = "OS-OPT-TEST-LENS-CYL-F100"
BANDPASS = "OS-OPT-TEST-FLT-BP-550"

PENTA_HALF_ANGLE_DEG = 22.5
WEDGE_ANGLE_DEG = 2.0
SAMPLER_WEDGE_DEG = 0.5
CYLINDER_RADIUS_MM = 51.5
PARENT_FOCAL_MM = 50.8
S = 1.0 / math.sqrt(2.0)


def d(value) -> str:
    """Canonical decimal string of a number (floats through their shortest repr)."""

    return decimal_to_string(Decimal(repr(value)) if isinstance(value, float) else Decimal(value))


def vec(*values) -> list[str]:
    return [d(v) for v in values]


def _v(value, status: str = "SPECIFIED") -> dict:
    return {"value": d(value), "status": status}


def _band(lo, hi, status: str = "SPECIFIED") -> dict:
    return {"min_nm": d(lo), "max_nm": d(hi), "status": status}


def _posed(i: int, origin, normal, u, *, front="AIR", back=NBK7, coating: str | None = None, traced="TRANSMITTED", **profile) -> dict:
    out = {"surface_id": f"S{i}", "origin_mm": vec(*origin), "normal": vec(*normal), "u_axis": vec(*u), "medium_front": front,
           "medium_back": back, "traced": traced, "status": "SPECIFIED", **profile}
    if coating is not None:
        out["coating_id"] = coating
    return out


def _faces(*pairs) -> list[dict]:
    return [{"point_mm": vec(*p), "outward_normal": vec(*n)} for p, n in pairs]


def _shift(point, by) -> tuple:
    return tuple(Decimal(d(c)) - b for c, b in zip(point, by))


def _centred(faces: list[dict], surfaces: list[dict], ports: list[dict], side_radius: str | None = None):
    """Move faces, surfaces and ports so the body's centroid is the origin; returns (body dict, surfaces, ports, body)."""

    body = {"medium": NBK7, "faces": faces}
    if side_radius is not None:
        body["side_radius_mm"] = side_radius
    _, centroid = ConvexBodyV1.model_validate(body).volume_and_centroid()
    if side_radius is not None:  # a round body stays centred on local Z: shift along Z only
        centroid = (Decimal(0), Decimal(0), centroid[2])
    centroid = tuple(Decimal(d(float(c))) for c in centroid)  # a 17-digit decimal of the centroid (within the 1e-6 allowance)
    for face in faces:
        face["point_mm"] = [decimal_to_string(c) for c in _shift(face["point_mm"], centroid)]
    for surface in surfaces:
        surface["origin_mm"] = [decimal_to_string(c) for c in _shift(surface["origin_mm"], centroid)]
    for port in ports:
        port["point_mm"] = [decimal_to_string(c) for c in _shift(port["point_mm"], centroid)]
    model = ConvexBodyV1.model_validate(body)
    return body, surfaces, ports, model


def _port(port_id: str, point, direction, status: str = "SPECIFIED") -> dict:
    return {"port_id": port_id, "role": "INPUT" if port_id == "IN" else "OUTPUT", "point_mm": vec(*point), "direction": vec(*direction),
            "status": status}


def _ports_frame(ports: list[dict], origin: str = "BODY_CENTER") -> dict:
    return {"origin": origin, "nominal_propagation": "PORTS", "reference_axis": "PORTS", "symmetry": "NONE", "ports": ports}


def _box_geometry(body: ConvexBodyV1) -> tuple[dict, list[str]]:
    vertices = body.vertices_mm()
    low = [min(v[k] for v in vertices) for k in range(3)]
    high = [max(v[k] for v in vertices) for k in range(3)]
    halves = [Decimal(decimal_to_string((b - a) / 2)) for a, b in zip(low, high)]
    geometry = {"kind": "BOX", "center_mm": [decimal_to_string((a + b) / 2) for a, b in zip(low, high)],
                "half_extents_mm": [decimal_to_string(h) for h in halves], "status": "NOMINAL"}
    return geometry, [decimal_to_string(2 * h) for h in halves]


def _product(sku: str, category: str, form: str, name: str, zh: str, frame: dict, specs: dict, model: dict, geometry: dict) -> dict:
    return {"optical_sku": sku, "category": category, "form": form, "display_name": name, "display_name_zh_cn": zh,
            "summary": f"Synthetic {name.lower()} for the Engine tests.", "commercial_readiness": "SIMULATION_ONLY", "frame": frame,
            "specifications": specs, "simulation_models": [model], "design_geometry": geometry}


def _set_model(surfaces, body, approximations, band=("400", "1100"), fidelity="SURFACE_PRESCRIPTION") -> dict:
    model = {"model_kind": "SURFACE_SET", "model_version": "1", "fidelity": fidelity, "surfaces": surfaces,
             "valid_band_nm": _band(*band), "approximations": sorted(approximations), "status": "SPECIFIED"}
    if body is not None:
        model["body"] = body
    return model


# --------------------------------------------------------------------------------------------- products


def cube(polarizing: bool = False) -> dict:
    a = Decimal("6.35")
    faces = _faces(((a, 0, 0), (1, 0, 0)), ((-a, 0, 0), (-1, 0, 0)), ((0, a, 0), (0, 1, 0)), ((0, -a, 0), (0, -1, 0)),
                   ((0, 0, a), (0, 0, 1)), ((0, 0, -a), (0, 0, -1)))
    diagonal = PBS if polarizing else NPBS
    surfaces = [
        _posed(1, (0, 0, -a), (0, 0, -1), (1, 0, 0), coating=AR, clear_radius_mm="5.08"),
        _posed(2, (0, 0, 0), (S, 0, -S), (0, 1, 0), front=NBK7, coating=diagonal, traced="TRANSMITTED_AND_REFLECTED",
               clear_half_widths_mm=vec("6.35", 6.35 * math.sqrt(2.0))),
        _posed(3, (0, 0, a), (0, 0, 1), (1, 0, 0), coating=AR, clear_radius_mm="5.08"),
        _posed(4, (a, 0, 0), (1, 0, 0), (0, 1, 0), coating=AR, clear_radius_mm="5.08"),
        _posed(5, (-a, 0, 0), (-1, 0, 0), (0, 1, 0), coating=AR, clear_radius_mm="5.08"),  # the fourth optical face
    ]
    ports = [_port("IN", (0, 0, -a), (0, 0, 1)), _port("OUT_T", (0, 0, a), (0, 0, 1)), _port("OUT_R", (a, 0, 0), (1, 0, 0))]
    body, surfaces, ports, model = _centred(faces, surfaces, ports)
    geometry, dims = _box_geometry(model)
    band = ("600", "700") if polarizing else ("400", "1100")
    specs = {"outer_dimensions_mm": {"value": dims, "status": "SPECIFIED"}, "clear_aperture_diameter_mm": _v("10.16", "DESIGN_BASIS"),
             "thickness_mm": _v("12.7"), "coating": "BEAMSPLITTER", "materials": [NBK7], "working_band_nm": _band(*band),
             "reflectance": _v("0.5225" if polarizing else "0.5", "DESIGN_BASIS"),
             "transmittance": _v("0.47515" if polarizing else "0.5", "DESIGN_BASIS"), "design_aoi_deg": _v("45")}
    approximations = ["CEMENT_LAYER_NOT_MODELLED", "COATING_AT_SPECIFIED_BOUND", "UNTRACED_FACES_BLOCK", "CATALOG_MATERIAL_DISPERSION"]
    if polarizing:
        specs["extinction_ratio"] = _v("3166.6", "DESIGN_BASIS")
        approximations.append("LOSSLESS_COATING")
    form = "POLARIZING_CUBE_BEAMSPLITTER" if polarizing else "CUBE_BEAMSPLITTER"
    return _product(PBS_CUBE if polarizing else CUBE, "BEAMSPLITTER", form, "Polarizing cube beamsplitter" if polarizing else "Cube beamsplitter",
                    "偏振立方分束镜" if polarizing else "立方分束镜", _ports_frame(ports), specs,
                    _set_model(surfaces, body, approximations, band, "SPECIFICATION_ENVELOPE"), geometry)


def _prism_specs(dims, *, coating: str = "UNCOATED", **extra) -> dict:
    return {"outer_dimensions_mm": {"value": dims, "status": "SPECIFIED"},
            "clear_aperture_dimensions_mm": {"value": [d(Decimal(dims[0]) * Decimal("0.8")), d(Decimal(dims[1]) * Decimal("0.8"))],
                                             "status": "DESIGN_BASIS"},
            "thickness_mm": _v(dims[2]), "coating": coating, "materials": [NBK7], "working_band_nm": _band("400", "1100"), **extra}


def right_angle_prism() -> dict:
    """Legs of 9 mm in the XZ plane, 10 mm tall along Y: IN along +Z through the leg z = -3, total internal reflection on
    the hypotenuse, OUT along +X through the leg x = 3."""

    faces = _faces(((0, 0, -3), (0, 0, -1)), ((3, 0, 0), (1, 0, 0)), ((-6, 0, -3), (-S, 0, S)), ((0, 5, 0), (0, 1, 0)), ((0, -5, 0), (0, -1, 0)))
    surfaces = [
        _posed(1, (-1.5, 0, -3), (0, 0, -1), (1, 0, 0), clear_half_widths_mm=vec("4", "4.5")),
        _posed(2, (-1.5, 0, 1.5), (-S, 0, S), (0, 1, 0), clear_half_widths_mm=vec("4.5", "6")),
        _posed(3, (3, 0, 1.5), (1, 0, 0), (0, 1, 0), clear_half_widths_mm=vec("4.5", "4")),
    ]
    ports = [_port("IN", (-1.5, 0, -3), (0, 0, 1)), _port("OUT", (3, 0, 1.5), (1, 0, 0))]
    body, surfaces, ports, model = _centred(faces, surfaces, ports)
    geometry, dims = _box_geometry(model)
    return _product(RIGHT_ANGLE, "PRISM", "RIGHT_ANGLE_PRISM", "Right-angle prism", "直角棱镜", _ports_frame(ports),
                    _prism_specs(dims, deviation_angle_deg=_v("90")),
                    _set_model(surfaces, body, ["CATALOG_MATERIAL_DISPERSION", "NOMINAL_PRESCRIPTION", "UNTRACED_FACES_BLOCK"]), geometry)


def penta_prism() -> dict:
    """Entrance face z = 0 and exit face x = 0 (perpendicular, 10 mm), mirrors with outward normals (sin 22.5, 0, cos 22.5)
    through (0, 0, 10) and (-cos 22.5, 0, -sin 22.5) through (-5 - 10/sqrt(2), 0, 5): IN along +Z at x = -5, two
    reflections at 22.5 deg inside the glass, OUT along +X at z = 5; 10 mm tall along Y."""

    sin_a, cos_a = math.sin(math.radians(PENTA_HALF_ANGLE_DEG)), math.cos(math.radians(PENTA_HALF_ANGLE_DEG))
    m1_point = (-5.0, 0.0, (10.0 * cos_a + 5.0 * sin_a) / cos_a)
    m2_point = (-5.0 - 10.0 * S, 0.0, 5.0)
    faces = _faces(((0, 0, 0), (0, 0, -1)), ((0, 0, 0), (1, 0, 0)), ((0, 0, 10), (sin_a, 0, cos_a)), (m2_point, (-cos_a, 0, -sin_a)),
                   ((0, 5, 0), (0, 1, 0)), ((0, -5, 0), (0, -1, 0)))
    surfaces = [
        _posed(1, (-5, 0, 0), (0, 0, -1), (1, 0, 0), coating=AR, clear_half_widths_mm=vec("4", "4")),
        _posed(2, m1_point, (-sin_a, 0, -cos_a), (0, 1, 0), front=NBK7, back="AIR", coating=HR_IN_GLASS, traced="REFLECTED",
               clear_half_widths_mm=vec("4", "4")),
        _posed(3, m2_point, (cos_a, 0, sin_a), (0, 1, 0), front=NBK7, back="AIR", coating=HR_IN_GLASS, traced="REFLECTED",
               clear_half_widths_mm=vec("4", "4")),
        _posed(4, (0, 0, 5), (1, 0, 0), (0, 1, 0), coating=AR, clear_half_widths_mm=vec("4", "4")),
    ]
    ports = [_port("IN", (-5, 0, 0), (0, 0, 1)), _port("OUT", (0, 0, 5), (1, 0, 0))]
    body, surfaces, ports, model = _centred(faces, surfaces, ports)
    geometry, dims = _box_geometry(model)
    return _product(PENTA, "PRISM", "PENTA_PRISM", "Penta prism", "五角棱镜", _ports_frame(ports),
                    _prism_specs(dims, coating="DIELECTRIC_HIGH_REFLECTOR", deviation_angle_deg=_v("90")),
                    _set_model(surfaces, body, ["CATALOG_MATERIAL_DISPERSION", "COATING_AT_SPECIFIED_BOUND", "NOMINAL_PRESCRIPTION",
                                                "UNTRACED_FACES_BLOCK"]), geometry)


def dove_prism(out_x: float = 0.0) -> dict:
    """A 45 deg dove prism, base 40 mm along Z at x = -5, top 20 mm at x = 5, 10 mm tall along Y: IN along +Z through the
    entrance face (uncoated: bare Fresnel transmission at 45 deg), total internal reflection on the base, OUT along +Z
    through the exit face (``out_x``: where the nominal
    beam leaves it, DERIVED)."""

    faces = _faces(((-5, 0, 0), (-1, 0, 0)), ((5, 0, 0), (1, 0, 0)), ((-5, 0, -20), (S, 0, -S)), ((-5, 0, 20), (S, 0, S)),
                   ((0, 5, 0), (0, 1, 0)), ((0, -5, 0), (0, -1, 0)))
    surfaces = [
        _posed(1, (0, 0, -15), (S, 0, -S), (S, 0, S), clear_half_widths_mm=vec("6", "4")),
        _posed(2, (-5, 0, 0), (-1, 0, 0), (0, 0, 1), clear_half_widths_mm=vec("16", "4")),
        _posed(3, (0, 0, 15), (S, 0, S), (-S, 0, S), clear_half_widths_mm=vec("6", "4")),
    ]
    ports = [_port("IN", (0, 0, -15), (0, 0, 1)), _port("OUT", (out_x, 0, 15.0 - out_x), (0, 0, 1), "DERIVED")]
    body, surfaces, ports, model = _centred(faces, surfaces, ports)
    geometry, dims = _box_geometry(model)
    return _product(DOVE, "PRISM", "DOVE_PRISM", "Dove prism", "道威棱镜", _ports_frame(ports),
                    _prism_specs(dims, deviation_angle_deg=_v("0")),
                    _set_model(surfaces, body, ["CATALOG_MATERIAL_DISPERSION", "NOMINAL_PRESCRIPTION", "UNTRACED_FACES_BLOCK"]), geometry)


def wedge_prism(out_direction=(0.0, 0.0, 1.0)) -> dict:
    """A round wedge prism, D25.4: entrance face z = 0, exit face through (0, 0, 5) tilted by 2 deg about Y (outward normal
    (sin a, 0, cos a)); IN along +Z at the centre, OUT along ``out_direction`` (DERIVED)."""

    sin_a, cos_a = math.sin(math.radians(WEDGE_ANGLE_DEG)), math.cos(math.radians(WEDGE_ANGLE_DEG))
    faces = _faces(((0, 0, 5), (sin_a, 0, cos_a)), ((0, 0, 0), (0, 0, -1)))
    surfaces = [
        _posed(1, (0, 0, 0), (0, 0, -1), (1, 0, 0), coating=AR, clear_radius_mm="11.43"),
        _posed(2, (0, 0, 5), (sin_a, 0, cos_a), (cos_a, 0, -sin_a), coating=AR, clear_radius_mm="11.43"),
    ]
    ports = [_port("IN", (0, 0, 0), (0, 0, 1)), _port("OUT", (0, 0, 5), out_direction, "DERIVED")]
    body, surfaces, ports, model = _centred(faces, surfaces, ports, side_radius="12.7")
    low, high = model.z_range_mm()
    specs = {"outer_diameter_mm": _v("25.4"), "clear_aperture_diameter_mm": _v("22.86"), "thickness_mm": _v("5"), "coating": "ANTI_REFLECTION",
             "materials": [NBK7], "working_band_nm": _band("400", "1100"), "wedge_angle_deg": _v(WEDGE_ANGLE_DEG)}
    geometry = {"kind": "CYLINDER", "diameter_mm": "25.4", "z_min_mm": d(math.floor(float(low))), "z_max_mm": d(math.ceil(float(high))),
                "status": "NOMINAL"}
    return _product(WEDGE, "PRISM", "WEDGE_PRISM", "Wedge prism", "楔形棱镜", _ports_frame(ports), specs,
                    _set_model(surfaces, body, ["CATALOG_MATERIAL_DISPERSION", "COATING_AT_SPECIFIED_BOUND", "EDGE_AS_OPAQUE_RIM",
                                                "NOMINAL_PRESCRIPTION"]), geometry)


def wedge_sampler() -> dict:
    """A round wedged beam sampler (face-normal frame), D25.4, 3 mm: an uncoated front face at the origin (normal +Z) that
    transmits and reflects its bare Fresnel fraction, and an AR-coated back face tilted by 0.5 deg."""

    sin_w, cos_w = math.sin(math.radians(SAMPLER_WEDGE_DEG)), math.cos(math.radians(SAMPLER_WEDGE_DEG))
    surfaces = [
        _posed(1, (0, 0, 0), (0, 0, 1), (1, 0, 0), traced="TRANSMITTED_AND_REFLECTED", clear_radius_mm="11.43"),
        _posed(2, (0, 0, -3), (sin_w, 0, -cos_w), (cos_w, 0, sin_w), coating=AR, clear_radius_mm="11.43"),
    ]
    body = {"medium": NBK7, "side_radius_mm": "12.7", "faces": _faces(((0, 0, 0), (0, 0, 1)), ((0, 0, -3), (sin_w, 0, -cos_w)))}
    specs = {"outer_diameter_mm": _v("25.4"), "clear_aperture_diameter_mm": _v("22.86"), "thickness_mm": _v("3"), "coating": "ANTI_REFLECTION",
             "materials": [NBK7], "working_band_nm": _band("400", "1100"), "reflectance": _v("0.05", "DERIVED"),
             "transmittance": _v("0.95", "DERIVED"), "design_aoi_deg": _v("45"), "wedge_angle_deg": _v(SAMPLER_WEDGE_DEG)}
    frame = {"origin": "REFLECTING_FACE_CENTER", "nominal_propagation": "SPLIT_ABOUT_+Z", "reference_axis": "FACE_NORMAL", "symmetry": "NONE"}
    geometry = {"kind": "CYLINDER", "diameter_mm": "25.4", "z_min_mm": "-4", "z_max_mm": "0", "status": "NOMINAL"}
    return _product(SAMPLER, "BEAMSPLITTER", "BEAM_SAMPLER", "Beam sampler", "光束取样镜", frame, specs,
                    _set_model(surfaces, body, ["CATALOG_MATERIAL_DISPERSION", "COATING_AT_SPECIFIED_BOUND", "EDGE_AS_OPAQUE_RIM"]), geometry)


def _sequence_surface(i: int, z: str, coating: str | None = AR, traced: str = "TRANSMITTED", **profile) -> dict:
    out = {"surface_id": f"S{i}", "vertex_z_mm": z, "traced": traced, "status": "SPECIFIED", **profile}
    if coating is not None:
        out["coating_id"] = coating
    return out


def rect_window() -> dict:
    """A 20 x 12 x 3 mm N-BK7 window, clear aperture 16 x 8 mm."""

    model = {"model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": "SURFACE_PRESCRIPTION",
             "surfaces": [_sequence_surface(1, "0", clear_half_widths_mm=vec("8", "4")), _sequence_surface(2, "3", clear_half_widths_mm=vec("8", "4"))],
             "media": [NBK7], "edge_half_widths_mm": vec("10", "6"), "valid_band_nm": _band("400", "1100"),
             "approximations": ["COATING_AT_SPECIFIED_BOUND", "EDGE_AS_OPAQUE_RIM"], "status": "SPECIFIED"}
    specs = {"outer_dimensions_mm": {"value": vec("20", "12", "3"), "status": "SPECIFIED"},
             "clear_aperture_dimensions_mm": {"value": vec("16", "8"), "status": "SPECIFIED"}, "thickness_mm": _v("3"),
             "coating": "ANTI_REFLECTION", "materials": [NBK7], "working_band_nm": _band("400", "1100")}
    geometry = {"kind": "BOX", "center_mm": vec("0", "0", "1.5"), "half_extents_mm": vec("10", "6", "1.5"), "status": "NOMINAL"}
    return _product(RECT_WINDOW, "WINDOW", "PLANE_WINDOW", "Plane window", "平面窗口片", {"origin": "FIRST_SURFACE_VERTEX", "symmetry": "NONE"},
                    specs, model, geometry)


def oap() -> dict:
    """A 90 deg off-axis paraboloid, parent focal length 50.8 mm: the aperture centre (101.6 mm off the parent axis, where
    the sag is 50.8 mm) is the frame origin; collimated light along -Z focuses at (-101.6, 0, 0)."""

    fp = Decimal(d(PARENT_FOCAL_MM))
    surface = _posed(1, (-2 * fp, 0, -fp), (0, 0, 1), (1, 0, 0), front="AIR", back="AIR", coating=GOLD, traced="REFLECTED",
                     radius_mm=d(2 * fp), conic="-1", clear_radius_mm="11.43", aperture_center_mm=vec(2 * fp, 0))
    frame = _ports_frame([_port("IN", (0, 0, 0), (0, 0, -1)), _port("OUT", (0, 0, 0), (-1, 0, 0), "DERIVED")], "APERTURE_CENTER")
    specs = {"outer_diameter_mm": _v("25.4"), "clear_aperture_diameter_mm": _v("22.86"), "thickness_mm": _v("12"), "coating": "METALLIC_REFLECTOR",
             "working_band_nm": _band("700", "2000"), "reflectance": _v("0.96"), "focal_length_mm": _v(2 * fp),
             "parent_focal_length_mm": _v(fp), "off_axis_angle_deg": _v("90")}
    model = _set_model([surface], None, ["COATING_AT_SPECIFIED_BOUND", "SUBSTRATE_NOT_MODELLED", "TRANSMITTED_LIGHT_NOT_TRACED"],
                       ("700", "2000"), "SPECIFICATION_ENVELOPE")
    geometry = {"kind": "CYLINDER", "diameter_mm": "25.4", "z_min_mm": "-25", "z_max_mm": "14", "status": "NOMINAL"}
    return _product(OAP, "MIRROR", "OFF_AXIS_PARABOLIC_MIRROR", "Off-axis parabolic mirror", "离轴抛物面反射镜", frame, specs, model, geometry)


def cylindrical_lens() -> dict:
    """Plano-convex cylindrical lens, R 51.5 mm curved along local X (straight along Y), 4 mm, 20 x 12 mm."""

    model = {"model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": "SURFACE_PRESCRIPTION",
             "surfaces": [_sequence_surface(1, "0", radius_mm=d(CYLINDER_RADIUS_MM), cylinder_axis="Y", clear_half_widths_mm=vec("9", "5.5")),
                          _sequence_surface(2, "4", clear_half_widths_mm=vec("9", "5.5"))],
             "media": [NBK7], "edge_half_widths_mm": vec("10", "6"), "valid_band_nm": _band("400", "1100"),
             "approximations": ["NOMINAL_PRESCRIPTION"], "status": "SPECIFIED"}
    specs = {"outer_dimensions_mm": {"value": vec("20", "12", "4"), "status": "SPECIFIED"},
             "clear_aperture_dimensions_mm": {"value": vec("18", "11"), "status": "DESIGN_BASIS"}, "thickness_mm": _v("4"),
             "coating": "ANTI_REFLECTION", "materials": [NBK7], "working_band_nm": _band("400", "1100"),
             "design_wavelength_nm": _v("587.6"), "focal_length_mm": _v("100")}
    geometry = {"kind": "BOX", "center_mm": vec("0", "0", "2"), "half_extents_mm": vec("10", "6", "2"), "status": "NOMINAL"}
    return _product(CYLINDER, "LENS", "CYLINDRICAL_LENS", "Plano-convex cylindrical lens", "平凸柱面透镜",
                    {"origin": "FIRST_SURFACE_VERTEX", "symmetry": "NONE"}, specs, model, geometry)


def bandpass() -> dict:
    model = {"model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": "SPECIFICATION_ENVELOPE",
             "surfaces": [_sequence_surface(1, "0", coating=SPECTRUM, clear_radius_mm="11")], "edge_radius_mm": "12.7",
             "valid_band_nm": _band("400", "700"), "approximations": ["SUBSTRATE_NOT_MODELLED", "TYPICAL_SPECTRAL_CURVE"], "status": "NOMINAL"}
    specs = {"outer_diameter_mm": _v("25.4"), "clear_aperture_diameter_mm": _v("22"), "thickness_mm": _v("5"), "coating": "BANDPASS",
             "working_band_nm": _band("400", "700"), "center_wavelength_nm": _v("550")}
    geometry = {"kind": "CYLINDER", "diameter_mm": "25.4", "z_min_mm": "0", "z_max_mm": "5", "status": "NOMINAL"}
    return _product(BANDPASS, "FILTER", "BANDPASS_FILTER", "Bandpass filter", "带通滤光片",
                    {"origin": "FIRST_SURFACE_VERTEX", "symmetry": "ROTATIONAL_ABOUT_LOCAL_Z"}, specs, model, geometry)


# ----------------------------------------------------------------------------------------------- catalog


def _coating(cid: str, kind: str, band: dict, aoi: tuple[str, str], reference: str = "AIR_EQUIVALENT") -> dict:
    return {"coating_id": cid, "kind": kind, "display_name": cid, "bands": [band], "aoi_reference": reference,
            "valid_aoi_deg": {"min_deg": aoi[0], "max_deg": aoi[1], "status": "SPECIFIED"}, "basis": "Synthetic test coating."}


def coatings() -> dict[str, dict]:
    def band(lo, hi, r, t, **sp):
        return {"min_nm": lo, "max_nm": hi, "reflectance": r, "transmittance": t, "status": "DESIGN_BASIS", **sp}

    return {
        AR: _coating(AR, "ANTI_REFLECTION", band("400", "1100", "0.005", "0.995"), ("0", "10")),
        NPBS: _coating(NPBS, "BEAMSPLITTER", band("400", "1100", "0.5", "0.5"), ("45", "45"), "INCIDENT_MEDIUM"),
        PBS: _coating(PBS, "BEAMSPLITTER", band("600", "700", "0.5225", "0.47515", reflectance_s="0.995", reflectance_p="0.05",
                                                transmittance_s="0.0003", transmittance_p="0.95"), ("45", "45"), "INCIDENT_MEDIUM"),
        HR_IN_GLASS: _coating(HR_IN_GLASS, "DIELECTRIC_HIGH_REFLECTOR", band("400", "1100", "0.99", "0"), ("15", "30"), "INCIDENT_MEDIUM"),
        GOLD: _coating(GOLD, "METALLIC_REFLECTOR", band("700", "2000", "0.96", "0"), ("0", "50")),
        SPECTRUM: {"coating_id": SPECTRUM, "kind": "BANDPASS", "display_name": "Typical bandpass curve", "aoi_reference": "AIR_EQUIVALENT",
                   "spectrum": {"wavelength_nm": ["400", "500", "550", "600", "700"], "transmittance": ["0.001", "0.4", "0.855", "0.3", "0.001"],
                                "reflectance": ["0.999", "0.6", "0.145", "0.7", "0.999"], "interpolation": "LINEAR", "status": "TYPICAL"},
                   "valid_aoi_deg": {"min_deg": "0", "max_deg": "5", "status": "SPECIFIED"}, "basis": "Synthetic typical curve."},
    }


def all_products() -> list[dict]:
    return [cube(), cube(polarizing=True), right_angle_prism(), penta_prism(), dove_prism(), wedge_prism(), wedge_sampler(), rect_window(),
            oap(), cylindrical_lens(), bandpass()]


def catalog3_document(*products: dict) -> dict:
    """A schema-3.0 catalog document of ``products`` (default: every product above), validated by the public contract."""

    products = tuple(copy.deepcopy(p) for p in (products or all_products()))
    every = coatings()
    used = sorted({s["coating_id"] for p in products for m in p["simulation_models"] for s in m.get("surfaces", []) if "coating_id" in s})
    materials = sorted({m for p in products for m in p["specifications"].get("materials", [])})
    document = {
        "schema": "opensci.design_catalog.public_optical_catalog", "schema_version": "3.0", "catalog_version": CATALOG_VERSION,
        "mechanical_accessories": "CONFIGURED_PER_DESIGN", "categories_offered": sorted({p["category"] for p in products}),
        "materials": [{"material_id": m, "display_name": m, "dispersion": NBK7_SELLMEIER, "valid_band_nm": _band("300", "2500"),
                       "status": "SPECIFIED", "basis": "Glass catalogue Sellmeier formula."} for m in materials],
        "coatings": [every[c] for c in used],
        "products": list(products),
    }
    return PublicOpticalCatalogV2.model_validate(document).payload()
