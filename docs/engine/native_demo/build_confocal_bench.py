"""Build the polarization-isolated confocal reflectance bench (632.8 nm) through the public JSON API only.

This is how a customer's Agent composes a non-template design: it discovers native component classes
(``native_components``), builds the parts OpenSci does not sell as models of the Project (``build_native_model``),
queries the public catalog for orderable laboratory optics (``catalog_products`` / ``catalog_product`` /
``catalog_materials``), places everything, attaches observations and targets, and checks the Project
(``check_project``). Nothing here imports engine internals; the only physics computed locally is a paraxial y-nu trace of
the chosen catalog lenses (from their published prescription and materials) to put the sample and the confocal
pinhole at their foci at 632.8 nm.

Run from the repository root (the engine environment)::

    uv run --no-sync python docs/engine/native_demo/build_confocal_bench.py \
        --catalog .cache/public-catalog/public_optical_catalog.2026-10-28.1.json \
        --out OUT_DIR/confocal_bench.project.json [--run RUN_DIR]

``--catalog`` installs that catalog release in memory (otherwise the active local release is used); ``--run`` also
simulates the Project and prints its targets (needs a runtime whose ``capabilities()["project_features"]`` lists
``optical_models``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from os_optics.local import json_api

WAVELENGTH_NM = 632.8
BAND = [600, 700]
VISIBLE = [450, 700]  # the achromat is corrected at the F, d and C lines (486.1, 587.6, 656.3 nm)
SQRT_HALF = math.sqrt(0.5)


def call(function, *args, **kwargs) -> dict:
    reply = function(*args, **kwargs)
    if not reply.get("ok"):
        raise SystemExit(f"{function.__name__} refused:\n{json.dumps(reply.get('error'), indent=2)}")
    return reply


# ------------------------------------------------------------------------------------------------ frames / poses


def quaternion(z_axis, x_axis) -> list[float]:
    """Unit quaternion (w, x, y, z) of the rotation taking local +Z to ``z_axis`` and local +X to ``x_axis``."""

    zx, zy, zz = z_axis
    xx, xy, xz = x_axis
    yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx
    m = ((xx, yx, zx), (xy, yy, zy), (xz, yz, zz))
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    if q[0] < 0:
        q = tuple(-v for v in q)
    return [round(v, 12) for v in q]


UP = (0.0, 0.0, 1.0)
ALONG_PLUS_X = quaternion((1, 0, 0), (0, 1, 0))      # beam along +X, local X horizontal (the reference designs' convention)
ALONG_MINUS_Y = quaternion((0, -1, 0), (1, 0, 0))
ALONG_PLUS_Y = quaternion((0, 1, 0), (-1, 0, 0))
FACE_SW = quaternion((-SQRT_HALF, -SQRT_HALF, 0), (SQRT_HALF, -SQRT_HALF, 0))  # face normal toward -X-Y: +X <-> -Y, -X <-> +Y
FACE_PLUS_Y = quaternion((0, 1, 0), (-1, 0, 0))     # a reflector facing light that travels -Y
QWP_45 = quaternion((1, 0, 0), (0, SQRT_HALF, SQRT_HALF))  # beam along +X, fast axis 45 deg between horizontal and vertical


def pose(position, q) -> dict:
    return {"position_mm": [float(v) for v in position], "quaternion_wxyz": q}


# ------------------------------------------------------------------------------------------------ catalog choices


def orderable(category: str, form: str, want) -> dict:
    rows = call(json_api.catalog_products, category=category, form=form, wavelength_nm=str(WAVELENGTH_NM), summary=True)["products"]
    rows = sorted((r for r in rows if r["commercial_readiness"] == "ORDERABLE" and want(r)),
                  key=lambda r: (-float(r.get("transmittance") or r.get("reflectance") or 0), r["optical_sku"]))
    if not rows:
        raise SystemExit(f"no orderable {form} matches")
    return rows[0]


def covers_aoi(sku: str, angle_deg: float) -> bool:
    product = call(json_api.catalog_product, sku)["product"]
    coatings = {c["coating_id"]: c for c in call(json_api.catalog_coatings)["coatings"]}
    ids = {s["coating_id"] for m in product["simulation_models"] for s in m.get("surfaces", []) if "coating_id" in s}
    return bool(ids) and all(float(coatings[i]["valid_aoi_deg"]["min_deg"]) <= angle_deg <= float(coatings[i]["valid_aoi_deg"]["max_deg"])
                             for i in ids)


def sellmeier_glasses(sku: str) -> bool:
    """Whether every glass of a catalog lens is published as a Sellmeier formula (what ``lens_focus`` evaluates)."""

    kinds = {m["material_id"]: m["dispersion"]["kind"] for m in call(json_api.catalog_materials)["materials"]}
    model = call(json_api.catalog_product, sku)["product"]["simulation_models"][0]
    return all(kinds[m] == "SELLMEIER" for m in model["media"] if m != "AIR")


def sellmeier(material: dict, wavelength_nm: float) -> float:
    dispersion = material["dispersion"]
    if dispersion["kind"] != "SELLMEIER":
        raise SystemExit(f"{material['material_id']}: this script evaluates Sellmeier materials only")
    lam2 = (wavelength_nm / 1000) ** 2
    return math.sqrt(1 + sum(float(b) * lam2 / (lam2 - float(c)) for b, c in zip(dispersion["b"], dispersion["c_um2"])))


def lens_focus(sku: str, wavelength_nm: float) -> tuple[float, float, float]:
    """(effective focal length, back focal length, last vertex z) of a catalog SURFACE_SEQUENCE lens by a y-nu trace."""

    model = call(json_api.catalog_product, sku)["product"]["simulation_models"][0]
    materials = {m["material_id"]: m for m in call(json_api.catalog_materials)["materials"]}
    media = ["AIR", *model["media"], "AIR"]
    n = [1.0 if m == "AIR" else sellmeier(materials[m], wavelength_nm) for m in media]
    y, nu, z_prev = 1.0, 0.0, None  # paraxial ray: height, reduced angle n u
    for k, surface in enumerate(model["surfaces"]):
        z = float(surface["vertex_z_mm"])
        if z_prev is not None:
            y += (z - z_prev) * nu / n[k]
        if "radius_mm" in surface:
            nu -= y * (n[k + 1] - n[k]) / float(surface["radius_mm"])
        z_prev = z
    return -1.0 / nu, -y / nu, z_prev


# ---------------------------------------------------------------------------------------------- native parts


def native(class_id: str, parameters: dict) -> dict:
    return call(json_api.build_native_model, class_id, parameters)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalog", help="install this public catalog release file first")
    parser.add_argument("--out", required=True, help="write the Project JSON here")
    parser.add_argument("--run", metavar="RUN_DIR", help="also run the Project and write the run directory")
    args = parser.parse_args(argv)
    if args.catalog:
        call(json_api.install_catalog, Path(args.catalog).read_bytes())
    capabilities = call(json_api.capabilities)
    catalog_version = capabilities["catalog_version"]
    classes = {c["class_id"] for c in call(json_api.native_components)["classes"]}
    assert {"polarizer.linear", "compound.keplerian_telescope", "aperture.circular", "splitter.plate", "mirror.flat"} <= classes

    # --- orderable laboratory optics from the catalog
    hwp = orderable("WAVEPLATE", "HALF_WAVE_PLATE", lambda r: r.get("outer_diameter_mm") == "25.4")
    qwp = orderable("WAVEPLATE", "QUARTER_WAVE_PLATE", lambda r: r.get("outer_diameter_mm") == "25.4")
    fold = orderable("MIRROR", "FLAT_MIRROR", lambda r: r.get("outer_diameter_mm") == "25.4" and covers_aoi(r["optical_sku"], 45.0))
    objective = orderable("LENS", "ACHROMATIC_DOUBLET", lambda r: r.get("outer_diameter_mm") == "25.4" and r.get("focal_length_mm") == "50"
                          and sellmeier_glasses(r["optical_sku"]))
    tube = orderable("LENS", "ACHROMATIC_DOUBLET", lambda r: r.get("outer_diameter_mm") == "25.4" and r.get("focal_length_mm") == "150"
                     and sellmeier_glasses(r["optical_sku"]))

    # --- native parts: what the catalog does not offer (or offers only for simulation)
    ar = {"reflectance": "0.005", "valid_aoi_deg": [0, 10]}
    polarizer = native("polarizer.linear", {"model_id": "clean-up-polarizer", "transmittance": "0.9", "extinction_ratio": "10000",
                                            "diameter_mm": "25.4", "valid_band_nm": BAND, "valid_aoi_deg": [0, 10]})
    expander = native("compound.keplerian_telescope", {
        "model_id": "spatial-filter-5x",
        "lens1": {"class_id": "lens.singlet", "parameters": {"focal_length_mm": "25", "design_wavelength_nm": str(WAVELENGTH_NM),
                                                             "material": "N-BK7", "diameter_mm": "12.7", "form": "PLANO_EXIT", "ar_coating": ar,
                                                             "valid_band_nm": BAND}},
        "lens2": {"class_id": "lens.achromat", "parameters": {"focal_length_mm": "125", "crown_material": "N-BK7", "flint_material": "N-SF5",
                                                              "diameter_mm": "25.4", "ar_coating": ar, "valid_band_nm": VISIBLE}},
        "design_wavelength_nm": str(WAVELENGTH_NM), "focal_stop_diameter_mm": "0.04"})
    iris = native("aperture.circular", {"model_id": "iris-6mm", "opening_diameter_mm": "6"})
    pbs = native("splitter.plate", {"model_id": "polarizing-plate-splitter", "diameter_mm": "25.4", "reflectance_s": "0.995",
                                    "reflectance_p": "0.005", "transmittance_s": "0.005", "transmittance_p": "0.995",
                                    "valid_band_nm": BAND, "valid_aoi_deg": [40, 50]})
    sample = native("mirror.flat", {"model_id": "sample-r50", "display_name": "Reflective sample (declared R = 0.5)", "diameter_mm": "10",
                                    "clear_aperture_diameter_mm": "10", "reflectance": "0.5", "coating_kind": "DIELECTRIC_HIGH_REFLECTOR",
                                    "valid_band_nm": BAND, "valid_aoi_deg": [0, 10], "substrate_thickness_mm": "1"})
    pinhole = native("aperture.circular", {"model_id": "confocal-pinhole-50um", "opening_diameter_mm": "0.05", "thickness_mm": "0.1"})
    natives = [polarizer, expander, iris, pbs, sample, pinhole]

    # --- layout (mm; beam plane z = 0; +Z up)
    _f_obj, bfl_obj, last_obj = lens_focus(objective["optical_sku"], WAVELENGTH_NM)
    _f_tube, bfl_tube, last_tube = lens_focus(tube["optical_sku"], WAVELENGTH_NM)
    expander_x = -350.0
    expander_end = expander_x + float(expander["optical_model"]["model"]["envelope"]["z_max_mm"])
    sample_y = -40.0 - last_obj - bfl_obj
    pinhole_y = 60.0 + last_tube + bfl_tube

    def product(instance_id, role, sku, position, q):
        return {"instance_id": instance_id, "optical_role": role, "product": {"optical_sku": sku, "catalog_version": catalog_version},
                "nominal_pose": pose(position, q)}

    def placed(build, instance_id, position, q):
        component = dict(build["component_template"])
        component["instance_id"] = instance_id
        component["nominal_pose"] = pose(position, q)
        return component

    def instrument(instance_id, key, description, position, q):
        return {"instance_id": instance_id, "optical_role": "detector", "custom_part": {"custom_part_key": key, "description": description},
                "nominal_pose": pose(position, q)}

    components = [
        {"instance_id": "LASER", "optical_role": "source", "custom_part": {"custom_part_key": "hene-laser", "description": "HeNe laser 632.8 nm"},
         "nominal_pose": pose((-440, 0, 0), ALONG_PLUS_X),
         "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": WAVELENGTH_NM, "power_w": 0.001, "waist_radius_mm": 0.4,
                          "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}},
        placed(polarizer, "POL", (-410, 0, 0), ALONG_PLUS_X),
        product("HWP", "waveplate", hwp["optical_sku"], (-390, 0, 0), ALONG_PLUS_X),
        placed(expander, "EXPANDER", (expander_x, 0, 0), ALONG_PLUS_X),
        placed(iris, "IRIS", (-170, 0, 0), ALONG_PLUS_X),
        placed(pbs, "PBS", (-100, 0, 0), FACE_SW),
        product("QWP", "waveplate", qwp["optical_sku"], (-60, 0, 0), QWP_45),
        product("FOLD", "mirror", fold["optical_sku"], (0, 0, 0), FACE_SW),
        product("OBJECTIVE", "lens", objective["optical_sku"], (0, -40, 0), ALONG_MINUS_Y),
        placed(sample, "SAMPLE", (0, sample_y, 0), FACE_PLUS_Y),
        product("TUBE_LENS", "lens", tube["optical_sku"], (-100, 60, 0), ALONG_PLUS_Y),
        placed(pinhole, "PINHOLE", (-100, pinhole_y, 0), ALONG_PLUS_Y),
        instrument("DETECTOR", "photodiode", "Photodiode behind the confocal pinhole", (-100, pinhole_y + 5, 0), ALONG_PLUS_Y),
        instrument("MONITOR", "monitor-photodiode", "Power monitor on the splitter's forward leak", (-100, -60, 0), ALONG_MINUS_Y),
    ]
    identity = {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    observations = [
        {"observation_id": "EXPANDED", "pose": pose((expander_end + 10, 0, 0), ALONG_PLUS_X), "aperture": {"shape": "circle", "radius_mm": 12.5},
         "absorbing": False, "medium": "air"},
        {"observation_id": "PINHOLE_APPROACH", "pose": pose((-100, pinhole_y - 0.5, 0), ALONG_PLUS_Y),
         "aperture": {"shape": "circle", "radius_mm": 5.0}, "absorbing": False, "medium": "air"},
        {"observation_id": "DETECTOR_FACE", "device_instance_id": "DETECTOR", "pose": identity, "aperture": {"shape": "circle", "radius_mm": 5.0},
         "absorbing": True, "medium": "air"},
        {"observation_id": "MONITOR_FACE", "device_instance_id": "MONITOR", "pose": identity, "aperture": {"shape": "circle", "radius_mm": 5.0},
         "absorbing": True, "medium": "air"},
    ]
    targets = [
        # the polarization leak of the returning beam crosses this plane backwards: select the first, forward crossing
        {"target_id": "expanded_beam_radius", "metric": "beam_radius_mm", "observation_id": "EXPANDED", "target_value": 2.0,
         "tolerance_abs": 0.1, "unit": "mm",
         "parameters": [{"key": "path_id", "value": "LASER#0"}, {"key": "crossing_index", "value": 0}]},
        {"target_id": "confocal_spot_radius", "metric": "beam_radius_mm", "observation_id": "PINHOLE_APPROACH", "maximum": 0.03, "unit": "mm"},
        {"target_id": "detected_power", "metric": "power_W", "observation_id": "DETECTOR_FACE", "minimum": 0.0001, "unit": "W"},
        {"target_id": "monitor_power", "metric": "power_W", "observation_id": "MONITOR_FACE", "minimum": 0.000001, "unit": "W"},
    ]

    def merged(key: str, field: str) -> list[dict]:
        out: dict[str, dict] = {}
        for build in natives:
            items = ((build.get(key) or []) if key != "optical_models"
                     else [*(build.get("member_models") or []), build["optical_model"]])
            for item in items:
                if item[field] in out and out[item[field]] != item:
                    raise SystemExit(f"two different {key} share the id {item[field]}")
                out.setdefault(item[field], item)
        return [out[k] for k in sorted(out)]

    project = {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "LAB_ASSEMBLY",
               "components": components, "observations": observations, "targets": targets,
               "optical_models": merged("optical_models", "model_id"), "model_materials": merged("model_materials", "material_id"),
               "model_coatings": merged("model_coatings", "coating_id")}
    Path(args.out).write_text(json.dumps(project, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"written": args.out, "catalog_version": catalog_version,
                      "catalog_parts": {"HWP": hwp["optical_sku"], "QWP": qwp["optical_sku"], "FOLD": fold["optical_sku"],
                                        "OBJECTIVE": objective["optical_sku"], "TUBE_LENS": tube["optical_sku"]},
                      "objective_bfl_mm": round(bfl_obj, 6), "tube_bfl_mm": round(bfl_tube, 6), "sample_y_mm": round(sample_y, 6),
                      "pinhole_y_mm": round(pinhole_y, 6),
                      "expander": expander["design"]}, indent=2))
    checked = json_api.check_project(project)
    print(json.dumps({"check_project": "ok" if checked["ok"] else checked["error"]}, indent=2))
    if args.run:
        if not checked["ok"]:
            return 1
        run = json_api.run_project(project, output_dir=args.run)
        if not run["ok"]:
            print(json.dumps(run["error"], indent=2))
            return 1
        summary = run["summary"]
        print(json.dumps({"status": run["status"], "targets": summary["targets"],
                          "issues": [(i.get("code"), i.get("message")) for i in summary["issues"]],
                          "model_scope_issues": summary["model_scope_issues"], "power_budget": summary["power_budget"]},
                         indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
