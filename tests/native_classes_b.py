"""Parameters of the native classes added in phase B of the native library (fixture data), shared by test_native_library.py (every class
builds; Pyodide parity) and test_native_library_classes_b.py (physics)."""

from __future__ import annotations

import copy

from test_tabulated_thin_lens import EVIDENCE, ROWS

BAND = [400, 1100]

CLASSES_B = {
    "lens.axicon": {"cone_angle_deg": "10", "material": "N-BK7", "diameter_mm": "25.4", "design_wavelength_nm": "780"},
    "prism.wedge": {"wedge_angle_deg": "5", "material": "N-BK7", "diameter_mm": "25.4", "design_wavelength_nm": "780"},
    "mirror.hollow_roof": {"aperture_width_mm": "20", "aperture_height_mm": "20", "reflectance": "1", "valid_band_nm": BAND},
    "retroreflector.hollow_corner": {"mirror_edge_mm": "20", "reflectance": "1", "valid_band_nm": BAND},
    "filter.variable_nd": {"law": "LINEAR_ROTARY", "od_start": "0.04", "od_end": "2", "arc_end_deg": "270", "diameter_mm": "50",
                           "coated_inner_radius_mm": "5", "uncoated_transmittance": "0.92", "valid_band_nm": [400, 700]},
    "lens.tabulated_thin": {"rows": [{"wavelength_nm": f"{w:g}", "focal_length_mm": f"{f:g}", "transmittance": f"{t:g}"} for w, f, t in ROWS],
                            "clear_aperture_diameter_mm": "20", "diameter_mm": "25.4", "max_field_angle_deg": "5",
                            "supported_metrics": [{"metric": "waist_distance_x_mm", "rel_error": "0.005", "basis": "fixture"},
                                                  {"metric": "power_W", "rel_error": "0.05", "basis": "fixture"}],
                            "evidence": copy.deepcopy(EVIDENCE)},
}
