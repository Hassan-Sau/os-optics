"""The recorded Michelson, and the numbers it must give, computed here without the Engine.

Catalog parts (repository seed catalog): a 3 mm plate splitter (R 50 / T 50 coating on S2, anti-reflection coating on S1) at the
origin and 45 deg, two silver mirrors (R 0.95 at 780 nm, normal incidence) 100 mm from it on +X and +Y, a 1 mW, 780 nm line of
0.001 nm with a 0.5 mm waist, s-polarized, 100 mm before the splitter. Both arms return through the splitter's coated
point and leave along -Y onto the plane ``out``:

    arm T (transmitted first):  S2 transmission, S1, mirror M1, S1, S2 reflection inside the plate, S1
    arm R (reflected first):    S2 reflection, mirror M2, S2 transmission, S1

Both then run down the same line, so their fields overlap on the whole plane. Arm T has the plate twice more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scipy.interpolate import PchipInterpolator

from declared_helpers import declared_project
from local_helpers import ALONG_Y, FOLD_X_TO_Y, SQRT_HALF, element, plane, project

BS_PLATE = "OS-OPT-BS-PLATE-D25P4-R50T50-650-900-AOI45-T3-HK9L-CA22P86"
FLAT_MIRROR_R95 = "OS-OPT-MIR-FLAT-D25P4-AG-450-12000-R95-AOI0T10-T4-CA22P86-HK9L"
WAVELENGTH_NM = 780.0
LINEWIDTH_NM = 0.001
WAIST_MM = 0.5
ARM_MM = 100.0
POWER_W = 1e-3
C_MM_PER_FS = 2.99792458e-4
AGAINST_X = (0.5, 0.5, -0.5, -0.5)
AGAINST_Y = (SQRT_HALF, SQRT_HALF, 0.0, 0.0)
#: Mirror steps of the recorded scan: lambda / 8 of mirror travel, a quarter wave of optical path difference, 9 positions.
SCAN_STEPS = 9


def michelson(catalog: dict, extra_mm: float = 0.0, *, targets: list[dict] | None = None) -> dict:
    """The Michelson of ``catalog``'s parts, its transmitted arm ``extra_mm`` longer (M1 moved along +X)."""
    version = catalog["catalog_version"]
    laser = {
        "instance_id": "SRC", "optical_role": "source",
        "custom_part": {"custom_part_key": "michelson-laser", "description": "780 nm line."},
        "nominal_pose": {"position_mm": [-100.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": WAVELENGTH_NM, "power_w": POWER_W, "waist_radius_mm": WAIST_MM,
                         "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 90.0},
                         "linewidth_nm": LINEWIDTH_NM},
    }
    components = [
        laser,
        element("BS1", "splitter", BS_PLATE, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=version),
        element("M1", "mirror", FLAT_MIRROR_R95, (ARM_MM + extra_mm, 0.0, 0.0), AGAINST_X, catalog_version=version),
        element("M2", "mirror", FLAT_MIRROR_R95, (0.0, ARM_MM, 0.0), AGAINST_Y, catalog_version=version),
    ]
    if targets is None:
        targets = [{"target_id": "coherent", "metric": "coherent_power_W", "observation_id": "out", "minimum": 4e-4, "unit": "W"},
                   {"target_id": "ordinary", "metric": "power_W", "observation_id": "out", "minimum": 0.0, "unit": "W"}]
    return project(components, [plane("out", (0.0, -50.0, 0.0), ALONG_Y)], targets)


def stated_phase_michelson(catalog: dict, extra_mm: float = 0.0, *, face_transmission_rad: str = "0") -> dict:
    """The same Michelson of declared copies of the catalog parts (the same R, T and group-delay bounds) whose coatings state
    their phases at the line in a flat table: the splitter's and the mirrors' are the same in both arms, and the
    anti-reflection face transmits with ``face_transmission_rad``, which arm T meets twice more than arm R."""
    document = declared_project(michelson(catalog, extra_mm), catalog)

    def table(**phases: str) -> dict:
        return {"wavelength_nm": ["400", "13000"], **{name: [value, value] for name, value in phases.items()}, "status": "DECLARED"}

    half_pi, pi = repr(math.pi / 2), repr(math.pi)
    for coating in document["model_coatings"]:
        if coating["kind"] == "METALLIC_REFLECTOR":
            coating["phase"] = table(reflection_s_rad=pi, reflection_p_rad=pi)
        elif coating["kind"] == "BEAMSPLITTER":
            coating["phase"] = table(reflection_s_rad=half_pi, reflection_p_rad=half_pi, transmission_s_rad="0", transmission_p_rad="0")
        else:
            coating["phase"] = table(reflection_s_rad=half_pi, reflection_p_rad=half_pi, transmission_s_rad=face_transmission_rad,
                                     transmission_p_rad=face_transmission_rad)
    return document


def _index_table(catalog: dict, material_id: str) -> PchipInterpolator:
    """The material's published index table as a monotone cubic over the wavelength (its declared interpolation)."""
    dispersion = next(m for m in catalog["materials"] if m["material_id"] == material_id)["dispersion"]
    return PchipInterpolator([float(x) for x in dispersion["wavelength_nm"]], [float(x) for x in dispersion["refractive_index"]])


def _model_surface(catalog: dict, sku: str, surface_id: str) -> dict:
    model = next(p for p in catalog["products"] if p["optical_sku"] == sku)["simulation_models"][0]
    return next(s for s in model["surfaces"] if s["surface_id"] == surface_id)


def _band(catalog: dict, sku: str, surface_id: str) -> tuple[float, float, float]:
    """``(R, T, group-delay bound in fs)`` of a surface's coating at the line."""
    coating_id = _model_surface(catalog, sku, surface_id)["coating_id"]
    coating = next(c for c in catalog["coatings"] if c["coating_id"] == coating_id)
    band = next(b for b in coating["bands"] if float(b["min_nm"]) <= WAVELENGTH_NM <= float(b["max_nm"]))
    return float(band["reflectance"]), float(band["transmittance"]), float(coating["group_delay_bound_fs"])


@dataclass(frozen=True)
class Expected:
    """The Michelson's numbers, computed from the catalog's data alone (at ``extra_mm`` = 0 for the paths)."""

    power_t_W: float  # arm T's power
    power_r_W: float  # arm R's power
    path_difference_mm: float  # OPL_T - OPL_R
    group_difference_mm: float  # the same with the group index
    coherence_length_mm: float
    bound_mm: float  # the sum of the bounds of the passes the arms do not share
    open_passes: tuple[str, ...]  # their short ids
    spatial_overlap: float

    @property
    def sum_W(self) -> float:
        return self.power_t_W + self.power_r_W

    def gamma(self, extra_mm: float = 0.0) -> float:
        """``exp(-(D / L_c)^2)`` at the group path difference with M1 ``extra_mm`` further out."""
        return math.exp(-(((self.group_difference_mm + 2.0 * extra_mm) / self.coherence_length_mm) ** 2))

    @property
    def delta(self) -> float:
        return 2.0 * self.bound_mm / (math.sqrt(math.pi) * self.coherence_length_mm)

    @property
    def largest_cross_term_W(self) -> float:
        """``2 sqrt(P_T P_R) eta_sp min(1, |gamma| + delta)``: the cross term's largest magnitude over every phase."""
        return 2.0 * math.sqrt(self.power_t_W * self.power_r_W) * self.spatial_overlap * min(1.0, self.gamma() + self.delta)

    def combined_W(self, extra_mm: float, stated_phase_rad: float = 0.0) -> float:
        """``P_T + P_R + 2 sqrt(P_T P_R) |gamma| eta_sp cos(k0 OPD + phase)``, where ``phase`` is what the coatings' stated
        phases leave between the arms."""
        phase = 2.0 * math.pi * (self.path_difference_mm + 2.0 * extra_mm) / (WAVELENGTH_NM * 1e-6) + stated_phase_rad
        return self.sum_W + 2.0 * math.sqrt(self.power_t_W * self.power_r_W) * self.gamma(extra_mm) * self.spatial_overlap * math.cos(phase)

    def bright_extra_mm(self) -> float:
        """The smallest mirror offset that puts ``k0 OPD`` on a whole number of 2 pi."""
        wavelength_mm = WAVELENGTH_NM * 1e-6
        return (-(self.path_difference_mm % wavelength_mm) + wavelength_mm) / 2.0


def expected(catalog: dict) -> Expected:
    """Snell's law and the catalog's index table for the plate; the coatings' published R, T and bounds for the powers."""
    r_bs, t_bs, bound_bs = _band(catalog, BS_PLATE, "S2")
    _, t_ar, bound_ar = _band(catalog, BS_PLATE, "S1")
    r_m, _, bound_m = _band(catalog, FLAT_MIRROR_R95, "S1")
    model = next(p for p in catalog["products"] if p["optical_sku"] == BS_PLATE)["simulation_models"][0]
    surfaces = model["surfaces"]
    thickness = float(surfaces[-1]["vertex_z_mm"]) - float(surfaces[0]["vertex_z_mm"])
    (material,) = model["media"]
    index = _index_table(catalog, material)
    n, n_g = float(index(WAVELENGTH_NM)), float(index(WAVELENGTH_NM) - WAVELENGTH_NM * index.derivative()(WAVELENGTH_NM))
    # the transmitted beam in the plate: Snell at 45 deg, its path L and how far along +X it leaves
    cos_i = math.sqrt(0.5)
    sin_t = cos_i / n
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    inside = thickness / cos_t
    normal = np.array([-cos_i, cos_i])
    along_x = float((inside * (np.array([1.0, 0.0]) / n + (cos_i / n - cos_t) * normal))[0])
    # arm T replaces (along_x) of air, out and back, by the path L in the plate, out and back; the common way out cancels
    excess = lambda value: 2.0 * value * inside - 2.0 * along_x  # noqa: E731
    # the Gaussian envelopes at the plane: arm R's way from the waist in air-equivalent length, arm T's that plus the plate's two
    # extra passes. A tilted plate's air-equivalent length is L / n across the plane of incidence and L cos^2(i) / (n cos^2(t))
    # within it (the beam is astigmatic), so the overlap is the geometric mean of the two axes'.
    z_r = math.pi * WAIST_MM ** 2 / (WAVELENGTH_NM * 1e-6)
    along_y = float((inside * (np.array([1.0, 0.0]) / n + (cos_i / n - cos_t) * normal))[1])
    way_r = 100.0 + 2.0 * ARM_MM + inside / n + 50.0 + along_y
    overlap = math.sqrt(_axis_overlap(way_r, excess(1.0 / n), z_r) * _axis_overlap(way_r, excess(cos_i * cos_i / (n * cos_t * cos_t)), z_r))
    return Expected(
        power_t_W=POWER_W * t_bs * t_ar * r_m * t_ar * r_bs * t_ar, power_r_W=POWER_W * r_bs * r_m * t_bs * t_ar,
        path_difference_mm=excess(n), group_difference_mm=excess(n_g),
        coherence_length_mm=2.0 * math.sqrt(math.log(2.0)) * (WAVELENGTH_NM * 1e-6) ** 2 / (math.pi * LINEWIDTH_NM * 1e-6),
        bound_mm=(2.0 * bound_bs + 2.0 * bound_ar + 2.0 * bound_m) * C_MM_PER_FS,
        open_passes=("BS1/S1:refract:+", "BS1/S1:refract:-", "BS1/S2:coating_reflection:+", "BS1/S2:coating_reflection:-", "M1/S1:reflect:+",
                     "M2/S1:reflect:+"),
        spatial_overlap=overlap)


def _axis_overlap(z_mm: float, dz_mm: float, z_r_mm: float) -> float:
    """``2 w_a w_b / (w_a^2 + w_b^2)``: what one transverse axis of two coaxial Gaussian beams, of radii ``w(z)`` and ``w(z + dz)``,
    gives ``(Integral sqrt(I_a I_b) dA)^2 / (P_a P_b)`` (the envelopes only: the two-beam model has no wavefront curvature across
    the plane)."""
    wa = math.sqrt(1.0 + (z_mm / z_r_mm) ** 2)
    wb = math.sqrt(1.0 + ((z_mm + dz_mm) / z_r_mm) ** 2)
    return 2.0 * wa * wb / (wa * wa + wb * wb)


def scan(expected_: Expected) -> list[float]:
    """The recorded scan's mirror offsets from a bright position: ``SCAN_STEPS`` steps of lambda / 8."""
    start = expected_.bright_extra_mm()
    return [start + step * WAVELENGTH_NM * 1e-6 / 8.0 for step in range(SCAN_STEPS)]
